import streamlit as st
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import yaml

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qdp.artifacts.checksums import verify_checksums
from qdp.artifacts.save_load import load_joblib, load_json
from qdp.config.schema import validate_config
from qdp.data.schema import make_record
from qdp.models.predictor import InferencePreprocessor, Predictor
from qdp.semantic.encoder import SemanticEncoder

# Streamlit Page Config
st.set_page_config(
    page_title="Question Difficulty Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# Function from predict.py
def load_lexicons(root):
    def read(name):
        with (root / "config" / name).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    code = read("code_signatures.yaml")
    concept = read("concept_lexicon.yaml")
    return {
        "task": read("task_lexicon.yaml"),
        "bloom": read("bloom_lexicon.yaml"),
        "concepts": concept["concepts"],
        "technical_terms": concept["technical_terms"],
        "algorithm": read("algorithm_complexity.yaml")["weights"],
        "constraint_patterns": read("constraint_patterns.yaml")["patterns"],
        "negation": read("negation_lexicon.yaml")["terms"],
        "code_signatures": code["patterns"],
        "branch_keywords": code["branch_keywords"],
        "loop_keywords": code["loop_keywords"],
        "mcq_markers": read("mcq_markers.yaml")["markers"],
    }

@st.cache_resource
def load_model():
    root = ROOT
    run = (root / "artifacts/latest").resolve()
    manifest = load_json(run / "run_manifest.json")
    
    cfg = load_json(run / "resolved_config.json")
    validate_config(cfg)
    schema = load_json(run / "feature_schema.json")
    mode = load_json(run / "production_feature_mode.json")
    rare = load_joblib(run / "rare_word_df.joblib")
    
    tfidf_path = run / "tfidf.joblib"
    tfidf = load_joblib(tfidf_path) if tfidf_path.exists() else None
    
    threshold_path = run / "tfidf_threshold.json"
    threshold = load_json(threshold_path)["threshold"] if threshold_path.exists() else 0.0
    
    imputer = load_joblib(run / "imputer.joblib")
    onehot = load_joblib(run / "onehot.joblib")
    selector = load_joblib(run / "feature_selector.joblib")
    model = load_joblib(run / "calibrator.joblib") if manifest.get("calibrated") else load_joblib(run / "model.joblib")
    
    nlp = None
    profile = manifest["dependency_profile"]
    if profile not in {"P2", "P4"}:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        
    embedder = None
    pca_path = run / "semantic_pca.joblib"
    pca = load_joblib(pca_path) if pca_path.exists() else None
    if pca is not None:
        encoder_meta = load_json(run / "semantic_encoder.json")
        embedder = SemanticEncoder(
            encoder_meta["name"],
            root / "data/features/embedding_cache",
            cfg["runtime"]["embedding_batch_size"],
            max_seq_length=encoder_meta["max_seq_length"],
            revision=encoder_meta.get("revision"),
        ).load()
        
    preprocessing_meta = load_json(run / "preprocessing_manifest.json")
    preprocessor = InferencePreprocessor(
        cfg,
        load_lexicons(root),
        nlp,
        embedder,
        rare,
        tfidf,
        threshold,
        imputer,
        onehot,
        selector,
        pca,
        preprocessing_meta["numeric_columns"],
        preprocessing_meta["engineered_columns"],
        schema,
        bool(mode["use_final_selector"]),
    )
    metadata = {
        "calibrated": bool(manifest.get("calibrated", False)),
        "model_version": "qdp_xgb_v1" if manifest["selected_hyperparameters"].get("production") else "qdp_model_v1",
        "feature_schema_version": schema["version"],
        "dependency_profile": profile,
    }
    return Predictor(preprocessor, model, metadata)


# UI Layout
st.title("🧠 Question Difficulty Predictor")
st.markdown("Analyze programming and technical questions to determine their difficulty level (Easy, Medium, Hard).")

with st.spinner("Loading model and artifacts..."):
    try:
        predictor = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model_loaded = False

if model_loaded:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Input Question")
        question_text = st.text_area(
            "Paste the question text here:",
            height=250,
            placeholder="Write a function to reverse a linked list...\n\nExample:\nInput: head = [1,2,3,4,5]\nOutput: [5,4,3,2,1]",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("Analyze Difficulty", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Model Info")
        st.info(
            f"**Model Version:** {predictor.metadata['model_version']}\n\n"
            f"**Calibrated:** {predictor.metadata['calibrated']}\n\n"
            f"**Schema Version:** {predictor.metadata['feature_schema_version']}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    if analyze_btn and question_text.strip():
        st.divider()
        with st.spinner("Analyzing question features and predicting difficulty..."):
            try:
                record = make_record(question_text.strip(), None, "ui_query")
                result = predictor.predict_record(record)
                
                # Determine colors based on prediction
                diff = result["predicted_label"]
                if diff == "Easy":
                    color = "#00CC96"
                elif diff == "Moderate":
                    color = "#FFA15A"
                else:
                    color = "#EF553B"
                
                st.markdown(f"<h2 style='text-align: center; color: {color};'>Predicted Difficulty: {diff}</h2>", unsafe_allow_html=True)
                
                st.write("")
                st.write("")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.subheader("Confidence Scores")
                    probs = result["probabilities"]
                    
                    df_probs = pd.DataFrame({
                        "Difficulty": list(probs.keys()),
                        "Probability": list(probs.values())
                    })
                    
                    # Chart
                    fig = px.bar(
                        df_probs, 
                        x="Probability", 
                        y="Difficulty", 
                        orientation='h',
                        color="Difficulty",
                        color_discrete_map={
                            "Easy": "#00CC96",
                            "Moderate": "#FFA15A",
                            "Hard": "#EF553B"
                        },
                        range_x=[0, 1]
                    )
                    fig.update_layout(
                        showlegend=False, 
                        height=250, 
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_res2:
                    st.subheader("Prediction Details")
                    
                    st.metric("Model Confidence", f"{max(probs.values())*100:.1f}%")
                    st.caption(f"The model is most confident that this question is **{diff}**.")
                    
                    with st.expander("View Raw Output JSON"):
                        st.json(result)
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")

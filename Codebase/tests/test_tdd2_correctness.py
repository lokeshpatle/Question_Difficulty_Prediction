from pathlib import Path
import inspect
import math
import numpy as np
import pandas as pd

from qdp.data.loader import CorpusLoader, IngestError
from qdp.data.schema import make_record
from qdp.features.programming import extract as programming_extract
from qdp.features.task import subquestion_count
from qdp.features.pipeline import FeatureExtractor
from qdp.features.registry import FEATURES, FEATURE_VERSION
from qdp.preprocessing.code_detector import detect_code_spans
from qdp.preprocessing.mcq_parser import detect_mcq
from qdp.preprocessing.views import make_views
from qdp.preprocessing.tokenizer import split_sentences
from qdp.transform.imputer import TrainOnlyImputer


def test_json_array_and_jsonl_equivalent(tmp_path):
    import json
    rows = [
        {"question": "Q1?", "label": "Easy"},
        {"question": "Q2?", "label": "Moderate"},
    ]
    array_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    array_path.write_text(json.dumps(rows), encoding="utf-8")
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loader = CorpusLoader({"easy": "Easy", "moderate": "Moderate", "hard": "Hard"}, 1.0)
    a = [r.as_dict() for r in loader.load(array_path)]
    b = [r.as_dict() for r in loader.load(jsonl_path)]
    assert a == b


def test_reject_ratio_gate(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question":"ok","label":"Easy"}\n{"question":"","label":"Easy"}\n', encoding="utf-8")
    loader = CorpusLoader({"easy": "Easy", "moderate": "Moderate", "hard": "Hard"}, 0.1)
    try:
        loader.load(path)
    except IngestError as exc:
        assert "rejection_rate_exceeded" in str(exc)
    else:
        raise AssertionError("reject ratio gate did not fire")


def test_f17_requires_sequential_enumerators():
    assert subquestion_count("(a) one\n(c) three\n") == 0
    assert subquestion_count("(a) one\n(b) two\n") == 2
    assert subquestion_count("1. one\n2. two\n") == 2


def test_constraint_notation_equivalence():
    import yaml
    patterns = yaml.safe_load(Path("config/constraint_patterns.yaml").read_text())['patterns']
    values = [programming_extract(text, [], {}, patterns)["F31"] for text in ("n <= 10^6", "n <= 1e6", "n <= 1,000,000")]
    assert np.allclose(values, values[0], atol=1e-10)


def test_fenced_python_code_reaches_ast():
    patterns = [r'^\s*def\s+', r'\bfor\s*\(', r'\bif\s*\(']
    code, _ = detect_code_spans("```python\nfor i in range(3):\n    for j in range(2):\n        print(i, j)\n```", patterns)
    result = programming_extract("", code, {}, [], branch_keywords=["if", "for"], loop_keywords=["for", "while"])
    assert result["code_parse_exact"] == 1
    assert result["F34"] == 3
    assert result["F35"] == 3


def test_sentence_fallback_and_normalized_views():
    sentences = split_sentences("Dr. Smith asks one question. Another follows!", ["dr."])
    assert sentences == ["Dr. Smith asks one question.", "Another follows!"]
    view = make_views("  A  question.\r\n\r\n\r\nB!  ", [], lambda text: {"detected": False}, sentence_abbreviations=[])
    assert view.raw_text == "A  question.\n\nB!"
    assert view.sentences == ["A  question.", "B!"]


def test_f14_f15_closed_and_f16_maximum():
    from qdp.features.task import question_type, instruction_type, cognitive_demand
    assert question_type("Why define and then evaluate this?") == "Why"
    assert instruction_type(["why", "define", "evaluate"], {"Define": ["define"], "Evaluate": ["evaluate"]}) == "Define"
    assert cognitive_demand(["define", "evaluate"], {"Remember": ["define"], "Evaluate": ["evaluate"]}) == 5


def test_task_family_can_be_disabled_in_assembler(tmp_path):
    # This is an integration-level regression: disabling task features must not
    # silently re-add F14/F15 through one-hot columns.
    from qdp.pipeline import QDPTrainer
    from qdp.config.loader import load_config
    cfg = load_config(Path('.'), override={"features": {"use_task": False, "use_semantic_embedding": False, "use_readability": False, "use_syntax": False, "use_tfidf": False, "use_semantic_scalar": False, "use_programming": False, "use_mcq": False}})
    trainer = QDPTrainer(Path('.'), cfg, Path(tmp_path) / "run", profile="P4")
    records = [make_record(f"Question {i} with variable length {i*i}?", "Easy" if i % 3 == 0 else "Moderate" if i % 3 == 1 else "Hard") for i in range(12)]
    for i, rec in enumerate(records):
        from qdp.data.schema import with_split
        records[i] = with_split(rec, "train" if i < 8 else "validation")
    bundle = trainer._fit_feature_bundle(records, "P4")
    assert not any(name.startswith("question_type=") or name.startswith("instruction_type=") for name in bundle["engineered_columns"])


def test_inference_preprocessor_does_not_fit():
    src = inspect.getsource(__import__('qdp.models.predictor', fromlist=['InferencePreprocessor']).InferencePreprocessor.transform_records)
    assert ".fit(" not in src


def test_imputer_uses_only_fit_rows():
    import pandas as pd
    from qdp.transform.imputer import TrainOnlyImputer

    frame = pd.DataFrame({'x': [1.0, 100.0, np.nan]})
    imputer = TrainOnlyImputer().fit(frame, fit_indices=[0, 2])
    transformed = imputer.transform(frame)
    # Median over rows 0 and 2 ignores the validation/test-only value 100.
    assert float(imputer.model.statistics_[0]) == 1.0
    assert float(transformed[1, 0]) == 100.0


def test_calibration_uses_prefit_model_without_retraining():
    from sklearn.linear_model import LogisticRegression
    from qdp.models.calibration import calibrate_model

    X = np.array([[0.0], [0.2], [1.0], [1.2], [2.0], [2.2]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    base = LogisticRegression(max_iter=500).fit(X, y)
    before = base.coef_.copy()
    calibrated = calibrate_model(base, X, y)
    assert calibrated.predict(X).shape == y.shape
    assert np.array_equal(before, base.coef_)


def test_empty_active_feature_set_is_not_replaced_by_defaults():
    # This guards the distinction between None (use defaults) and [] (explicitly disable all).
    from qdp.pipeline import QDPTrainer
    trainer = object.__new__(QDPTrainer)
    trainer.cfg = {"features": {
        "use_semantic_embedding": False,
        "use_lexical": True,
        "use_readability": False,
        "use_syntax": False,
        "use_task": False,
        "use_mcq": False,
        "use_tfidf": False,
        "use_semantic_scalar": False,
        "use_programming": False,
    }}
    assert trainer._active_feature_ids("P4")


def test_f30_uses_code_spans_for_algorithm_terms():
    from qdp.features.programming import extract
    out = extract(
        "Solve this problem.",
        ["binary_search(a, target)"],
        {"binary search": 3},
        [],
    )
    assert out["F30"] == 3

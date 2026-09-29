from __future__ import annotations

from qdp.utils.hashing import stable_json_hash

FEATURES = [
    ("F01", "word_count", "lexical"),
    ("F02", "avg_sentence_length", "lexical"),
    ("F03", "avg_word_length", "lexical"),
    ("F04", "type_token_ratio", "lexical"),
    ("F05", "rare_word_ratio", "lexical"),
    ("F06", "flesch_reading_ease", "readability"),
    ("F07", "flesch_kincaid_grade", "readability"),
    ("F08", "gunning_fog", "readability"),
    ("F09", "clause_count", "syntax"),
    ("F10", "avg_noun_phrase_length", "syntax"),
    ("F11", "parse_tree_depth", "syntax"),
    ("F12", "dependency_complexity", "syntax"),
    ("F13", "negation_count", "syntax"),
    ("F14", "question_type", "task"),
    ("F15", "instruction_type", "task"),
    ("F16", "cognitive_demand", "task"),
    ("F17", "subquestion_count", "task"),
    ("F18", "option_count", "mcq"),
    ("F19", "avg_option_length", "mcq"),
    ("F20", "option_length_variance", "mcq"),
    ("F21", "stem_option_similarity_mean", "mcq"),
    ("F22", "option_option_similarity_mean", "mcq"),
    ("F23", "mean_tfidf", "tfidf"),
    ("F24", "max_tfidf", "tfidf"),
    ("F25", "high_tfidf_term_count", "tfidf"),
    ("F26", "concept_count", "semantic_scalar"),
    ("F27", "concept_density", "semantic_scalar"),
    ("F28", "technical_term_ratio", "semantic_scalar"),
    ("F29", "semantic_similarity", "semantic_scalar"),
    ("F30", "algorithm_or_ds_complexity", "programming"),
    ("F31", "max_input_size_log", "programming"),
    ("F32", "constraint_count", "programming"),
    ("F33", "code_length", "programming"),
    ("F34", "cyclomatic_complexity", "programming"),
    ("F35", "loop_nesting_complexity", "programming"),
]

FAMILIES = {}
for fid, _, family in FEATURES:
    FAMILIES.setdefault(family, []).append(fid)

Q_TYPES = ["What", "Why", "How", "Which", "When", "Where", "YesNo", "Other"]
I_TYPES = ["Define", "Explain", "Calculate", "Compare", "Analyze", "Design", "Prove", "Describe", "Identify", "Other"]
INDICATORS = ["mcq_detected", "code_detected", "input_size_detected", "parse_ok", "multi_segment"]

FEATURE_VERSION = "fs_v3"
EXTRACTOR_VERSION = "extractor_v3"


def schema_hash(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("schema_hash", None)
    return stable_json_hash(clean)

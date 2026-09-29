from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qdp.features import lexical, mcq, programming, readability, semantic_scalar, syntax, task, tfidf
from qdp.features.registry import FEATURES, FAMILIES, EXTRACTOR_VERSION
from qdp.preprocessing.mcq_parser import detect_mcq
from qdp.preprocessing.views import TextViews, make_views


@dataclass
class FeatureContext:
    record_id: str
    views: TextViews
    doc: object | None
    semantic: dict[str, np.ndarray]


class FeatureExtractor:
    VERSION = EXTRACTOR_VERSION

    def __init__(self, cfg, lexicons, nlp=None, embedder=None, cache_root: Path | None = None):
        self.cfg = cfg
        self.lex = lexicons
        self.nlp = nlp
        self.embedder = embedder
        self.cache_root = Path(cache_root) if cache_root else None
        self.rare_df: dict[str, int] = {}
        self.tfidf = None
        self.tfidf_threshold = 0.0
        self._views: dict[str, TextViews] = {}
        self._docs: dict[str, object | None] = {}

    def _mcq(self, text):
        return detect_mcq(text, self.lex.get("mcq_markers", []))

    def _make_view(self, record):
        if record.question_id in self._views:
            return self._views[record.question_id]
        view = make_views(
            record.question,
            self.lex.get("code_signatures", []),
            self._mcq,
            max_chars=int(self.cfg["preprocess"]["max_chars"]),
            semantic_piece_limit=256,
            encoder=self.embedder,
            sentence_abbreviations=self.cfg["preprocess"].get("sentence_abbreviations", []),
        )
        self._views[record.question_id] = view
        return view

    def _parse_many(self, texts, ids):
        if self.nlp is None:
            return [None] * len(texts)
        docs = [self._docs.get(qid) for qid in ids]
        missing_positions = [index for index, doc in enumerate(docs) if doc is None]
        if missing_positions:
            missing_texts = [texts[index] for index in missing_positions]
            parsed = list(
                self.nlp.pipe(
                    missing_texts,
                    batch_size=64,
                    n_process=int(self.cfg["runtime"].get("spacy_n_process", 1)),
                )
            )
            for index, doc in zip(missing_positions, parsed):
                docs[index] = doc
                self._docs[ids[index]] = doc
        return docs

    def _tokens_and_lemmas(self, text, doc):
        if doc is not None:
            tokens = lexical.parser_tokens(doc)
            lemmas = [token.lemma_.casefold() for token in doc if not token.is_space]
            return tokens, lemmas, len(list(doc.sents))
        tokens = lexical.regex_tokens(text)
        from qdp.preprocessing.tokenizer import split_sentences
        sentences = split_sentences(text, self.cfg["preprocess"].get("sentence_abbreviations", []))
        return tokens, [token.casefold() for token in tokens], max(1, len(sentences))

    def fit_artifacts(self, train_records):
        views = [self._make_view(record) for record in train_records]
        texts = [view.nl_view for view in views]
        ids = [record.question_id for record in train_records]
        docs = self._parse_many(texts, ids)
        token_lists = [self._tokens_and_lemmas(view.nl_view, doc)[0] for view, doc in zip(views, docs)]
        self.rare_df = lexical.fit_rare_word_df(token_lists)
        if self.cfg["features"].get("use_tfidf", True):
            self.tfidf, self.tfidf_threshold = tfidf.fit(texts, float(self.cfg["features"].get("tfidf_high_percentile", 90)))

    def semantic_units(self, records):
        units = []
        for record in records:
            view = self._make_view(record)
            if self.embedder is None:
                continue
            units.append((record.question_id, "question", view.nl_view))
            if view.option_block:
                units.append((record.question_id, "stem", view.nl_stem))
                for index, option in enumerate(view.option_block):
                    units.append((record.question_id, f"option:{index}", option))
            for index, sentence in enumerate(view.sentences[: int(self.cfg["features"].get("max_sentences_for_f29", 40))]):
                units.append((record.question_id, f"sentence:{index}", sentence))
        return units

    def _semantic_map(self, records):
        if self.embedder is None:
            return {}
        units = self.semantic_units(records)
        if not units:
            return {}
        embeddings = self.embedder.encode_units(units)
        mapped = {}
        for unit, vector in zip(units, embeddings):
            qid, role, _ = unit
            mapped[(qid, role)] = vector
        return mapped

    def extract_many(self, records):
        records = list(records)
        if any(not hasattr(record, "question") for record in records):
            raise TypeError("FeatureExtractor.extract_many accepts Record objects only")
        for record in records:
            self._make_view(record)
        texts = [self._views[r.question_id].nl_view for r in records]
        ids = [r.question_id for r in records]
        docs = self._parse_many(texts, ids) if self.nlp is not None else [None] * len(records)
        semantic_map = self._semantic_map(records)
        tfidf_rows = tfidf.extract(texts, self.tfidf, self.tfidf_threshold) if self.tfidf is not None else [None] * len(records)
        output = []
        for index, (record, doc, tfidf_row) in enumerate(zip(records, docs, tfidf_rows)):
            view = self._views[record.question_id]
            tokens, lemmas, n_sentences = self._tokens_and_lemmas(view.nl_view, doc)
            row = {}
            enabled = {
                "lexical": self.cfg["features"].get("use_lexical", True),
                "readability": self.cfg["features"].get("use_readability", True),
                "syntax": self.cfg["features"].get("use_syntax", True),
                "task": self.cfg["features"].get("use_task", True),
                "mcq": self.cfg["features"].get("use_mcq", True),
                "tfidf": self.cfg["features"].get("use_tfidf", True),
                "semantic_scalar": self.cfg["features"].get("use_semantic_scalar", True),
                "programming": self.cfg["features"].get("use_programming", True),
            }
            if enabled["lexical"]:
                row.update(lexical.extract(tokens, self.rare_df, int(self.cfg["features"]["rare_word_df_threshold"]), n_sentences))
            else:
                row.update({fid: math.nan for fid, _, fam in FEATURES if fam == "lexical"})
            row.update(readability.extract(view.nl_view, row.get("F01", len(tokens)), n_sentences) if enabled["readability"] else {fid: math.nan for fid, _, fam in FEATURES if fam == "readability"})
            if enabled["syntax"]:
                row.update(syntax.extract(doc, {x.casefold() for x in self.lex.get("negation", [])}, self.cfg["features"]["clause_heads"]))
            else:
                row.update({fid: math.nan for fid, _, fam in FEATURES if fam == "syntax"})
            if enabled["task"]:
                option_lines = set()
                mcq_info = self._mcq(record.question)
                if mcq_info.get("detected"):
                    option_lines = set(mcq_info.get("option_lines", []))
                row["F14"] = task.question_type(view.nl_view)
                row["F15"] = task.instruction_type(lemmas, self.lex.get("task", {}))
                row["F16"] = task.cognitive_demand(lemmas, self.lex.get("bloom", {}))
                row["F17"] = task.subquestion_count(view.nl_view, option_lines)
            else:
                row.update({"F14": "Other", "F15": "Other", "F16": 1, "F17": 0})
            mcq_info = self._mcq(record.question)
            option_vectors = [semantic_map.get((record.question_id, f"option:{i}")) for i in range(len(view.option_block))]
            stem_vector = semantic_map.get((record.question_id, "stem"))
            if enabled["mcq"]:
                mcq_row = mcq.extract(view.option_block, view.nl_stem, stem_vector, option_vectors)
                row.update(mcq_row)
            else:
                row.update({"F18": 0, "F19": math.nan, "F20": math.nan, "F21": math.nan, "F22": math.nan})
            if enabled["tfidf"] and tfidf_row is not None:
                row.update(tfidf_row)
            else:
                row.update({"F23": math.nan, "F24": math.nan, "F25": 0})
            if enabled["semantic_scalar"]:
                max_sentences = min(len(view.sentences), int(self.cfg["features"].get("max_sentences_for_f29", 40)))
                sentence_vectors = [
                    semantic_map[(record.question_id, f"sentence:{i}")]
                    for i in range(max_sentences)
                    if (record.question_id, f"sentence:{i}") in semantic_map
                ]
                row.update(semantic_scalar.extract(view.nl_view, row.get("F01", len(tokens)), self.lex.get("concepts", []), self.lex.get("technical_terms", []), sentence_vectors or None))
            else:
                row.update({fid: math.nan for fid, _, fam in FEATURES if fam == "semantic_scalar"})
            if enabled["programming"]:
                row.update(programming.extract(
                    view.nl_view,
                    view.code_spans,
                    self.lex.get("algorithm", {}),
                    self.lex.get("constraint_patterns", []),
                    recursion_weight=int(self.cfg["features"]["recursion_weight"]),
                    indent_width=int(self.cfg["features"]["indent_width"]),
                    branch_keywords=self.lex.get("branch_keywords", []),
                    loop_keywords=self.lex.get("loop_keywords", []),
                ))
            else:
                row.update({fid: math.nan for fid, _, fam in FEATURES if fam == "programming"})
            row["mcq_detected"] = int(bool(view.option_block))
            row["code_detected"] = int(bool(view.code_spans))
            row["input_size_detected"] = int(not math.isnan(row.get("F31", math.nan)))
            row["parse_ok"] = int(doc is not None)
            row["multi_segment"] = int(n_sentences >= 2)
            row["semantic_truncated"] = bool(view.semantic_truncated)
            row["code_parse_exact"] = row.get("code_parse_exact", 0)
            row["code_detection_tiers"] = [tier for _, _, tier in getattr(view, "code_detection_tiers", [])]
            row["mcq_detection_tier"] = getattr(view, "mcq_detection_tier", None)
            output.append(row)
        return output

from __future__ import annotations
from qdp.evaluation.metrics import evaluate
FAMILY_FLAGS=['lexical','readability','syntax','task','mcq','tfidf','semantic_scalar','programming','semantic_embedding']

def ablation_plan():
    return [('A0',[]),('A1',['lexical','readability']),('A2',['lexical','readability','syntax']),('A3',['lexical','readability','syntax','task']),('A4',['lexical','readability','syntax','task','mcq']),('A5',['lexical','readability','syntax','task','mcq','tfidf']),('A6',['lexical','readability','syntax','task','mcq','tfidf','semantic_scalar']),('A7',['lexical','readability','syntax','task','mcq','tfidf','semantic_scalar','programming']),('A8',FAMILY_FLAGS),('A8b',[32,48,64])]

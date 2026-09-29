from qdp.preprocessing.mcq_parser import detect_mcq
from qdp.preprocessing.code_detector import detect_code_spans

def test_mcq_options():
    r=detect_mcq('(A) one\n(B) two\n(C) three\n(D) four',[r'^\s*\([A-Da-d]\)\s+']); assert r['detected'] and len(r['options'])==4 and r['options'][0]=='one'

def test_code_fence():
    c,_=detect_code_spans('```python\nfor i in range(3):\n print(i)\n```',[r'^\s*def\s+',r'\bfor\s*\(']); assert c


def test_mcq_rejects_nonsequential_markers():
    r=detect_mcq('(A) one\n(C) three\n(D) four',[r'^\s*\([A-Da-d]\)\s+'])
    assert not r['detected']

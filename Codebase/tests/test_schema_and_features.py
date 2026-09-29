import math, json
from pathlib import Path
from qdp.data.schema import make_record
from qdp.data.loader import CorpusLoader
from qdp.features.registry import FEATURES
from qdp.features.lexical import extract,fit_rare_word_df
from qdp.features.task import question_type,cognitive_demand

def test_schema_01_jsonl_load():
    lm={'easy':'Easy','moderate':'Moderate','hard':'Hard'}; loader=CorpusLoader(lm,1.0); p=Path('tests/_tmp.jsonl'); p.write_text('{"question":"What?","label":"easy"}\n'); rs=loader.load(p); assert len(rs)==1; p.unlink()
def test_schema_04_content_id():
    r=make_record('hello','Easy'); assert r.question_id==r.raw_hash[:16]
def test_feat_01_05():
    df=fit_rare_word_df(['one two','one three']); x=extract('one four',df,2,1); assert x['F01']==2 and x['F03']==3.5 and x['F04']==1

def test_closed_feature_registry(): assert len(FEATURES)==35 and FEATURES[0][0]=='F01' and FEATURES[-1][0]=='F35'
def test_qtype_and_bloom(): assert question_type('Why is this hard?')=='Why'; assert cognitive_demand(['define','evaluate'],{'Remember':['define'],'Evaluate':['evaluate']})==5

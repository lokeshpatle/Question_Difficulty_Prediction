import numpy as np
from qdp.transform.correlation_filter import CorrelationFilter
from qdp.models.baselines import make_logistic

def test_corr_filter():
    X=np.array([[1,2,3],[2,4,6],[3,6,9]],dtype=float); f=CorrelationFilter(.9).fit(X,['a','b','c']); assert len(f.columns())<3

def test_logistic_trains():
    X=np.array([[0],[1],[2],[3],[4],[5]],dtype=float); y=np.array(['Easy','Easy','Moderate','Moderate','Hard','Hard']); m=make_logistic(); m.fit(X,y); assert m.predict(X).shape==(6,)


def test_categorical_encoder_stable_names():
    import pandas as pd
    from qdp.transform.encoder import CategoricalEncoder
    enc = CategoricalEncoder().fit(pd.DataFrame({"F14": ["What", "Other"], "F15": ["Define", "Other"]}))
    names = enc.names()
    assert names[0] == "question_type=What"
    assert "instruction_type=Other" in names
    assert enc.transform(pd.DataFrame({"F14": ["What"], "F15": ["Define"]})).shape[1] == len(names)


def test_feature_selector_drops_variance_and_correlation():
    import numpy as np
    from qdp.transform.correlation_filter import FeatureSelector
    X = np.array([[1, 10, 4], [2, 20, 5], [3, 30, 6], [4, 40, 7]], dtype=float)
    selector = FeatureSelector(0.9).fit(X, ["a", "b", "c"])
    assert selector.columns() == ["a"]

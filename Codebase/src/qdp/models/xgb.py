from __future__ import annotations

from xgboost import XGBClassifier


def make_xgb(max_depth=6, learning_rate=0.05, subsample=0.8, seed=42, n_estimators=2000, early_stopping_rounds=50):
    kwargs = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_weight=1,
        subsample=float(subsample),
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=1,
    )
    # xgboost 3.x accepts the parameter on the estimator, older versions accept it too.
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBClassifier(**kwargs)


def grid():
    return [(depth, lr, subsample) for depth in (4, 6, 8) for lr in (0.03, 0.05, 0.1) for subsample in (0.8, 1.0)]

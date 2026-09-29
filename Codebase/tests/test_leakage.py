import numpy as np
from qdp.data.schema import make_record
from qdp.data.deduplicator import exact_deduplicate
from qdp.data.splitter import group_split

def test_no_duplicate_ids_after_exact():
    rs=[make_record('same','Easy','a'),make_record('same','Easy','b')]; kept,_,_=exact_deduplicate(rs); assert len(kept)==1

def test_group_split_assigns_all_once():
    rs=[make_record(f'q{i}','Easy' if i%3==0 else 'Moderate' if i%3==1 else 'Hard',str(i)) for i in range(30)]
    out,info=group_split(rs, small_dataset_threshold=5000); assert all(r.split in {'train','validation','test'} for r in out); assert len({r.question_id for r in out})==30; assert {r.label for r in out if r.split=='test'}=={'Easy','Moderate','Hard'}


def test_group_cv_validation_folds_cover_all_classes():
    from qdp.data.splitter import repeated_group_cv
    rs=[make_record(f'q{i}', 'Easy' if i%3==0 else 'Moderate' if i%3==1 else 'Hard', str(i)) for i in range(27)]
    folds = repeated_group_cv(rs, n_splits=3, n_repeats=1, seed=42)
    assert len(folds) == 3
    by_id = {r.question_id: r.label for r in rs}
    for fold in folds:
        assert {by_id[qid] for qid in fold['validation_ids']} == {'Easy','Moderate','Hard'}
        assert {by_id[qid] for qid in fold['train_ids']} == {'Easy','Moderate','Hard'}


def test_standard_group_split_targets_70_15_15():
    rs=[make_record(f'Question {i} with unique token {i}', 'Easy' if i%3==0 else 'Moderate' if i%3==1 else 'Hard', str(i)) for i in range(60)]
    out, info = group_split(rs, small_dataset_threshold=0, seed=42, max_ratio_deviation=0.02)
    assert info['sizes'] == [42, 9, 9]
    assert info['ratios'] == [0.7, 0.15, 0.15]

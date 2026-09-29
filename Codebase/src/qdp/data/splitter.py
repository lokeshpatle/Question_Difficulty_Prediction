from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

from qdp.data.schema import with_split
from qdp.data.deduplicator import norm


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def shingles(text: str, k: int = 5):
    if not text:
        return {""}
    if len(text) < k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _stable_u64(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def minhash_signature(shingle_set, H=128, seed=42):
    ordered = sorted(shingle_set)
    signature = []
    for i in range(H):
        minimum = 2**64 - 1
        prefix = f"{seed}|{i}|".encode()
        for shingle in ordered:
            minimum = min(minimum, _stable_u64(prefix + shingle.encode("utf-8")))
        signature.append(minimum)
    return tuple(signature)


def lsh_families(records, H=128, bands=32, threshold=0.85, seed=42):
    if H % bands != 0:
        raise ValueError("minhash H must be divisible by LSH band count")
    n = len(records)
    if not n:
        return {}
    rows = H // bands
    signatures = []
    buckets = defaultdict(list)
    for idx, record in enumerate(records):
        sig = minhash_signature(shingles(norm(record.question)), H, seed)
        signatures.append(sig)
        for band in range(bands):
            start = band * rows
            chunk = sig[start:start + rows]
            key = hashlib.sha256(f"{band}|{chunk}".encode()).hexdigest()
            buckets[key].append(idx)
    uf = UnionFind(n)
    candidates = set()
    for values in buckets.values():
        values = sorted(set(values))
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                candidates.add((a, b))
    for i, j in sorted(candidates):
        sim = sum(a == b for a, b in zip(signatures[i], signatures[j])) / H
        if sim >= threshold:
            uf.union(i, j)
    families = defaultdict(list)
    for idx in range(n):
        families[uf.find(idx)].append(idx)
    normalized = {}
    for members in families.values():
        normalized[min(members)] = sorted(members)
    return normalized


def _greedy_group_assignment(records, ratios, seed, max_ratio_deviation, H=128, bands=32, threshold=0.85, minhash_seed=None):
    families = lsh_families(records, H=H, bands=bands, threshold=threshold, seed=seed if minhash_seed is None else minhash_seed)
    groups = []
    for gid, idxs in families.items():
        counts = Counter(records[i].label for i in idxs)
        majority = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        groups.append((gid, idxs, majority))
    groups.sort(key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    # Stable seeded tie-breaking only; group order remains deterministic.
    tie_values = {gid: float(rng.random()) for gid, _, _ in groups}
    groups.sort(key=lambda item: (item[0], tie_values[item[0]]))
    total = len(records)
    targets = [ratio * total for ratio in ratios]
    label_totals = Counter(r.label for r in records)
    label_targets = {label: [ratio * label_totals[label] for ratio in ratios] for label in label_totals}
    counts = [0, 0, 0]
    assigned = Counter()
    split_items = [[], [], []]
    for gid, idxs, majority in groups:
        size = len(idxs)
        choices = []
        for split_idx in range(3):
            label_target = label_targets[majority][split_idx]
            label_deficit = label_target - assigned[(split_idx, majority)]
            label_fraction = (label_deficit / label_target) if label_target > 0 else 0.0
            total_deficit = targets[split_idx] - counts[split_idx]
            total_fraction = total_deficit / targets[split_idx] if targets[split_idx] > 0 else 0.0
            tie = tie_values[gid] if split_idx == 0 else tie_values[gid] + split_idx * 1e-12
            # Primary rule: send the group to the split furthest below its
            # target quota for that group's majority label. Total split deficit
            # is the secondary criterion; final tie is deterministic.
            choices.append((-label_fraction, -total_fraction, tie, split_idx))
        chosen = min(choices)[3]
        split_items[chosen].extend(idxs)
        counts[chosen] += size
        assigned[(chosen, majority)] += size
    actual = [count / total for count in counts]
    deviation = max(abs(actual[i] - ratios[i]) for i in range(3))
    if deviation > max_ratio_deviation:
        raise ValueError(f"split_ratio_deviation:{actual} target={ratios} max={max_ratio_deviation}")
    family_map = {}
    split_names = ("train", "validation", "test")
    for split_idx, idxs in enumerate(split_items):
        for idx in idxs:
            family_id = None
            # Find family key without relying on source metadata. This is only for audit/group tests.
            for key, members in families.items():
                if idx in members:
                    family_id = f"family_{key:08d}"
                    break
            records[idx] = with_split(records[idx], split_names[split_idx], family_id)
            family_map[records[idx].question_id] = family_id
    return records, {"sizes": counts, "ratios": actual, "families": len(groups), "deviation": deviation, "family_map": family_map}


def repeated_group_cv(records, n_splits=5, n_repeats=3, seed=42):
    pool = list(records)
    if len(pool) < n_splits * 3:
        raise ValueError("insufficient_records_for_group_cv")
    families = lsh_families(pool, H=128, bands=32, threshold=0.85, seed=seed)
    groups = []
    for gid, idxs in families.items():
        counts = Counter(pool[i].label for i in idxs)
        majority = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        groups.append((gid, sorted(idxs), majority, counts))
    if len(groups) < n_splits:
        raise ValueError(f"insufficient_groups_for_group_cv:{len(groups)}<{n_splits}")
    group_class_counts = Counter()
    for _, idxs, _, counts in groups:
        for label in counts:
            group_class_counts[label] += 1
    required = {"Easy", "Moderate", "Hard"}
    insufficient = [label for label in required if group_class_counts[label] < n_splits]
    if insufficient:
        raise ValueError(f"insufficient_groups_per_class_for_cv:{insufficient}")
    groups.sort(key=lambda x: x[0])
    folds = []
    for repeat in range(n_repeats):
        rng = np.random.RandomState(seed + repeat)
        order = list(range(len(groups)))
        rng.shuffle(order)
        buckets = [[] for _ in range(n_splits)]
        label_counts = [Counter() for _ in range(n_splits)]
        assigned_groups = set()

        # Seed each validation fold with real groups covering all three labels.
        # This guarantees class coverage whenever the group-capacity precheck passed.
        for fold_idx in range(n_splits):
            for label in ("Easy", "Moderate", "Hard"):
                if label in label_counts[fold_idx]:
                    continue
                candidates = [
                    group_idx for group_idx in order
                    if group_idx not in assigned_groups and label in groups[group_idx][3]
                ]
                if not candidates:
                    raise ValueError(
                        f"unable_to_seed_group_cv_class:{repeat}:{fold_idx}:{label}"
                    )
                chosen = min(
                    candidates,
                    key=lambda group_idx: (
                        len(groups[group_idx][1]),
                        groups[group_idx][0],
                    ),
                )
                _, idxs, _, counts = groups[chosen]
                assigned_groups.add(chosen)
                buckets[fold_idx].extend(idxs)
                label_counts[fold_idx].update(counts)

        total_label_counts = Counter(record.label for record in pool)
        target_per_fold = {
            label: total_label_counts[label] / n_splits
            for label in total_label_counts
        }
        for group_idx in order:
            if group_idx in assigned_groups:
                continue
            _, idxs, _, counts = groups[group_idx]
            size = len(idxs)

            def score(fold_idx):
                projected = label_counts[fold_idx].copy()
                projected.update(counts)
                class_error = sum(
                    abs(projected[label] - target_per_fold[label])
                    for label in target_per_fold
                )
                size_error = len(buckets[fold_idx])
                return (class_error, size_error, fold_idx)

            chosen = min(range(n_splits), key=score)
            buckets[chosen].extend(idxs)
            label_counts[chosen].update(counts)

        all_ids = {record.question_id for record in pool}
        for fold_idx in range(n_splits):
            validation_ids = [pool[i].question_id for i in buckets[fold_idx]]
            validation_set = set(validation_ids)
            train_ids = [
                record.question_id
                for record in pool
                if record.question_id not in validation_set
            ]
            if not validation_ids:
                raise ValueError(f"empty_group_cv_validation_fold:{repeat}:{fold_idx}")
            validation_labels = {pool[i].label for i in buckets[fold_idx]}
            train_labels = {
                record.label
                for record in pool
                if record.question_id not in validation_set
            }
            if validation_labels != required:
                raise ValueError(
                    f"group_cv_validation_class_coverage_failure:{repeat}:{fold_idx}:"
                    f"validation={sorted(validation_labels)}"
                )
            if train_labels != required:
                raise ValueError(
                    f"group_cv_train_class_coverage_failure:{repeat}:{fold_idx}:"
                    f"train={sorted(train_labels)}"
                )
            if set(train_ids) | validation_set != all_ids or set(train_ids) & validation_set:
                raise ValueError(f"group_cv_partition_failure:{repeat}:{fold_idx}")
            folds.append({
                "repeat": repeat,
                "fold": fold_idx,
                "train_ids": train_ids,
                "validation_ids": validation_ids,
            })
    return folds

def group_split(
    records,
    ratios=(0.70, 0.15, 0.15),
    seed=42,
    minhash_seed=42,
    max_ratio_deviation=0.02,
    H=128,
    bands=32,
    threshold=0.85,
    small_dataset_threshold=5000,
    cv_seed=42,
):
    if not records:
        raise ValueError("empty_dataset")
    # Small corpora reserve a deterministic ~15% held-out test set and use
    # repeated stratified group CV over the remaining pool for model selection.
    # There is intentionally no external validation split in this mode.
    if len(records) < small_dataset_threshold:
        families = lsh_families(records, H=H, bands=bands, threshold=threshold, seed=minhash_seed)
        groups = []
        for gid, idxs in families.items():
            counts = Counter(records[i].label for i in idxs)
            majority = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            groups.append((gid, sorted(idxs), majority))
        groups.sort(key=lambda item: item[0])
        rng = np.random.RandomState(seed)
        tie = {gid: float(rng.random()) for gid, _, _ in groups}

        target_test = max(1, int(round(len(records) * ratios[2])))
        max_test_groups = len(groups) - 5
        if max_test_groups < 3:
            raise ValueError(f"insufficient_groups_for_small_dataset_test_and_cv:{len(groups)}<8")
        # Start by reserving one group from each class when feasible. This guards
        # against the TDD's requirement that the held-out test set cover all classes.
        selected = set()
        used = 0
        class_first = {}
        for gid, idxs, _ in sorted(groups, key=lambda g: (tie[g[0]], g[0])):
            labels_in_group = {records[idx].label for idx in idxs}
            for label in ("Easy", "Moderate", "Hard"):
                if label in labels_in_group and label not in class_first:
                    class_first[label] = (gid, idxs)
        for label in ("Easy", "Moderate", "Hard"):
            if label in class_first and len(selected) < max_test_groups:
                gid, idxs = class_first[label]
                if gid not in selected:
                    selected.add(gid); used += len(idxs)

        # Fill toward the target using a deterministic closest-size rule.
        remaining = [g for g in groups if g[0] not in selected]
        remaining.sort(key=lambda g: (abs((used + len(g[1])) - target_test), tie[g[0]], g[0]))
        for gid, idxs, _ in remaining:
            if used >= target_test or len(selected) >= max_test_groups:
                break
            selected.add(gid); used += len(idxs)

        split_counts = [0, 0, 0]
        family_map = {}
        for gid, idxs, _ in groups:
            split_name = 'test' if gid in selected else 'train'
            for idx in idxs:
                family_id = f'family_{gid:08d}'
                records[idx] = with_split(records[idx], split_name, family_id)
                family_map[records[idx].question_id] = family_id
                split_counts[2 if split_name == 'test' else 0] += 1

        pool = [r for r in records if r.split == 'train']
        folds = repeated_group_cv(pool, n_splits=5, n_repeats=3, seed=cv_seed) if len(pool) >= 15 else []
        actual = [split_counts[0]/len(records), 0.0, split_counts[2]/len(records)]
        split_info = {
            'sizes': split_counts,
            'ratios': actual,
            'families': len(groups),
            'deviation': abs(actual[2] - ratios[2]),
            'family_map': family_map,
            'cv_folds': folds,
            'selection_pool': 'train',
            'small_dataset_mode': True,
        }
        return records, split_info
    records, split_info = _greedy_group_assignment(records, ratios, seed, max_ratio_deviation, H=H, bands=bands, threshold=threshold, minhash_seed=minhash_seed)
    # Only large-corpus path uses the explicit validation split.
    split_info["cv_folds"] = []
    split_info["selection_pool"] = "train_validation"
    return records, split_info

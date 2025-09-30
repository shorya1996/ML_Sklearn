# util/rulefit_model.py
from typing import List, Optional, Tuple, Dict, Any
import math
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

# --- helpers ---
class UniqueList:
    def __init__(self, iterable=None):
        self.items = []
        self.item_set = set()
        if iterable:
            for it in iterable:
                self.add(it)
    def add(self, it):
        if it not in self.item_set:
            self.items.append(it)
            self.item_set.add(it)
    def __iter__(self): return iter(self.items)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class RuleCondition:
    def __init__(self, feature_index: int, threshold: float, operator: str, support: float, feature_name: Optional[str] = None, is_binary: bool = False):
        self.feature_index = int(feature_index)
        self.threshold = float(threshold) if threshold is not None else None
        self.operator = operator  # "<=", ">", "==" etc.
        self.support = float(support)
        self.feature_name = feature_name
        self.is_binary = is_binary
    def __str__(self):
        name = self.feature_name or f"f{self.feature_index}"
        return f"{name} {self.operator} {self.threshold}"

    def transform(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_index]
        if self.is_binary:
            return (col == self.threshold).astype(int)
        if self.operator == "<=":
            return (col <= self.threshold).astype(int)
        else:
            return (col > self.threshold).astype(int)

class Rule:
    def __init__(self, rule_conditions: List[RuleCondition], prediction_value: Optional[float] = None):
        self.conditions = UniqueList(rule_conditions)
        self.support = min([c.support for c in rule_conditions]) if rule_conditions else 0.0
        self.prediction_value = prediction_value
    def __str__(self):
        return " & ".join([str(c) for c in self.conditions])
    def transform(self, X: np.ndarray) -> np.ndarray:
        if len(self.conditions) == 0:
            return np.ones(X.shape[0], dtype=int)
        masks = [c.transform(X) for c in self.conditions]
        out = masks[0].copy()
        for m in masks[1:]:
            out = out * m
        return out

# --- RuleEnsemble ---
class RuleEnsemble:
    def __init__(self, tree_list, feature_names: List[str], categorical_features: List[str], max_rules: int = 200, min_support: float = 0.005):
        self.tree_list = tree_list
        self.feature_names = feature_names or []
        self.categorical_features = categorical_features or []  # names of binary features (like "col=val")
        self.max_rules = int(max_rules)
        self.min_support = float(min_support)
        self.rules: List[Rule] = []
        self._extract_rules()

    def _extract_rules(self):
        candidates: List[Rule] = []
        for tw in self.tree_list:
            tree = tw[0].tree_
            total = float(tree.n_node_samples[0])
            def traverse(node=0, conditions=None):
                if conditions is None:
                    conditions = []
                if tree.children_left[node] != tree.children_right[node]:
                    feat = int(tree.feature[node])
                    thr = float(tree.threshold[node])
                    left = tree.children_left[node]
                    right = tree.children_right[node]
                    left_support = tree.n_node_samples[left] / total
                    right_support = tree.n_node_samples[right] / total
                    fname = self.feature_names[feat] if feat < len(self.feature_names) else f"f{feat}"
                    # For binary features (one-hot), prefer equality conditions:
                    if fname in self.categorical_features:
                        cond_left = conditions + [RuleCondition(feat, 0.0, "==", left_support, feature_name=fname, is_binary=True)]
                        traverse(left, cond_left)
                        cond_right = conditions + [RuleCondition(feat, 1.0, "==", right_support, feature_name=fname, is_binary=True)]
                        traverse(right, cond_right)
                    else:
                        cond_left = conditions + [RuleCondition(feat, thr, "<=", left_support, feature_name=fname, is_binary=False)]
                        traverse(left, cond_left)
                        cond_right = conditions + [RuleCondition(feat, thr, ">", right_support, feature_name=fname, is_binary=False)]
                        traverse(right, cond_right)
                else:
                    if conditions:
                        est_support = min([c.support for c in conditions])
                        if est_support >= self.min_support:
                            candidates.append(Rule(conditions))
                if len(candidates) >= self.max_rules:
                    return
            traverse()
            if len(candidates) >= self.max_rules:
                break
        # dedupe
        uniq = {}
        for r in candidates:
            k = str(r)
            if k not in uniq:
                uniq[k] = r
            if len(uniq) >= self.max_rules:
                break
        self.rules = list(uniq.values())

    def transform_sparse(self, X: np.ndarray) -> csr_matrix:
        n_rows = X.shape[0]
        n_rules = len(self.rules)
        if n_rules == 0:
            return csr_matrix((n_rows, 0), dtype=int)
        rows, cols, data = [], [], []
        for j, rule in enumerate(self.rules):
            mask = None
            for cond in rule.conditions:
                col_vals = X[:, cond.feature_index]
                if cond.is_binary:
                    cond_mask = (col_vals == cond.threshold)
                else:
                    cond_mask = (col_vals <= cond.threshold) if cond.operator == "<=" else (col_vals > cond.threshold)
                cond_mask = np.asarray(cond_mask, dtype=bool)
                if mask is None:
                    mask = cond_mask
                else:
                    mask &= cond_mask
                if not mask.any():
                    break
            if mask is None:
                continue
            idxs = mask.nonzero()[0]
            if idxs.size > 0:
                rows.extend(idxs.tolist())
                cols.extend([j] * idxs.size)
                data.extend([1] * idxs.size)
        if len(data) == 0:
            return csr_matrix((n_rows, n_rules), dtype=int)
        return csr_matrix((data, (rows, cols)), shape=(n_rows, n_rules), dtype=int)

# --- RuleFit main ---
class RuleFit:
    def __init__(self,
                 tree_size: int = 4,
                 max_rules: int = 200,
                 min_support: float = 0.01,
                 sample_size_cap: int = 50000,
                 top_n_for_cats: int = 30,
                 n_estimators_cap: int = 100,
                 random_state: Optional[int] = None):
        self.tree_size = int(tree_size)
        self.max_rules = int(max_rules)
        self.min_support = float(min_support)
        self.sample_size_cap = int(sample_size_cap)
        self.top_n_for_cats = int(top_n_for_cats)
        self.n_estimators_cap = int(n_estimators_cap)
        self.random_state = random_state

        self.feature_names: List[str] = []
        self.binary_feature_names: List[str] = []
        self.freq_maps: Dict[str, Dict[Any, float]] = {}
        self.rule_ensemble: Optional[RuleEnsemble] = None
        self.selector_clf = None
        self.X = None
        self.y = None
        self.fit_time_info = {}

    def _prepare_inputs(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Dict[Any, float]]]:
        X = X_df.copy()
        feature_names: List[str] = []
        binary_feature_names: List[str] = []
        freq_maps: Dict[str, Dict[Any, float]] = {}
        mats: List[np.ndarray] = []
        n_rows = X.shape[0]

        # numeric
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            col = pd.to_numeric(X[c], errors='coerce').fillna(0.0).astype('float32')
            mats.append(col.values.reshape(-1, 1))
            feature_names.append(c)

        # categorical
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for c in cat_cols:
            nunique = int(X[c].nunique(dropna=False))
            if nunique == 0:
                mats.append(np.zeros((n_rows, 1), dtype='float32'))
                feature_names.append(f"{c}_empty")
                continue
            if nunique <= self.top_n_for_cats:
                s = X[c].astype(object).fillna("__MISSING__")
                cats = pd.Index(s.unique())
                one_hot = np.zeros((n_rows, len(cats)), dtype='float32')
                for j, cat_val in enumerate(cats):
                    mask = (s == cat_val)
                    one_hot[:, j] = mask.astype('float32')
                    fname = f"{c}={cat_val}"
                    feature_names.append(fname)
                    binary_feature_names.append(fname)
                mats.append(one_hot)
            else:
                s = X[c].astype(object).fillna("__MISSING__")
                freq = s.value_counts(normalize=True)
                mapped = s.map(freq).astype('float32').values.reshape(-1, 1)
                mats.append(mapped)
                fname = f"{c}_freq"
                feature_names.append(fname)
                # register freq map
                freq_maps[fname] = freq.to_dict()

        if len(mats) == 0:
            mats.append(np.zeros((n_rows, 1), dtype='float32'))
            feature_names.append('const')

        X_np = np.hstack(mats).astype('float32')
        return X_np, feature_names, binary_feature_names, freq_maps

    def _get_sample(self, X_np: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_rows = X_np.shape[0]
        if n_rows <= self.sample_size_cap:
            return X_np, y
        rng = np.random.RandomState(self.random_state)
        unique_y = np.unique(y)
        if len(unique_y) == 2:
            idxs = []
            frac = float(self.sample_size_cap) / float(n_rows)
            for val in unique_y:
                pos = np.where(y == val)[0]
                k = max(1, int(math.floor(len(pos) * frac)))
                if len(pos) <= k:
                    idxs.extend(pos.tolist())
                else:
                    idxs.extend(rng.choice(pos, size=k, replace=False).tolist())
            idxs = np.array(sorted(set(idxs)), dtype=int)
            return X_np[idxs], y[idxs]
        else:
            idx = rng.choice(n_rows, size=self.sample_size_cap, replace=False)
            return X_np[idx], y[idx]

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        t_start = time.time()
        if isinstance(X, pd.DataFrame):
            X_np, feature_names_list, binary_names, freq_maps = self._prepare_inputs(X)
            self.feature_names = feature_names_list
            self.binary_feature_names = binary_names
            self.freq_maps = freq_maps
        else:
            X_np = np.asarray(X).astype('float32')
            self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(X_np.shape[1])]
            self.binary_feature_names = []
            self.freq_maps = {}

        self.X = X_np
        self.y = np.asarray(y)

        X_sample, y_sample = self._get_sample(self.X, self.y)

        n_estimators = max(10, int(math.ceil(self.max_rules / max(1, self.tree_size))))
        n_estimators = min(self.n_estimators_cap, n_estimators)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_leaf_nodes=self.tree_size,
            max_features='sqrt',
            n_jobs=-1,
            random_state=self.random_state,
            class_weight='balanced'
        )

        t0 = time.time()
        rf.fit(X_sample, y_sample)
        t1 = time.time()

        tree_list = [[est] for est in rf.estimators_]
        self.rule_ensemble = RuleEnsemble(
            tree_list=tree_list,
            feature_names=self.feature_names,
            categorical_features=self.binary_feature_names,
            max_rules=self.max_rules,
            min_support=self.min_support
        )

        t2 = time.time()
        transformed_sparse = self.rule_ensemble.transform_sparse(X_sample)
        t3 = time.time()

        if transformed_sparse.shape[1] == 0:
            self.selector_clf = None
            self.fit_time_info = {'rf_fit_s': t1 - t0, 'rule_extract_s': t2 - t1, 'sparse_build_s': t3 - t2, 'sgd_fit_s': 0.0, 'total_s': time.time() - t_start}
            return self

        # robust SGD loss selection across sklearn versions
        loss_candidates = ['log_loss', 'log']
        clf = None
        last_err = None
        for loss_name in loss_candidates:
            try:
                clf = SGDClassifier(loss=loss_name, penalty='l1', max_iter=2000, tol=1e-4, random_state=self.random_state)
                break
            except (TypeError, ValueError) as e:
                last_err = e
        if clf is None:
            try:
                clf = SGDClassifier(loss='modified_huber', penalty='l1', max_iter=2000, tol=1e-4, random_state=self.random_state)
            except Exception as e:
                raise last_err or e

        t4 = time.time()
        clf.fit(transformed_sparse, y_sample)
        t5 = time.time()

        self.selector_clf = clf
        self.transformed_sparse_sample = transformed_sparse
        self.fit_time_info = {
            'rf_fit_s': t1 - t0,
            'rule_extract_s': t2 - t1,
            'sparse_build_s': t3 - t2,
            'sgd_fit_s': t5 - t4,
            'total_s': time.time() - t_start
        }
        return self

    # --- interpret freq conditions back to categories ---
    def _interpret_freq_condition(self, feature_name: str, operator: str, threshold: float, top_k: int = 10) -> Dict[str, Any]:
        """
        Map a frequency-rule like 'col_freq <= 0.02' to actual category values captured.
        Returns dict {covered: [...], note: str}
        """
        if not hasattr(self, "freq_maps") or feature_name not in getattr(self, "freq_maps", {}):
            return {"covered": [], "note": "no freq map"}
        freq_map = self.freq_maps[feature_name]
        if operator == "<=":
            cats = [cat for cat, f in freq_map.items() if f <= threshold]
        elif operator == ">":
            cats = [cat for cat, f in freq_map.items() if f > threshold]
        else:
            cats = [cat for cat, f in freq_map.items() if eval(f"{f} {operator} {threshold}")]
        total = len(cats)
        if total == 0:
            return {"covered": [], "note": "no categories match"}
        cats_sorted = sorted(cats, key=lambda c: freq_map[c], reverse=True)
        if total > top_k:
            return {"covered": cats_sorted[:top_k], "note": f"{total} categories matched (showing top {top_k})"}
        else:
            return {"covered": cats_sorted, "note": f"{total} categories matched"}

    def get_rules(self, min_support: float = 0.001) -> pd.DataFrame:
        import pandas as pd
        if self.rule_ensemble is None or len(self.rule_ensemble.rules) == 0:
            return pd.DataFrame(columns=['rule', 'coef', 'support', 'fraud_rate', 'captured_cases', 'length', 'human_readable'])
        full_sparse = self.rule_ensemble.transform_sparse(self.X)
        if full_sparse.shape[1] == 0:
            return pd.DataFrame(columns=['rule', 'coef', 'support', 'fraud_rate', 'captured_cases', 'length', 'human_readable'])
        coefs = self.selector_clf.coef_.ravel() if self.selector_clf is not None else np.zeros(full_sparse.shape[1])
        n_rows = self.X.shape[0]
        rows_sum = np.array(full_sparse.sum(axis=0)).ravel()
        data = []
        for i, rule in enumerate(self.rule_ensemble.rules):
            support = float(rows_sum[i]) / float(n_rows) if n_rows > 0 else 0.0
            if support < min_support:
                continue
            preds_idx = full_sparse[:, i].nonzero()[0]
            captured_cases = int(np.sum(self.y[preds_idx] == 1)) if preds_idx.size > 0 else 0
            fraud_rate = float(captured_cases / preds_idx.size) if preds_idx.size > 0 else 0.0
            # build human_readable interpretation for any freq conditions or binary features
            human_parts = []
            for cond in rule.conditions:
                fname = cond.feature_name or ""
                thr = cond.threshold
                # frequency-encoded interpretation
                if fname.endswith("_freq"):
                    interp = self._interpret_freq_condition(fname, cond.operator, thr, top_k=8)
                    if interp["covered"]:
                        human_parts.append(f"{fname} {cond.operator} {thr} -> {interp['covered']} ({interp['note']})")
                    else:
                        human_parts.append(f"{fname} {cond.operator} {thr} -> {interp['note']}")
                    continue

                # binary one-hot features like "phonetype=WORK"
                if cond.is_binary and "=" in fname:
                    # threshold should be 0.0 or 1.0 for binary features; use tolerance
                    try:
                        thr_f = float(thr)
                    except Exception:
                        thr_f = thr
                    if isinstance(thr_f, float) and abs(thr_f - 1.0) < 1e-6:
                        col_name, val = fname.split("=", 1)
                        human_parts.append(f"{col_name} == {val}")
                    elif isinstance(thr_f, float) and abs(thr_f - 0.0) < 1e-6:
                        col_name, val = fname.split("=", 1)
                        human_parts.append(f"{col_name} != {val}")
                    else:
                        human_parts.append(f"{fname} {cond.operator} {thr}")
                    continue

                # default numeric fallback
                human_parts.append(f"{fname} {cond.operator} {thr}")
            human = "; ".join(human_parts)
            data.append({
                'rule': str(rule),
                'coef': float(coefs[i]) if i < len(coefs) else 0.0,
                'support': support,
                'fraud_rate': fraud_rate,
                'captured_cases': captured_cases,
                'length': len(rule.conditions),
                'human_readable': human
            })
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df = df.sort_values(by=['coef', 'support'], ascending=False).reset_index(drop=True)
        return df

    def apply_rules_to_dataframe(self, df: pd.DataFrame, chunk_size: int = 100_000):
        if self.rule_ensemble is None:
            return []
        n = df.shape[0]
        counts = np.zeros(len(self.rule_ensemble.rules), dtype=int)
        for start in range(0, n, chunk_size):
            sub = df.iloc[start:start + chunk_size]
            X_sub_np, _, _, _ = self._prepare_inputs(sub)
            mat = self.rule_ensemble.transform_sparse(X_sub_np)
            counts += np.array(mat.sum(axis=0)).ravel()
        return [{'rule': str(self.rule_ensemble.rules[i]), 'count': int(counts[i])} for i in range(len(self.rule_ensemble.rules))]

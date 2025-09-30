# util/data_preprocessing.py
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    categorical_custom_columns: Optional[List[str]] = None,
    drop_columns_from_rules: Optional[List[str]] = None,
    dropna_rows: bool = True,
    missing_threshold: float = 0.30
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (DataFrame) and y (Series) for modeling.
    - Validates target_col exists.
    - Optionally marks given columns as categorical.
    - Drops user-specified columns and columns with > missing_threshold fraction missing.
    - Optionally drops rows with any remaining NaN (dropna_rows).
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe columns")

    data = df.copy()

    if categorical_custom_columns:
        for c in categorical_custom_columns:
            if c in data.columns:
                data[c] = data[c].astype("category")

    if drop_columns_from_rules:
        data = data.drop(columns=drop_columns_from_rules, errors="ignore")

    cols_to_drop = [col for col in data.columns if data[col].isnull().mean() > missing_threshold]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop, errors="ignore")

    if dropna_rows:
        before = len(data)
        data = data.dropna()
        after = len(data)
        if after == 0:
            raise ValueError("All rows dropped after dropna(); consider imputation or lowering threshold")

    if target_col not in data.columns:
        raise ValueError("After dropping, target_col no longer present. Check inputs.")

    y = data[target_col].copy()
    X = data.drop(columns=[target_col])
    return X, y


# --- Safe encoding helpers (coerce to object before adding sentinel values) ---
def safe_frequency_encode(series: pd.Series) -> np.ndarray:
    s = series.astype(object).fillna("__MISSING__")
    freq = s.value_counts(normalize=True)
    mapped = s.map(freq).astype("float32")
    return mapped.values

def safe_topn_collapse(series: pd.Series, n: int = 20) -> np.ndarray:
    s = series.astype(object).fillna("__MISSING__")
    top = pd.Series(s).value_counts().nlargest(n).index
    collapsed = s.where(s.isin(top), other="__OTHER__")
    cat = pd.Categorical(collapsed)
    return cat.codes.astype("int32")

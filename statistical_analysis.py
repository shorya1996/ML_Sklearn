# util/statistical_analysis.py
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

def _find_outliers_mask(s: pd.Series, iqr_factor: float = 1.5, z_thresh: float = 3.0) -> pd.Series:
    non_na = s.dropna()
    if non_na.empty:
        return pd.Series(False, index=s.index, dtype=bool)
    Q1 = non_na.quantile(0.25)
    Q3 = non_na.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_factor * IQR
    upper = Q3 + iqr_factor * IQR
    mean = non_na.mean()
    std = non_na.std(ddof=0)
    if std == 0 or np.isnan(std):
        z = pd.Series(0.0, index=s.index)
    else:
        z = (s - mean) / std
    mask = (s < lower) | (s > upper) | (z.abs() > z_thresh)
    return mask.fillna(False).astype(bool)

def perform_statistical_analysis(
    data: pd.DataFrame,
    drop_columns: Optional[list] = None,
    min_occurrence: int = 10,
    iqr_factor: float = 1.5,
    z_thresh: float = 3.0
) -> Dict[str, Any]:
    df = data.copy()
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    results = {
        "Numerical Summary": {},
        "Categorical Summary": {},
        "Outliers Dataset": pd.DataFrame(),
        "Non-Outliers Dataset": pd.DataFrame()
    }
    masks = []
    for col in numerical_columns:
        s = pd.to_numeric(df[col], errors='coerce')
        non_na = s.dropna()
        mean = non_na.mean() if not non_na.empty else np.nan
        median = non_na.median() if not non_na.empty else np.nan
        std = non_na.std() if not non_na.empty else np.nan
        minv = float(non_na.min()) if not non_na.empty else np.nan
        maxv = float(non_na.max()) if not non_na.empty else np.nan
        mask = _find_outliers_mask(s, iqr_factor=iqr_factor, z_thresh=z_thresh)
        masks.append(mask)
        results["Numerical Summary"][col] = {
            "Mean": float(mean) if not np.isnan(mean) else None,
            "Median": float(median) if not np.isnan(median) else None,
            "Standard Deviation": float(std) if not np.isnan(std) else None,
            "Min": minv,
            "Max": maxv,
            "Outliers Count": int(mask.sum())
        }
    for col in categorical_columns:
        value_counts = df[col].value_counts(dropna=False)
        filtered = value_counts[value_counts >= min_occurrence]
        results["Categorical Summary"][col] = {
            "Value Counts": filtered.to_dict(),
            "Unique Values": int(filtered.size)
        }
    if masks:
        combined_mask = pd.concat(masks, axis=1).any(axis=1)
        combined_mask = combined_mask.reindex(df.index).fillna(False).astype(bool)
        results["Outliers Dataset"] = df.loc[combined_mask].copy()
        results["Non-Outliers Dataset"] = df.loc[~combined_mask].copy()
    else:
        results["Outliers Dataset"] = pd.DataFrame(columns=df.columns)
        results["Non-Outliers Dataset"] = df.copy()
    return results

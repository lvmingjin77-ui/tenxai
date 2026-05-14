"""Feature quality metrics for cross-sectional alpha research."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def batch_cs_ic(df: pd.DataFrame, x_cols: list, target: str,
                date_col: str = "trade_date") -> dict:
    """
    Vectorized batch cross-sectional Spearman IC.
    Ranks all features in one pass per date → Pearson on ranks = Spearman.
    Returns {feature: np.array of per-date IC values}
    """
    ic_dict: dict = {f: [] for f in x_cols}
    cols = x_cols + [target]

    for _, g in df.groupby(date_col):
        sub = g[cols].dropna()
        if len(sub) < 5:
            continue
        ranked = sub.rank()
        normed = (ranked - ranked.mean()) / (ranked.std(ddof=1) + 1e-10)

        y = normed[target].values
        X = normed[x_cols].values
        corrs = (X * y[:, None]).mean(axis=0)

        for i, f in enumerate(x_cols):
            v = float(corrs[i])
            if not np.isnan(v):
                ic_dict[f].append(v)

    return {f: np.array(v) for f, v in ic_dict.items()}


def ic_ir(ics: np.ndarray) -> float:
    if len(ics) < 3:
        return 0.0
    return float(np.mean(ics) / (np.std(ics) + 1e-9))


def feature_auc(df: pd.DataFrame, feat: str, target: str) -> float:
    sub = df[[feat, target]].dropna()
    if len(sub) < 100:
        return 0.5
    y = (sub[target] > 0).astype(int)
    if y.nunique() < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, sub[feat].rank(pct=True)))
    except Exception:
        return 0.5


def group_monotonicity(df: pd.DataFrame, feat: str, target: str, n: int = 5) -> float:
    """Quintile monotonicity: Spearman corr(group_rank, group_mean_return)"""
    sub = df[[feat, target]].dropna().copy()
    try:
        sub["q"] = pd.qcut(sub[feat], n, labels=False, duplicates="drop")
    except Exception:
        return 0.0
    grp = sub.groupby("q")[target].mean()
    if len(grp) < 3:
        return 0.0
    r = np.corrcoef(np.arange(len(grp)), grp.values)[0, 1]
    return float(r) if not np.isnan(r) else 0.0

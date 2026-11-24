"""
TurfGuard AI - analysis helpers.

Pure functions (no Streamlit) so they can be unit-tested and reused in notebooks.
Feature lists match Notebook 07 exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse

LABEL = "recorded_injury_label"

NUMERIC_FEATURES = [
    "Temperature", "tracking_points", "duration", "distance_total",
    "speed_mean", "speed_max", "speed_std", "acceleration_mean",
    "acceleration_abs_mean", "acceleration_abs_max", "x_range", "y_range",
]
CATEGORICAL_FEATURES = [
    "FieldType", "StadiumType", "Weather", "PlayType", "Position",
    "PositionGroup", "RosterPosition", "temperature_category",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Movement metrics used in Notebook 05
MOVEMENT_FEATURES = [
    "speed_mean", "speed_max", "speed_std", "distance_total",
    "acceleration_abs_mean", "acceleration_abs_max", "duration",
    "x_range", "y_range",
]

# Features that scale with how long the play lasted / how far the player travelled.
# If an injury ends a player's play early, these are the most exposed to leakage.
PLAY_EXTENT_FEATURES = ["tracking_points", "duration", "distance_total", "x_range", "y_range"]


# --------------------------------------------------------------------------
# Labels & derived columns
# --------------------------------------------------------------------------
def add_labels(plays: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    out = plays.copy()
    keys = set(injuries["PlayKey"].dropna().astype(str).unique())
    out[LABEL] = out["PlayKey"].astype(str).isin(keys).astype(int)
    return out


def weather_group(value) -> str:
    """Coarse grouping of the free-text Weather column (heuristic, documented in the app)."""
    if pd.isna(value):
        return "Unknown"
    w = str(value).lower()
    if any(t in w for t in ("rain", "shower", "snow", "drizzle", "storm", "thunder", "sleet")):
        return "Precipitation"
    if any(t in w for t in ("indoor", "controlled", "dome", "roof closed", "retractable")):
        return "Indoor / controlled"
    if any(t in w for t in ("cloud", "overcast", "haz", "fog", "mist")):
        return "Cloudy / hazy"
    if any(t in w for t in ("clear", "sunny", "fair", "bright")):
        return "Clear / sunny"
    return "Other"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Weather" in out.columns:
        out["weather_group"] = out["Weather"].map(weather_group)
    if "FieldType" in out.columns:
        out["FieldType"] = out["FieldType"].astype(str).str.strip().str.lower()
    for c in MOVEMENT_FEATURES + ["Temperature", "acceleration_mean", "tracking_points"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def rank_biserial(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """
    Mann-Whitney U test plus rank-biserial effect size.
    effect = P(x > y) - P(x < y); positive means x tends to be larger. Range -1..1.
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan, np.nan)
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    effect = 2 * res.statistic / (len(x) * len(y)) - 1
    return (float(res.statistic), float(res.pvalue), float(effect))


def effect_label(effect: float) -> str:
    if pd.isna(effect):
        return "n/a"
    a = abs(effect)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "medium"
    return "large"


def compare_surfaces(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Synthetic vs natural movement comparison with effect sizes and FDR correction."""
    nat = df[df["FieldType"] == "natural"]
    syn = df[df["FieldType"] == "synthetic"]
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        _, p, eff = rank_biserial(syn[f], nat[f])
        rows.append({
            "Metric": f,
            "Natural median": nat[f].median(),
            "Synthetic median": syn[f].median(),
            "Natural mean": nat[f].mean(),
            "Synthetic mean": syn[f].mean(),
            "n natural": int(nat[f].notna().sum()),
            "n synthetic": int(syn[f].notna().sum()),
            "p-value": p,
            "Effect size (+ = synthetic higher)": eff,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["q-value (BH)"] = bh_fdr(out["p-value"])
        out["Effect magnitude"] = out["Effect size (+ = synthetic higher)"].map(effect_label)
    return out


def stratified_surface_effects(df: pd.DataFrame, strat: str, metric: str, min_n: int = 200) -> pd.DataFrame:
    rows = []
    for level, g in df.groupby(strat, dropna=False, observed=True):
        nat = g.loc[g["FieldType"] == "natural", metric]
        syn = g.loc[g["FieldType"] == "synthetic", metric]
        if nat.notna().sum() < min_n or syn.notna().sum() < min_n:
            continue
        _, p, eff = rank_biserial(syn, nat)
        rows.append({
            strat: level,
            "n natural": int(nat.notna().sum()),
            "n synthetic": int(syn.notna().sum()),
            "Natural median": nat.median(),
            "Synthetic median": syn.median(),
            "p-value": p,
            "Effect size (+ = synthetic higher)": eff,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["q-value (BH)"] = bh_fdr(out["p-value"])
        out["Effect magnitude"] = out["Effect size (+ = synthetic higher)"].map(effect_label)
    return out


def injured_vs_not(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    inj = df[df[LABEL] == 1]
    non = df[df[LABEL] == 0]
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        _, p, eff = rank_biserial(inj[f], non[f])
        rows.append({
            "Metric": f,
            "Matched-injury median": inj[f].median(),
            "No-match median": non[f].median(),
            "n injury": int(inj[f].notna().sum()),
            "p-value": p,
            "Effect size (+ = injury higher)": eff,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["q-value (BH)"] = bh_fdr(out["p-value"])
        out["Effect magnitude"] = out["Effect size (+ = injury higher)"].map(effect_label)
    return out


def rate_table(df: pd.DataFrame, by: list[str], min_plays: int = 0) -> pd.DataFrame:
    g = (df.groupby(by, dropna=False, observed=True)[LABEL]
           .agg(plays="size", injuries="sum").reset_index())
    g = g[g["plays"] >= min_plays].copy()
    if g.empty:
        return g.assign(rate_per_10k=[], ci_low_per_10k=[], ci_high_per_10k=[])
    g["rate_per_10k"] = g["injuries"] / g["plays"] * 1e4
    cis = [wilson_ci(k, n) for k, n in zip(g["injuries"], g["plays"])]
    g["ci_low_per_10k"] = [c[0] * 1e4 for c in cis]
    g["ci_high_per_10k"] = [c[1] * 1e4 for c in cis]
    return g.sort_values("rate_per_10k", ascending=False).reset_index(drop=True)


def surface_rate_ratio(df: pd.DataFrame) -> dict | None:
    """Synthetic-vs-natural recorded-injury rate ratio with 95% CI and Fisher exact p."""
    s = df[df["FieldType"] == "synthetic"]
    n = df[df["FieldType"] == "natural"]
    a, n1 = int(s[LABEL].sum()), len(s)
    b, n2 = int(n[LABEL].sum()), len(n)
    if min(n1, n2) == 0:
        return None
    out = {"synthetic_injuries": a, "synthetic_plays": n1, "natural_injuries": b, "natural_plays": n2}
    if a > 0 and b > 0:
        rr = (a / n1) / (b / n2)
        se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n2)
        out.update(rate_ratio=rr, ci_low=float(np.exp(np.log(rr) - 1.96 * se)),
                   ci_high=float(np.exp(np.log(rr) + 1.96 * se)))
    else:
        out.update(rate_ratio=np.nan, ci_low=np.nan, ci_high=np.nan)
    out["fisher_p"] = float(stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1])
    return out


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
def make_pipeline(num_cols: list[str], cat_cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                              ("scaler", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000,
                                          solver="liblinear", random_state=42)),
    ])


def grouped_cv_scores(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
                      n_splits: int = 5, seed: int = 42) -> dict:
    """Out-of-fold ROC-AUC / PR-AUC with GameID-grouped, stratified folds (as in Notebook 07)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    d = df[df["GameID"].notna()].reset_index(drop=True)
    X = d[num_cols + cat_cols].replace([np.inf, -np.inf], np.nan)
    y = d[LABEL].astype(int)
    groups = d["GameID"].astype(str)
    oof = np.full(len(d), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        m = make_pipeline(num_cols, cat_cols)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "prevalence": float(y.mean()),
        "n": int(len(d)),
        "positives": int(y.sum()),
    }


def leakage_ablation(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Re-run grouped CV on feature subsets to see where the signal comes from."""
    context_num = ["Temperature"]
    context_cat = [c for c in CATEGORICAL_FEATURES]
    movement_num = [c for c in NUMERIC_FEATURES if c != "Temperature"]
    safer_num = [c for c in NUMERIC_FEATURES if c not in PLAY_EXTENT_FEATURES]
    sets = {
        "All 20 features (current model)": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "Without play-extent features": (safer_num, CATEGORICAL_FEATURES),
        "Context only (no movement)": (context_num, context_cat),
        "Movement only (no context)": (movement_num, []),
        "Play-extent features only": (PLAY_EXTENT_FEATURES, []),
    }
    rows = []
    for name, (num, cat) in sets.items():
        s = grouped_cv_scores(df, num, cat, n_splits=n_splits)
        rows.append({"Feature set": name, "ROC-AUC": s["roc_auc"], "PR-AUC": s["pr_auc"],
                     "Prevalence (PR-AUC baseline)": s["prevalence"],
                     "PR-AUC lift vs baseline": s["pr_auc"] / s["prevalence"] if s["prevalence"] else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
def _original_feature(transformed_name: str) -> str:
    name = transformed_name.split("__", 1)[-1]
    best = None
    for c in CATEGORICAL_FEATURES:
        if name.startswith(c + "_") and (best is None or len(c) > len(best)):
            best = c
    return best or name


def model_coefficients(model) -> pd.DataFrame:
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    return pd.DataFrame({
        "Encoded feature": [n.split("__", 1)[-1] for n in names],
        "Original feature": [_original_feature(n) for n in names],
        "Coefficient": coefs,
        "Odds ratio per +1 SD / category present": np.exp(coefs),
    })


def record_contributions(model, row: pd.DataFrame) -> pd.DataFrame:
    """Per-original-feature contribution (log-odds) for a single record."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    xt = pre.transform(row[FEATURES].replace([np.inf, -np.inf], np.nan))
    xt = xt.toarray()[0] if issparse(xt) else np.asarray(xt)[0]
    names = pre.get_feature_names_out()
    contrib = xt * clf.coef_[0]
    df = pd.DataFrame({"Feature": [_original_feature(n) for n in names], "Contribution": contrib})
    out = df.groupby("Feature", as_index=False)["Contribution"].sum()
    out["abs"] = out["Contribution"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    out.attrs["intercept"] = float(clf.intercept_[0])
    return out


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
def anomaly_scores(df: pd.DataFrame, features: list[str], normalize_by: str | None = None,
                   fit_sample: int = 50_000, n_estimators: int = 200, seed: int = 42) -> pd.Series:
    """
    Isolation Forest anomaly score (higher = more unusual) for every row.
    Optionally z-scores each feature within a group (e.g. PositionGroup) first so that
    'unusual' means unusual *for that kind of player*, not just 'a quarterback vs a receiver'.
    """
    from sklearn.ensemble import IsolationForest

    X = df[features].replace([np.inf, -np.inf], np.nan).astype(float)
    X = X.fillna(X.median())
    if normalize_by and normalize_by in df.columns:
        grp = df[normalize_by].fillna("Unknown").astype(str)
        mu = X.groupby(grp).transform("mean")
        sd = X.groupby(grp).transform("std").replace(0, 1).fillna(1)
        X = (X - mu) / sd
    fit_idx = X.sample(min(fit_sample, len(X)), random_state=seed).index
    iso = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    iso.fit(X.loc[fit_idx])
    return pd.Series(-iso.score_samples(X), index=df.index, name="anomaly_score")


def lift_table(scores: pd.Series, labels: pd.Series,
               fractions=(0.01, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    order = scores.sort_values(ascending=False).index
    lab = labels.loc[order].to_numpy()
    total = lab.sum()
    rows = []
    for f in fractions:
        k = max(1, int(round(len(lab) * f)))
        caught = int(lab[:k].sum())
        share = caught / total if total else np.nan
        rows.append({"Top fraction flagged": f"{f:.0%}", "Records flagged": k,
                     "Matched injuries inside": caught,
                     "Share of all matched injuries": share,
                     "Lift (random = 1.0)": share / f if total else np.nan})
    return pd.DataFrame(rows)

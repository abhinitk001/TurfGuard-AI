import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import analysis as ta

st.set_page_config(page_title="TurfGuard AI", page_icon="🏈", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------- CONFIG ----------------
MODEL_PATH = os.getenv("TURFGUARD_MODEL", "models/turfguard_logistic_model.pkl")
DATA_PATH = os.getenv("TURFGUARD_DATA", "data/processed/play_level_features.csv")
INJURY_PATH = os.getenv("TURFGUARD_INJURIES", "data/processed/injury_enriched_features.csv")

LABEL = ta.LABEL
FEATURES = ta.FEATURES

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.35rem 1.5rem; border-radius: 16px;
       background: linear-gradient(115deg, #101d35 0%, #173d50 100%);
       color: white; margin-bottom: 1.1rem;}
.hero h1 {font-size: 2.1rem; margin: 0 0 .25rem 0;}
.hero p {color: #d8e5ef; margin: 0;}
</style>
""", unsafe_allow_html=True)


# ---------------- LOADING ----------------
@st.cache_resource
def load_model(path, mtime):
    return joblib.load(path)


@st.cache_data
def load_data(data_path, injury_path, mtime_a, mtime_b):
    plays = pd.read_csv(data_path, low_memory=False)
    injuries = pd.read_csv(injury_path, low_memory=False)
    plays = ta.add_derived_columns(plays)
    return ta.add_labels(plays, injuries), injuries


@st.cache_data(show_spinner=False)
def cached_surface_tests(_df, key):
    return ta.compare_surfaces(_df, ta.MOVEMENT_FEATURES)


@st.cache_data(show_spinner=False)
def cached_injury_tests(_df, key):
    return ta.injured_vs_not(_df, ta.MOVEMENT_FEATURES)


@st.cache_data(show_spinner="Scoring all records…")
def cached_scores(_model, _df, key):
    X = _df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return pd.Series(_model.predict_proba(X)[:, 1], index=_df.index, name="model_score")


@st.cache_data(show_spinner="Fitting Isolation Forest…")
def cached_anomaly(_df, features, normalize_by, key):
    return ta.anomaly_scores(_df, list(features), normalize_by)


@st.cache_data(show_spinner="Running grouped cross-validation on 5 feature sets (can take a few minutes)…")
def cached_ablation(_df, key):
    return ta.leakage_ablation(_df)


def fmt_p(p):
    return "<1e-300" if p == 0 else f"{p:.2e}"


# ---------------- SIDEBAR ----------------
st.sidebar.title("🏈 TurfGuard AI")
PAGES = [
    "Overview",
    "1 · Player Biomechanics",
    "2 · Surface Analytics",
    "3 · Injury Patterns",
    "4 · ML Classifier & Explainability",
    "5 · Movement Anomaly Detection",
    "Leakage Audit",
    "Model Evaluation (upload)",
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.divider()
st.sidebar.caption("Question: how do surface, conditions and biomechanics interact, and which movement patterns are associated with recorded lower-limb injuries?")
st.sidebar.caption("Research prototype — not a medical tool")

st.markdown("""
<div class="hero">
  <h1>🏈 TurfGuard AI</h1>
  <p>NFL playing-surface &amp; player-biomechanics analytics — explainable, leakage-aware</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD ASSETS ----------------
missing_paths = [p for p in [MODEL_PATH, DATA_PATH, INJURY_PATH] if not os.path.exists(p)]
if missing_paths:
    st.error("Some project files could not be found: " + ", ".join(missing_paths))
    st.code(
        "project/\n├── app.py\n├── analysis.py\n├── models/\n│   └── turfguard_logistic_model.pkl\n"
        "└── data/processed/\n    ├── play_level_features.csv\n    └── injury_enriched_features.csv"
    )
    st.code('TURFGUARD_MODEL="path/to/turfguard_logistic_model.pkl"\n'
            'TURFGUARD_DATA="path/to/play_level_features.csv"\n'
            'TURFGUARD_INJURIES="path/to/injury_enriched_features.csv"')
    st.stop()

try:
    model = load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))
    data, injuries = load_data(DATA_PATH, INJURY_PATH,
                               os.path.getmtime(DATA_PATH), os.path.getmtime(INJURY_PATH))
except Exception as exc:
    st.exception(exc)
    st.stop()

DATA_KEY = f"{len(data)}-{int(data[LABEL].sum())}"
surface = data[data["FieldType"].isin(["natural", "synthetic"])].copy()

# =====================================================================
# OVERVIEW
# =====================================================================
if page == "Overview":
    st.subheader("Project overview")
    total, positive = len(data), int(data[LABEL].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player-play records", f"{total:,}")
    c2.metric("Matched injury records", f"{positive:,}")
    c3.metric("Natural-surface records", f"{int((data['FieldType'] == 'natural').sum()):,}")
    c4.metric("Synthetic-surface records", f"{int((data['FieldType'] == 'synthetic').sum()):,}")

    st.info(
        "**Research question.** How do playing surfaces, environmental conditions and player "
        "biomechanics interact, and what movement patterns are associated with recorded "
        "lower-limb injuries? The classifier is one supporting component, not the whole project."
    )
    st.markdown("""
| Module | What it answers |
|---|---|
| **1 · Player Biomechanics** | How do speed, acceleration, distance and field coverage vary by position and play type? |
| **2 · Surface Analytics** | Do movement metrics differ between natural and synthetic turf — and by how much (effect size, not just p-values)? |
| **3 · Injury Patterns** | Which surface / weather / position / temperature combinations have higher matched-injury rates, with confidence intervals? |
| **4 · ML Classifier & Explainability** | What does the Logistic Regression learn, and why is a given record scored high? |
| **5 · Movement Anomaly Detection** | Which plays are unusual *for that position group*, and are matched injuries over-represented among them? |
| **Leakage Audit** | Do whole-play features reveal the injury after the fact rather than predict it? |
""")
    st.warning(
        "A positive label means a play matched an injury record. A missing match does not prove a "
        "player was uninjured. Nothing here predicts future injury or medical risk."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Matched-label distribution")
        lc = pd.DataFrame({"Label": ["No matched record", "Matched injury record"],
                           "Records": [total - positive, positive]})
        fig = px.bar(lc, x="Label", y="Records", color="Label", log_y=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Log scale — the positive class is a tiny fraction of records.")
    with right:
        st.markdown("#### Playing-surface distribution")
        counts = data["FieldType"].fillna("Unknown").value_counts().rename_axis("Surface").reset_index(name="Records")
        st.plotly_chart(px.pie(counts, names="Surface", values="Records", hole=0.55), width="stretch")
    st.markdown("#### Data preview")
    st.dataframe(data.head(20), width="stretch")

# =====================================================================
# MODULE 1 - BIOMECHANICS
# =====================================================================
elif page == "1 · Player Biomechanics":
    st.subheader("Module 1 · Player biomechanics")
    st.write("Play-level movement summaries derived from 10 Hz tracking data (Notebook 04).")

    m = surface[ta.MOVEMENT_FEATURES]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median mean speed", f"{m['speed_mean'].median():.2f}")
    c2.metric("Median max speed", f"{m['speed_max'].median():.2f}")
    c3.metric("Median |accel| mean", f"{m['acceleration_abs_mean'].median():.2f}")
    c4.metric("Median distance", f"{m['distance_total'].median():.1f}")

    group_options = [c for c in ["PositionGroup", "Position", "PlayType", "weather_group", "temperature_category"] if c in surface.columns]
    a, b = st.columns(2)
    metric = a.selectbox("Movement metric", ta.MOVEMENT_FEATURES, index=1)
    by = b.selectbox("Group by", group_options)
    top_levels = surface[by].value_counts().head(15).index
    plot_df = surface[surface[by].isin(top_levels)]
    st.plotly_chart(px.box(plot_df, x=by, y=metric, points=False, title=f"{metric} by {by}"),
                    width="stretch")
    st.dataframe(surface.groupby(by, observed=True)[metric].agg(["count", "mean", "median", "std"])
                 .sort_values("count", ascending=False).head(20), width="stretch")

    st.markdown("#### Speed vs acceleration")
    sample = surface.sample(min(6000, len(surface)), random_state=1)
    color = st.radio("Colour by", ["PositionGroup", "FieldType"], horizontal=True)
    st.plotly_chart(px.scatter(sample, x="speed_mean", y="acceleration_abs_mean", color=color,
                               opacity=0.45, title="Sample of 6,000 player-plays"),
                    width="stretch")

    st.markdown("#### Correlation between movement metrics")
    corr = surface[ta.MOVEMENT_FEATURES].sample(min(50000, len(surface)), random_state=1).corr(method="spearman")
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Spearman correlation"), width="stretch")
    st.caption(
        "Not yet represented: directional change and sudden-direction-change metrics. Notebook 04 "
        "loads `dir` and `o` but does not use them — adding those features is the main gap "
        "against the competition's movement-characterisation goal."
    )

# =====================================================================
# MODULE 2 - SURFACE ANALYTICS
# =====================================================================
elif page == "2 · Surface Analytics":
    st.subheader("Module 2 · Natural vs synthetic surface")
    st.write(
        f"{len(surface):,} records with a known surface. With samples this large, almost every "
        "p-value is tiny — **read the effect size first**."
    )

    res = cached_surface_tests(surface, DATA_KEY)
    show = res.copy()
    show["p-value"] = show["p-value"].map(fmt_p)
    show["q-value (BH)"] = show["q-value (BH)"].map(fmt_p)
    st.markdown("#### Movement comparison (Mann-Whitney U, Benjamini-Hochberg corrected)")
    st.dataframe(show.drop(columns=["Natural mean", "Synthetic mean"]).round(4), width="stretch")
    eff_col = "Effect size (+ = synthetic higher)"
    fig = px.bar(res, x="Metric", y=eff_col, color="Effect magnitude",
                 title="Effect size by metric (rank-biserial)")
    fig.add_hline(y=0, line_color="gray")
    st.plotly_chart(fig, width="stretch")
    st.caption("Effect size = P(synthetic > natural) − P(synthetic < natural). |0.1| small, |0.3| medium, |0.5| large. "
               "Statistically significant ≠ practically meaningful, and these are not causal estimates: "
               "teams, stadium climates and game types differ by surface.")

    st.markdown("#### Does the surface effect depend on context?")
    strat_options = [c for c in ["PositionGroup", "PlayType", "weather_group", "temperature_category", "StadiumType"] if c in surface.columns]
    a, b, c = st.columns(3)
    strat = a.selectbox("Split by", strat_options)
    metric = b.selectbox("Metric", ta.MOVEMENT_FEATURES, key="strat_metric")
    min_n = c.slider("Min records per surface per level", 50, 5000, 300, step=50)
    st_res = ta.stratified_surface_effects(surface, strat, metric, min_n)
    if st_res.empty:
        st.info("No levels meet the minimum sample size. Lower the threshold.")
    else:
        fig = px.bar(st_res, x=strat, y="Effect size (+ = synthetic higher)", color="Effect magnitude",
                     hover_data=["n natural", "n synthetic"], title=f"Synthetic-vs-natural effect on {metric}")
        fig.add_hline(y=0, line_color="gray")
        st.plotly_chart(fig, width="stretch")
        s2 = st_res.copy()
        s2["p-value"] = s2["p-value"].map(fmt_p)
        s2["q-value (BH)"] = s2["q-value (BH)"].map(fmt_p)
        st.dataframe(s2.round(4), width="stretch")

    st.markdown("#### Matched-injury rate by surface")
    rt = ta.rate_table(surface, ["FieldType"])
    st.dataframe(rt.round(3), width="stretch")
    rr = ta.surface_rate_ratio(surface)
    if rr:
        x1, x2, x3 = st.columns(3)
        x1.metric("Rate ratio (synthetic / natural)", "n/a" if np.isnan(rr["rate_ratio"]) else f"{rr['rate_ratio']:.2f}")
        x2.metric("95% CI", "n/a" if np.isnan(rr["ci_low"]) else f"{rr['ci_low']:.2f} – {rr['ci_high']:.2f}")
        x3.metric("Fisher exact p", f"{rr['fisher_p']:.3f}")
        st.caption("Crude rate ratio per player-play. It does not adjust for position, play type, weather or exposure mix, "
                   "and matched labels are very few — a wide CI means the data cannot settle the question.")

# =====================================================================
# MODULE 3 - INJURY PATTERNS
# =====================================================================
elif page == "3 · Injury Patterns":
    st.subheader("Module 3 · Injury-pattern investigation")
    n_pos = int(surface[LABEL].sum())
    st.warning(f"Only **{n_pos}** matched injury records have a known surface. Treat every subgroup result as "
               "hypothesis-generating, and read the confidence intervals.")

    st.markdown("#### A. Do matched-injury plays move differently from the rest?")
    inj_res = cached_injury_tests(surface, DATA_KEY)
    s = inj_res.copy()
    s["p-value"] = s["p-value"].map(fmt_p)
    s["q-value (BH)"] = s["q-value (BH)"].map(fmt_p)
    st.dataframe(s.round(4), width="stretch")
    fig = px.bar(inj_res, x="Metric", y="Effect size (+ = injury higher)", color="Effect magnitude")
    fig.add_hline(y=0, line_color="gray")
    st.plotly_chart(fig, width="stretch")
    st.caption("If duration, tracking points or distance show the largest effects, suspect post-injury leakage — "
               "see the Leakage Audit page.")

    st.markdown("#### B. Condition explorer")
    dims = [c for c in ["FieldType", "PositionGroup", "Position", "PlayType", "weather_group",
                        "temperature_category", "StadiumType"] if c in surface.columns]
    a, b, c = st.columns(3)
    d1 = a.selectbox("Dimension 1", dims, index=0)
    d2 = b.selectbox("Dimension 2 (optional)", ["—"] + [d for d in dims if d != d1], index=0)
    min_plays = c.slider("Min plays per cell", 100, 20000, 2000, step=100)
    by = [d1] if d2 == "—" else [d1, d2]
    tbl = ta.rate_table(surface, by, min_plays)
    if tbl.empty:
        st.info("No combinations meet the minimum plays threshold.")
    else:
        tbl["reliability"] = np.where(tbl["injuries"] < 3, "⚠ <3 injuries", "")
        st.dataframe(tbl.round(3), width="stretch")
        if d2 != "—":
            pv = tbl.pivot(index=d1, columns=d2, values="rate_per_10k")
            st.plotly_chart(px.imshow(pv, text_auto=".1f", color_continuous_scale="Reds",
                                      title="Matched injuries per 10,000 plays"), width="stretch")
        else:
            fig = px.bar(tbl, x=d1, y="rate_per_10k", error_y=tbl["ci_high_per_10k"] - tbl["rate_per_10k"],
                         error_y_minus=tbl["rate_per_10k"] - tbl["ci_low_per_10k"], hover_data=["plays", "injuries"])
            fig.update_layout(yaxis_title="Matched injuries per 10,000 plays (95% Wilson CI)")
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### C. Recorded injuries by body part and surface")
    if {"BodyPart", "FieldType"}.issubset(injuries.columns):
        inj = injuries.copy()
        inj["FieldType"] = inj["FieldType"].astype(str).str.lower()
        ct = pd.crosstab(inj["BodyPart"], inj["FieldType"])
        st.plotly_chart(px.bar(ct.reset_index().melt(id_vars="BodyPart", var_name="Surface", value_name="Records"),
                               x="BodyPart", y="Records", color="Surface", barmode="group"), width="stretch")
    else:
        st.info("BodyPart / FieldType columns not found in the injury file.")

# =====================================================================
# MODULE 4 - ML + EXPLAINABILITY
# =====================================================================
elif page == "4 · ML Classifier & Explainability":
    st.subheader("Module 4 · Exploratory classifier & explainability")
    st.warning("Scores are model estimates for the *matched-record label*. They are not injury probabilities, "
               "diagnoses or predictions of future injury. The saved model was fit on all rows, so scoring the "
               "dataset here is **in-sample**; use the Leakage Audit / Evaluation pages for honest performance.")

    scores = cached_scores(model, data, DATA_KEY)
    scored = data.assign(model_score=scores)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean score", f"{scores.mean():.4f}")
    c2.metric("Max score", f"{scores.max():.4f}")
    c3.metric("Records ≥ 0.5", f"{int((scores >= 0.5).sum()):,}")
    st.plotly_chart(px.histogram(scored, x="model_score", nbins=50, log_y=True, title="Distribution of model scores"),
                    width="stretch")
    st.caption("Class-weighted training pushes scores up, so a 0.5 cut-off flags many more plays than there are labels.")

    st.markdown("#### What the model learned")
    coefs = ta.model_coefficients(model)
    n_show = st.slider("Features to show", 5, 30, 15)
    top = coefs.reindex(coefs["Coefficient"].abs().sort_values(ascending=False).index).head(n_show)
    fig = px.bar(top.sort_values("Coefficient"), x="Coefficient", y="Encoded feature", orientation="h",
                 color="Coefficient", color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
    st.plotly_chart(fig, width="stretch")
    agg = coefs.assign(abs_coef=coefs["Coefficient"].abs()).groupby("Original feature")["abs_coef"].sum().sort_values(ascending=False)
    st.plotly_chart(px.bar(agg.reset_index(), x="abs_coef", y="Original feature", orientation="h",
                           title="Total |coefficient| per original feature").update_yaxes(autorange="reversed"),
                    width="stretch")
    st.caption("Numeric coefficients are per 1 SD (standardised). Correlated features share credit, so read these as "
               "associations within this model, not causal drivers.")

    st.markdown("#### Why was this record scored this way?")
    ranked = scored.sort_values("model_score", ascending=False)
    pool = ranked.head(500)
    key_col = "PlayKey"
    choice = st.selectbox("Pick one of the 500 highest-scoring records", pool[key_col].astype(str).tolist())
    row = scored[scored[key_col].astype(str) == choice].head(1)
    contrib = ta.record_contributions(model, row)
    b1, b2 = st.columns([2, 1])
    with b1:
        fig = px.bar(contrib.head(12).sort_values("Contribution"), x="Contribution", y="Feature", orientation="h",
                     color="Contribution", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                     title="Contribution to log-odds (intercept excluded)")
        st.plotly_chart(fig, width="stretch")
    with b2:
        st.metric("Model score", f"{float(row['model_score'].iloc[0]):.4f}")
        st.metric("Matched injury label", int(row[LABEL].iloc[0]))
        st.caption(f"Intercept: {contrib.attrs['intercept']:.3f}")
    show_cols = [c for c in ["PlayKey", "GameID", "FieldType", "PositionGroup", "weather_group",
                             "speed_mean", "speed_max", "acceleration_abs_max", "duration",
                             "model_score", LABEL] if c in ranked.columns]
    st.markdown("#### Highest-scoring records (for research review)")
    st.dataframe(ranked[show_cols].head(100), width="stretch")
    st.download_button("Download all scored records", ranked.to_csv(index=False).encode("utf-8"),
                       file_name="turfguard_scored_records.csv", mime="text/csv")

# =====================================================================
# MODULE 5 - ANOMALY DETECTION
# =====================================================================
elif page == "5 · Movement Anomaly Detection":
    st.subheader("Module 5 · Movement anomaly detection")
    st.write(
        "An Isolation Forest is fit on movement features only — **no labels are used**. We then ask whether "
        "matched-injury plays are over-represented among the most unusual plays."
    )
    feats = st.multiselect("Movement features", ta.MOVEMENT_FEATURES,
                           default=["speed_mean", "speed_max", "speed_std", "acceleration_abs_mean",
                                    "acceleration_abs_max", "x_range", "y_range"])
    norm_options = ["None"] + [c for c in ["PositionGroup", "Position", "PlayType"] if c in data.columns]
    norm = st.selectbox("Compare each play against its own…", norm_options, index=1 if "PositionGroup" in norm_options else 0,
                        help="Z-scores features within the group first, so 'unusual' means unusual for that kind of player.")
    if len(feats) < 2:
        st.info("Select at least two features.")
        st.stop()
    an = cached_anomaly(data, tuple(feats), None if norm == "None" else norm, DATA_KEY)
    scored = data.assign(anomaly_score=an)

    st.plotly_chart(px.histogram(scored.sample(min(50000, len(scored)), random_state=1), x="anomaly_score",
                                 nbins=60, title="Anomaly score distribution (higher = more unusual)"),
                    width="stretch")

    st.markdown("#### Are matched-injury plays over-represented among anomalies?")
    lt = ta.lift_table(an, data[LABEL])
    st.dataframe(lt.round(3), width="stretch")
    st.caption("Lift of 1.0 means no better than random. With very few positives, lift estimates are noisy. "
               "If the lift is high mainly because injured plays are shorter or truncated, that is the leakage "
               "problem again — try removing `duration`/`distance_total` above.")

    c1, c2 = st.columns(2)
    with c1:
        by_surface = scored[scored["FieldType"].isin(["natural", "synthetic"])]
        st.plotly_chart(px.box(by_surface, x="FieldType", y="anomaly_score", points=False,
                               title="Anomaly score by surface"), width="stretch")
    with c2:
        st.plotly_chart(px.box(scored, x=scored[LABEL].map({0: "No match", 1: "Matched injury"}), y="anomaly_score",
                               points=False, title="Anomaly score by matched label").update_xaxes(title=""),
                        width="stretch")

    thr = st.slider("Flag top % as anomalous", 0.5, 10.0, 1.0, 0.5)
    cut = an.quantile(1 - thr / 100)
    flagged = scored[scored["anomaly_score"] >= cut]
    st.metric("Flagged plays", f"{len(flagged):,}")
    if "FieldType" in flagged:
        comp = pd.DataFrame({
            "Flagged share": flagged["FieldType"].value_counts(normalize=True),
            "Overall share": scored["FieldType"].value_counts(normalize=True),
        }).dropna()
        st.dataframe(comp.round(4), width="stretch")
    cols = [c for c in ["PlayKey", "GameID", "FieldType", "PositionGroup", "PlayType", "speed_max",
                        "acceleration_abs_max", "duration", "anomaly_score", LABEL] if c in flagged.columns]
    st.dataframe(flagged.sort_values("anomaly_score", ascending=False)[cols].head(200), width="stretch")

# =====================================================================
# LEAKAGE AUDIT
# =====================================================================
elif page == "Leakage Audit":
    st.subheader("Leakage audit")
    st.write(
        "Notebook 04 aggregates tracking over the **whole play**. If an injury ends a play early (the player "
        "goes down or stops), features like `duration`, `tracking_points` and `distance_total` can encode the "
        "injury itself rather than anything that preceded it. This page tests for that."
    )

    st.markdown("#### A. Do matched-injury plays look truncated?")
    check = [f for f in ["duration", "tracking_points", "distance_total"] if f in data.columns]
    rows = []
    for f in check:
        _, p, eff = ta.rank_biserial(data.loc[data[LABEL] == 1, f], data.loc[data[LABEL] == 0, f])
        rows.append({"Feature": f,
                     "Matched-injury median": data.loc[data[LABEL] == 1, f].median(),
                     "No-match median": data.loc[data[LABEL] == 0, f].median(),
                     "Effect (+ = injury higher)": eff, "p-value": fmt_p(p)})
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
    feat_plot = st.selectbox("Distribution", check)
    plot_df = data.assign(Label=data[LABEL].map({0: "No match", 1: "Matched injury"}))
    st.plotly_chart(px.histogram(plot_df, x=feat_plot, color="Label", histnorm="probability density",
                                 barmode="overlay", opacity=0.6, nbins=60, marginal="box"),
                    width="stretch")
    st.caption("A clear shift towards shorter plays for matched injuries is a red flag. No shift does not prove "
               "absence of leakage, since other whole-play features can still be affected.")

    st.markdown("#### B. Where does the model's signal come from?")
    st.write("Grouped-by-game, 5-fold cross-validation on feature subsets. If performance collapses once "
             "play-extent features are dropped, the headline score was mostly leakage or play length.")
    if st.button("Run feature-set ablation", type="primary"):
        st.session_state["ablation"] = cached_ablation(data, DATA_KEY)
    if "ablation" in st.session_state:
        ab = st.session_state["ablation"]
        st.dataframe(ab.round(4), width="stretch")
        long = ab.melt(id_vars="Feature set", value_vars=["ROC-AUC", "PR-AUC"], var_name="Metric", value_name="Value")
        st.plotly_chart(px.bar(long, x="Feature set", y="Value", color="Metric", barmode="group"),
                        width="stretch")
        st.caption("ROC-AUC 0.5 = chance. PR-AUC should be judged against prevalence (shown in the table), not against 0.5.")

    st.markdown("#### C. A cleaner design (needs Notebook 04 re-run)")
    st.markdown("""
- Recompute features from a **fixed early window** (e.g. the first 2–3 s of each play) so every record contributes the same amount of tracking, regardless of when the play ended.
- Restrict to plays that last at least as long as that window, for both injured and non-injured players.
- Add directional-change features from `dir` / `o` using the same window.
- Report results with and without play-extent features, and keep every split grouped by `GameID` (ideally by player as well).
""")

# =====================================================================
# MODEL EVALUATION (UPLOAD)
# =====================================================================
elif page == "Model Evaluation (upload)":
    st.subheader("Model evaluation")
    st.write("Upload an evaluation CSV containing the 20 model features and a `recorded_injury_label` column.")
    uploaded = st.file_uploader("Upload held-out evaluation data", type=["csv"])
    if uploaded is not None:
        try:
            eval_data = pd.read_csv(uploaded)
            if LABEL not in eval_data:
                st.error("The uploaded CSV must include recorded_injury_label.")
            else:
                missing = [c for c in FEATURES if c not in eval_data.columns]
                if missing:
                    st.error("Missing model features: " + ", ".join(missing))
                else:
                    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                                 precision_score, recall_score, roc_auc_score)
                    X = eval_data[FEATURES].replace([np.inf, -np.inf], np.nan)
                    y_score = pd.Series(model.predict_proba(X)[:, 1])
                    y_true = eval_data[LABEL].astype(int)
                    threshold = st.slider("Classification threshold", 0.01, 0.99, 0.50, 0.01)
                    y_pred = (y_score >= threshold).astype(int)
                    a, b, c, d = st.columns(4)
                    a.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
                    b.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
                    c.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
                    if y_true.nunique() == 2:
                        d.metric("PR-AUC", f"{average_precision_score(y_true, y_score):.6f}")
                        st.metric("ROC-AUC", f"{roc_auc_score(y_true, y_score):.4f}")
                        st.caption(f"PR-AUC baseline (prevalence): {y_true.mean():.6f}")
                    else:
                        d.metric("PR-AUC", "Unavailable")
                        st.warning("ROC-AUC and PR-AUC require both classes in the evaluation data.")
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted 0", "Predicted 1"],
                                              y=["Actual 0", "Actual 1"], title="Confusion matrix"),
                                    width="stretch")
                    st.caption("Use genuinely held-out, game-grouped data. Do not evaluate on training rows.")
        except Exception as exc:
            st.exception(exc)

st.sidebar.divider()
st.sidebar.caption("TurfGuard AI v2.0 • Research prototype")

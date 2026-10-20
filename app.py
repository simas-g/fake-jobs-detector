"""Fake Jobs Detector: Streamlit app for the trust and safety team.

Run:  streamlit run app.py
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

import db
from features import CATEGORIES, FEATURES, prepare
from knn import KNN

st.set_page_config(page_title="Fake Jobs Detector", page_icon="🔎", layout="wide")

SCAM_COLOR, LEGIT_COLOR = "#C2571A", "#2B5F9E"

EXAMPLES = {
    "Example: likely scam": dict(
        has_salary=True, salary_min=2500, salary_max=7500, required_experience_years=0.5,
        skills_listed=6, vague_word_count=5, exclamation_marks=3, allcaps_ratio=0.09,
        posting_hour=3, employer_response_delay_h=4.0, is_remote=True, free_mail_contact=True,
        company_age_months=12, company_size=4, category="finance", mentions_trial_or_payment=True),
    "Example: likely legitimate": dict(
        has_salary=True, salary_min=4800, salary_max=5900, required_experience_years=6.0,
        skills_listed=18, vague_word_count=2, exclamation_marks=1, allcaps_ratio=0.015,
        posting_hour=14, employer_response_delay_h=15.0, is_remote=False, free_mail_contact=False,
        company_age_months=60, company_size=25, category="sales", mentions_trial_or_payment=False),
    # borderline: 2 of 3 neighbours are scams; moving the hour to 21 flips the vote
    "Example: borderline, no pay range": dict(
        has_salary=False, salary_min=4000, salary_max=5000, required_experience_years=1.6,
        skills_listed=9, vague_word_count=3, exclamation_marks=2, allcaps_ratio=0.02,
        posting_hour=22, employer_response_delay_h=6.0, is_remote=True, free_mail_contact=False,
        company_age_months=20, company_size=8, category="marketing", mentions_trial_or_payment=False),
}

FEATURE_LABELS = {
    "salary_width": "salary width (max ÷ min)",
    "no_salary": "no pay range",
    "required_experience_years": "experience (years)",
    "skills_listed": "skills listed",
    "allcaps_ratio": "capitals ratio",
    "hour_sin": "posting hour (sin)",
    "hour_cos": "posting hour (cos)",
    "vague_word_count": "vague words",
}


@st.cache_resource
def get_connection():
    return db.connect()


@st.cache_data
def query(name):
    conn, _ = get_connection()
    q = db.load_queries()[name]
    return q["title"], q["sql"], db.run_query(conn, q["sql"])


@st.cache_resource
def get_model():
    """Fit on the whole reviewed backlog: every stored posting can be a neighbour."""
    _, _, backlog = query("all_postings")
    model = KNN(k=3, scaling="standard").fit(prepare(backlog), backlog["scam"])
    return model, backlog


def pct(x):
    return f"{x:.0%}" if pd.notna(x) else "–"


def show_query(name, chart=None):
    title, sql, df = query(name)
    st.subheader(title)
    if chart is not None:
        st.altair_chart(chart(df), width="stretch")
    shown = df.copy()
    for c in shown.columns:
        if "rate" in c:
            shown[c] = shown[c].map(pct)
    st.dataframe(shown, hide_index=True, width="stretch")
    with st.expander("SQL"):
        st.code(sql, language="sql")


def rate_bars(label_col):
    def chart(df):
        d = df.copy()
        d[label_col] = d[label_col].astype(str).str.replace(r"^\d ", "", regex=True)
        return (
            alt.Chart(d)
            .mark_bar(color=SCAM_COLOR, cornerRadiusEnd=3)
            .encode(
                x=alt.X("scam_rate:Q", title="scam rate", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y(f"{label_col}:N", sort=None, title=None),
                tooltip=[label_col, "postings", "scams",
                         alt.Tooltip("scam_rate:Q", format=".0%")],
            )
            .properties(height=36 * len(d) + 20)
        )
    return chart


# ---------------------------------------------------------------- layout
conn, backend = get_connection()
st.title("🔎 Fake Jobs Detector")
st.caption("Moderation support for the trust and safety team: which listing traits go with "
           "confirmed scams, and a verdict for new postings, explained by the 3 most similar "
           "reviewed postings.")
if backend.startswith("SQLite"):
    st.warning(f"Database: {backend}. Start PostgreSQL and run `python load_data.py`.")
else:
    st.sidebar.success(f"Connected to {backend}")
st.sidebar.markdown("**Backlog**")
_, _, ov = query("overview")
st.sidebar.metric("Reviewed postings", int(ov.postings[0]))
st.sidebar.metric("Confirmed scams", f"{int(ov.confirmed_scams[0])} ({pct(ov.scam_rate[0])})")

tab_check, tab_backlog, tab_model = st.tabs(
    ["Check a posting", "Backlog insights", "Model performance"])

# ---------------------------------------------------------------- check a posting
with tab_check:
    model, backlog = get_model()

    cols = st.columns(len(EXAMPLES) + 1)
    for col, name in zip(cols, EXAMPLES):
        if col.button(name, width="stretch"):
            for key, value in EXAMPLES[name].items():
                st.session_state[key] = value
    if cols[-1].button("Clear", width="stretch"):
        for key in EXAMPLES[next(iter(EXAMPLES))]:
            st.session_state.pop(key, None)

    defaults = EXAMPLES["Example: likely legitimate"]
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.form("posting"):
        st.markdown("**Pay**")
        c1, c2, c3 = st.columns([1, 1, 1])
        has_salary = c1.checkbox("Pay range published", key="has_salary")
        salary_min = c2.number_input("Minimum salary (€/month)", 0, 50000, step=100, key="salary_min")
        salary_max = c3.number_input("Maximum salary (€/month)", 0, 50000, step=100, key="salary_max")

        st.markdown("**Text of the posting**")
        c1, c2, c3, c4 = st.columns(4)
        exp = c1.number_input("Experience required (years)", 0.0, 40.0, step=0.5, key="required_experience_years")
        skills = c2.number_input("Skills listed", 0, 100, key="skills_listed")
        vague = c3.number_input("Vague words (rockstar, ninja…)", 0, 50, key="vague_word_count")
        excl = c4.number_input("Exclamation marks", 0, 50, key="exclamation_marks")
        c1, c2, c3, _ = st.columns(4)
        caps = c1.number_input("Share of capitals (0–1)", 0.0, 1.0, step=0.005, format="%.3f", key="allcaps_ratio")
        hour = c2.slider("Hour the listing went live", 0, 23, key="posting_hour")
        trial = c3.checkbox("Asks for trial fee / deposit / payment details", key="mentions_trial_or_payment")

        st.markdown("**Employer and company**")
        c1, c2, c3, c4 = st.columns(4)
        delay = c1.number_input("Response delay (hours)", 0.0, 1000.0, step=0.5, key="employer_response_delay_h")
        age = c2.number_input("Company age on the board (months)", 0, 1000, key="company_age_months")
        size = c3.number_input("Claimed company size", 1, 100000, key="company_size")
        category = c4.selectbox("Category", CATEGORIES, key="category")
        c1, c2, _, _ = st.columns(4)
        remote = c1.checkbox("Remote job", key="is_remote")
        freemail = c2.checkbox("Contact via gmail / hotmail", key="free_mail_contact")

        submitted = st.form_submit_button("Check posting", type="primary")

    if submitted:
        if has_salary and salary_min > salary_max:
            st.error("Minimum salary is above the maximum salary.")
            st.stop()
        if has_salary and salary_min == 0:
            st.error("Enter a minimum salary above 0, or untick “Pay range published”.")
            st.stop()
        new = pd.DataFrame([{
            "salary_min": salary_min if has_salary else None,
            "salary_max": salary_max if has_salary else None,
            "required_experience_years": exp, "skills_listed": skills,
            "vague_word_count": vague, "exclamation_marks": excl, "allcaps_ratio": caps,
            "posting_hour": hour, "employer_response_delay_h": delay,
            "is_remote": int(remote), "free_mail_contact": int(freemail),
            "company_age_months": age, "company_size": size, "category": category,
            "mentions_trial_or_payment": int(trial),
        }])
        X_new = prepare(new)
        idx, dist = model.kneighbors(X_new)
        votes = int(backlog["scam"].iloc[idx[0]].sum())
        is_scam = votes >= 2

        if is_scam:
            st.error(f"### Verdict: likely SCAM\n{votes} of the 3 most similar reviewed postings were confirmed scams.")
        else:
            st.success(f"### Verdict: likely LEGITIMATE\n{3 - votes} of the 3 most similar reviewed postings were legitimate.")
        st.caption("Advisory only: a moderator makes the final decision.")

        st.subheader("The 3 closest reviewed postings")
        neighbours = backlog.iloc[idx[0]].copy()
        neighbours.insert(0, "distance", dist[0].round(2))
        neighbours.insert(1, "outcome", neighbours.pop("scam").map({1: "🚩 scam", 0: "✅ legitimate"}))
        show_cols = ["posting_id", "distance", "outcome", "salary_min", "salary_max",
                     "required_experience_years", "skills_listed", "allcaps_ratio",
                     "posting_hour", "vague_word_count", "company_size", "category",
                     "mentions_trial_or_payment"]
        compare = pd.concat([new.assign(posting_id="new", distance=0.0, outcome="?"),
                             neighbours.assign(posting_id=neighbours.posting_id.astype(str))])
        st.dataframe(compare[show_cols], hide_index=True, width="stretch")

        with st.expander("How the distance is computed"):
            st.markdown(
                "Each posting is turned into 8 numbers: " +
                ", ".join(FEATURE_LABELS[f] for f in FEATURES) +
                ". Every number is standardized (minus the backlog mean, divided by the "
                "standard deviation) so no feature dominates, and the Euclidean distance to "
                "every reviewed posting is computed. The 3 closest vote.")
            z = pd.DataFrame(model._scale(X_new), columns=[FEATURE_LABELS[f] for f in FEATURES])
            st.dataframe(z.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- backlog insights
with tab_backlog:
    st.markdown("Every table below is a live SQL query against the moderation backlog.")
    _, _, ov = query("overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviewed postings", int(ov.postings[0]))
    c2.metric("Confirmed scams", int(ov.confirmed_scams[0]))
    c3.metric("Scam rate", pct(ov.scam_rate[0]))
    c4.metric("Without a pay range", int(ov.without_salary[0]))

    title, sql, prof = query("profile_by_outcome")
    st.subheader(title)
    st.dataframe(prof.set_index("outcome").T.round(3), width="stretch")
    with st.expander("SQL"):
        st.code(sql, language="sql")

    left, right = st.columns(2)
    with left:
        show_query("rate_by_posting_time", rate_bars("time_of_day"))
        show_query("rate_by_experience", rate_bars("experience"))
        show_query("largest_claimed_scam_company")
    with right:
        show_query("rate_by_salary_width", rate_bars("salary_range"))
        show_query("rate_by_category", rate_bars("category"))
        show_query("rate_by_flag")

# ---------------------------------------------------------------- model performance
with tab_model:
    path = db.ROOT / "results" / "metrics.json"
    if not path.exists():
        st.info("Run `python evaluate.py` to generate the evaluation results.")
    else:
        r = json.loads(path.read_text())
        t, base = r["test"], r["baseline_always_legit"]
        st.markdown(f"**3-nearest neighbours**, Euclidean distance on 8 standardized features. "
                    f"Chosen with 5-fold cross-validation on {r['n_train']} postings, "
                    f"then tested once on {r['n_test']} held-out postings.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", pct(t["precision"]), help="Of the postings flagged as scam, the share that really are")
        c2.metric("Recall", pct(t["recall"]), help="Of all scams, the share we catch")
        c3.metric("Accuracy", pct(t["accuracy"]))
        c4.metric("Baseline accuracy", pct(base["accuracy"]), help="Always answering “legitimate”")

        st.subheader("Confusion matrix (test set)")
        cm = pd.DataFrame(
            [[t["tn"], t["fp"]], [t["fn"], t["tp"]]],
            index=["actually legitimate", "actually scam"],
            columns=["predicted legitimate", "predicted scam"])
        st.dataframe(cm, width="stretch")

        def cv_table(section):
            return pd.DataFrame(r[section]).T[["accuracy", "precision", "recall"]].astype(float)

        left, right = st.columns(2)
        with left:
            st.subheader("Feature sets (CV, k = 3)")
            st.dataframe(cv_table("feature_sets").map(pct), width="stretch")
            st.subheader("Scaling (CV, all raw columns)")
            st.dataframe(cv_table("scaling").map(pct), width="stretch")
        with right:
            st.subheader("Choice of k (CV)")
            kv = cv_table("k_values").reset_index(names="k").melt("k", var_name="metric")
            st.altair_chart(
                alt.Chart(kv).mark_line(point=True).encode(
                    x=alt.X("k:O"), y=alt.Y("value:Q", scale=alt.Scale(domain=[0.9, 1]),
                                             axis=alt.Axis(format="%"), title=None),
                    color=alt.Color("metric:N", scale=alt.Scale(
                        range=[LEGIT_COLOR, SCAM_COLOR, "#6B7385"]))),
                width="stretch")
        st.caption("A perfect test score on 80 postings does not mean the model is perfect: the "
                   "test set is small and scammers will adapt once traits like the pay range "
                   "are known to be checked.")

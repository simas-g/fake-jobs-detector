"""Fake Jobs Detector.  Run:  streamlit run app.py"""
import json
import pickle

import pandas as pd
import streamlit as st

import db
from features import prepare

st.set_page_config(page_title="Fake Jobs Detector", layout="wide")


@st.cache_data
def load_tables():
    return db.load_tables()


posting, category, source = load_tables()


@st.cache_data
def sql(query):
    return db.run(query, posting, category)


backlog = sql(db.ALL_POSTINGS)
model = pickle.load(open("model/knn.pkl", "rb"))  # trained by evaluate.py
metrics = json.load(open("results/metrics.json"))

st.sidebar.metric("Reviewed postings", len(posting))
st.sidebar.metric("Confirmed scams", int(posting["scam"].sum()))
st.sidebar.caption(f"Data: {source}")

st.title("Fake Jobs Detector")
st.caption("For the trust and safety team: which traits go with scams, and a verdict for a new "
           "posting based on the 3 most similar postings we already reviewed.")

tab_check, tab_backlog, tab_model = st.tabs(["Check a posting", "Backlog insights", "Model performance"])

# ------------------------------------------------------------------ check a posting
with tab_check:
    with st.form("posting"):
        c1, c2, c3 = st.columns(3)
        has_salary = c1.checkbox("Pay range published", value=True)
        salary_min = c2.number_input("Minimum salary (€/month)", 0, 50000, 2500, step=100)
        salary_max = c3.number_input("Maximum salary (€/month)", 0, 50000, 7500, step=100)

        c1, c2, c3, c4 = st.columns(4)
        experience = c1.number_input("Experience required (years)", 0.0, 40.0, 0.5, step=0.5)
        skills = c2.number_input("Skills listed", 0, 100, 6)
        vague = c3.number_input("Vague words (rockstar, ninja...)", 0, 50, 5)
        caps = c4.number_input("Share of capital letters (0-1)", 0.0, 1.0, 0.09, step=0.01)

        hour = st.slider("Hour the posting went live", 0, 23, 3)
        submitted = st.form_submit_button("Check posting")

    if submitted:
        if has_salary and not 0 < salary_min <= salary_max:
            st.error("The minimum salary must be above 0 and not higher than the maximum.")
            st.stop()

        new = pd.DataFrame([{
            "salary_min": salary_min if has_salary else float("nan"),
            "salary_max": salary_max if has_salary else float("nan"),
            "required_experience_years": experience,
            "skills_listed": skills,
            "vague_word_count": vague,
            "allcaps_ratio": caps,
            "posting_hour": hour,
        }])

        rows, distances = model.kneighbors(prepare(new))
        nearest = backlog.iloc[rows[0]].copy()
        nearest.insert(1, "distance", distances[0].round(2))
        scams = int(nearest["scam"].sum())

        if scams >= 2:
            st.error(f"**Likely a SCAM**: {scams} of the 3 most similar reviewed postings were scams.")
        else:
            st.success(f"**Likely legitimate**: {3 - scams} of the 3 most similar reviewed postings were legitimate.")
        st.caption("This is advice only. A moderator makes the final decision.")

        st.write("The 3 most similar reviewed postings:")
        st.dataframe(nearest[["posting_id", "distance", "scam", "salary_min", "salary_max",
                              "required_experience_years", "skills_listed", "vague_word_count",
                              "allcaps_ratio", "posting_hour", "category"]], hide_index=True)

# ------------------------------------------------------------------ backlog insights
QUERIES = {
    "Scam rate by time of day": """
        SELECT CASE WHEN posting_hour < 6  THEN '1 night (00-05)'
                    WHEN posting_hour < 12 THEN '2 morning (06-11)'
                    WHEN posting_hour < 18 THEN '3 afternoon (12-17)'
                    ELSE '4 evening (18-23)' END AS time_of_day,
               COUNT(*) AS postings, SUM(scam) AS scams, AVG(scam) AS scam_rate
        FROM posting
        GROUP BY time_of_day""",
    "Scam rate by pay range width (max / min)": """
        SELECT CASE WHEN salary_min IS NULL THEN '0 no pay range'
                    WHEN salary_max * 1.0 / salary_min <= 1.2 THEN '1 up to 1.2x'
                    WHEN salary_max * 1.0 / salary_min <= 1.4 THEN '2 1.2x to 1.4x'
                    ELSE '3 more than 1.4x' END AS pay_range,
               COUNT(*) AS postings, SUM(scam) AS scams, AVG(scam) AS scam_rate
        FROM posting
        GROUP BY pay_range""",
    "Scam rate by experience required": """
        SELECT CASE WHEN required_experience_years < 2 THEN '1 under 2 years'
                    WHEN required_experience_years < 5 THEN '2 2 to 5 years'
                    ELSE '3 5 years or more' END AS experience,
               COUNT(*) AS postings, SUM(scam) AS scams, AVG(scam) AS scam_rate
        FROM posting
        GROUP BY experience""",
    "Scam rate by claimed company size": """
        SELECT CASE WHEN company_size < 10 THEN '1 under 10'
                    WHEN company_size < 50 THEN '2 10 to 49'
                    ELSE '3 50 or more' END AS claimed_size,
               COUNT(*) AS postings, SUM(scam) AS scams, AVG(scam) AS scam_rate
        FROM posting
        GROUP BY claimed_size""",
    "Scam rate per job category": """
        SELECT c.name AS category, COUNT(*) AS postings, SUM(p.scam) AS scams, AVG(p.scam) AS scam_rate
        FROM posting p
        JOIN category c ON c.category_id = p.category_id
        GROUP BY c.name
        ORDER BY scam_rate DESC""",
}

with tab_backlog:
    st.write("Every table on this page is a SQL query run with pandasql.")

    st.subheader("Average traits: legitimate vs scam")
    averages = sql("""
        SELECT scam, COUNT(*) AS postings,
               AVG(required_experience_years) AS experience_years,
               AVG(skills_listed) AS skills_listed,
               AVG(vague_word_count) AS vague_words,
               AVG(allcaps_ratio) AS allcaps_ratio,
               AVG(company_size) AS company_size,
               AVG(salary_max * 1.0 / salary_min) AS pay_range_width,
               SUM(salary_min IS NULL) AS without_pay_range
        FROM posting
        GROUP BY scam""")
    st.dataframe(averages.round(3), hide_index=True)

    for title, query in QUERIES.items():
        st.subheader(title)
        result = sql(query)
        left, right = st.columns(2)
        left.dataframe(result.round(2), hide_index=True)
        right.bar_chart(result, x=result.columns[0], y="scam_rate", horizontal=True, height=60 + 35 * len(result))
        with st.expander("SQL"):
            st.code(query, language="sql")

    st.header("Explore the backlog")
    c1, c2, c3 = st.columns(3)
    categories = c1.multiselect("Category", category["name"], default=list(category["name"]))
    outcome = c2.selectbox("Outcome", ["all", "scam", "legitimate"])
    pay = c3.selectbox("Pay range", ["all", "published", "not published"])
    hours = st.slider("Hour the posting went live", 0, 23, (0, 23), key="explore_hours")

    if not categories:
        st.info("Pick at least one category.")
    else:
        names = ", ".join(f"'{name}'" for name in categories)
        query = f"""
        SELECT p.posting_id, c.name AS category, p.salary_min, p.salary_max,
               p.required_experience_years, p.posting_hour, p.scam
        FROM posting p
        JOIN category c ON c.category_id = p.category_id
        WHERE c.name IN ({names})
          AND p.posting_hour BETWEEN {hours[0]} AND {hours[1]}"""
        if outcome == "scam":
            query += "\n          AND p.scam = 1"
        if outcome == "legitimate":
            query += "\n          AND p.scam = 0"
        if pay == "published":
            query += "\n          AND p.salary_min IS NOT NULL"
        if pay == "not published":
            query += "\n          AND p.salary_min IS NULL"

        found = sql(query)
        c1, c2 = st.columns(2)
        c1.metric("Matching postings", len(found))
        c2.metric("Of which scams", int(found["scam"].sum()))
        found["outcome"] = found["scam"].map({0: "legitimate", 1: "scam"})
        st.bar_chart(found.groupby(["posting_hour", "outcome"]).size().unstack(fill_value=0))
        st.dataframe(found, hide_index=True)
        with st.expander("SQL"):
            st.code(query, language="sql")

    with st.expander("Write your own query (tables: posting, category)"):
        own = st.text_area("SQL", "SELECT category_id, COUNT(*) AS postings FROM posting GROUP BY category_id")
        if st.button("Run"):
            try:
                st.dataframe(sql(own), hide_index=True)
            except Exception as error:
                st.error(error)

# ------------------------------------------------------------------ model performance
with tab_model:
    test, baseline = metrics["test"], metrics["baseline"]
    st.write(f"3 nearest neighbours, Euclidean distance on 8 standardized features. Chosen with "
             f"5-fold cross-validation on {metrics['n_train']} postings, tested once on "
             f"{metrics['n_test']} postings the model never saw.")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{test['accuracy']:.1%}")
    c2.metric("Baseline accuracy (always legitimate)", f"{baseline['accuracy']:.1%}")

    st.subheader("Confusion matrix (test set)")
    st.dataframe(pd.DataFrame(
        [[test["tn"], test["fp"]], [test["fn"], test["tp"]]],
        index=["actually legitimate", "actually scam"],
        columns=["predicted legitimate", "predicted scam"]))

    def table(name):
        return pd.DataFrame(metrics[name]).T[["accuracy"]].astype(float).round(3)

    st.subheader("Feature sets (cross-validation, k = 3)")
    st.dataframe(table("feature_sets"))
    st.subheader("Scaling (cross-validation, final features)")
    st.dataframe(table("scaling_final"))
    st.subheader("Choice of k (cross-validation)")
    st.dataframe(table("k_values"))

    st.caption("A perfect score on 80 test postings does not mean the model is perfect: the test set "
               "is small, and scammers will change their postings once they know what we check.")

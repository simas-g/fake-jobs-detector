"""Get the two tables (posting, category) and run SQL on them with pandasql.

The tables come from PostgreSQL when it is running, otherwise straight from
the CSV (Streamlit Cloud has no database).
"""
import os

import pandas as pd
from pandasql import sqldf
from sqlalchemy import create_engine

CSV = "data/fake_jobs.csv"
DB_URL = os.environ.get(
    "DATABASE_URL", f"postgresql+psycopg2://{os.environ.get('USER')}@localhost:5433/fakejobs")

ALL_POSTINGS = """
SELECT p.*, c.name AS category
FROM posting p
JOIN category c ON c.category_id = p.category_id
ORDER BY p.posting_id
"""


def from_csv():
    posting = pd.read_csv(CSV)
    posting.insert(0, "posting_id", range(1, len(posting) + 1))

    names = sorted(posting["category"].unique())
    category = pd.DataFrame({"category_id": range(1, len(names) + 1), "name": names})

    posting["category_id"] = posting["category"].map(dict(zip(category["name"], category["category_id"])))
    posting = posting.drop(columns="category")
    return posting, category


def from_postgres():
    engine = create_engine(DB_URL, connect_args={"connect_timeout": 3})
    posting = pd.read_sql("SELECT * FROM posting ORDER BY posting_id", engine)
    category = pd.read_sql("SELECT * FROM category ORDER BY category_id", engine)
    return posting, category


def load_tables():
    try:
        posting, category = from_postgres()
        return posting, category, "PostgreSQL"
    except Exception:
        posting, category = from_csv()
        return posting, category, "CSV (no database connection)"


def run(query, posting, category):
    return sqldf(query, {"posting": posting, "category": category})

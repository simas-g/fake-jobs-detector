"""Database access: PostgreSQL by default, SQLite in memory as a demo fallback.

Connection settings come from the usual libpq environment variables:
PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD (defaults: localhost:5433/fakejobs).
"""
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "fake_jobs.csv"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
QUERIES_PATH = ROOT / "sql" / "queries.sql"

POSTING_COLUMNS = [
    "salary_min", "salary_max", "required_experience_years", "skills_listed",
    "vague_word_count", "exclamation_marks", "allcaps_ratio", "posting_hour",
    "employer_response_delay_h", "is_remote", "free_mail_contact",
    "company_age_months", "company_size", "mentions_trial_or_payment", "scam",
]


def pg_settings():
    return dict(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5433")),
        dbname=os.environ.get("PGDATABASE", "fakejobs"),
        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")),
        password=os.environ.get("PGPASSWORD") or None,
    )


def load_queries():
    """Parse sql/queries.sql into {name: {"title": ..., "sql": ...}}."""
    text = QUERIES_PATH.read_text()
    queries = {}
    for block in re.split(r"^-- name: ", text, flags=re.M)[1:]:
        name, rest = block.split("\n", 1)
        title = re.search(r"^-- title: (.*)$", rest, flags=re.M).group(1)
        sql = "\n".join(l for l in rest.splitlines() if not l.startswith("--")).strip()
        queries[name.strip()] = {"title": title.strip(), "sql": sql.rstrip(";")}
    return queries


def read_csv():
    df = pd.read_csv(CSV_PATH)
    df.insert(0, "posting_id", range(1, len(df) + 1))
    return df


def insert_rows(conn, df, placeholder):
    """Fill category and posting from the CSV dataframe."""
    categories = sorted(df["category"].unique())
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    cur = conn.cursor()
    cur.executemany(
        f"INSERT INTO category (category_id, name) VALUES ({placeholder}, {placeholder})",
        [(i, name) for name, i in cat_id.items()],
    )
    cols = ["posting_id"] + POSTING_COLUMNS + ["category_id"]
    rows = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        values = [r["posting_id"]]
        for c in POSTING_COLUMNS:
            v = r[c]
            if pd.isna(v):
                v = None
            elif c in ("salary_min", "salary_max", "skills_listed", "vague_word_count",
                       "exclamation_marks", "posting_hour", "is_remote", "free_mail_contact",
                       "company_age_months", "company_size", "mentions_trial_or_payment", "scam"):
                v = int(v)
            else:
                v = float(v)
            values.append(v)
        values.append(cat_id[r["category"]])
        rows.append(values)
    cur.executemany(
        f"INSERT INTO posting ({', '.join(cols)}) VALUES ({', '.join([placeholder] * len(cols))})",
        rows,
    )
    conn.commit()


def connect():
    """Return (connection, backend name). Falls back to SQLite if Postgres is down."""
    try:
        import psycopg2
        conn = psycopg2.connect(connect_timeout=3, **pg_settings())
        return conn, "PostgreSQL"
    except Exception:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.executescript(SCHEMA_PATH.read_text())
        insert_rows(conn, read_csv(), "?")
        return conn, "SQLite (fallback: PostgreSQL unreachable)"


def run_query(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    # Postgres returns Decimal for AVG; make every numeric column a float
    for c in df.columns:
        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() == df[c].notna().sum():
                df[c] = converted
    return df

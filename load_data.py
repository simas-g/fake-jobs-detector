"""Put data/fake_jobs.csv into PostgreSQL.

First create the database once:  createdb -p 5433 fakejobs
Then run:                         python load_data.py
"""
from sqlalchemy import create_engine

import db

posting, category = db.from_csv()
engine = create_engine(db.DB_URL)

with engine.begin() as conn:
    conn.exec_driver_sql(open("sql/schema.sql").read())
    category.to_sql("category", conn, if_exists="append", index=False)
    posting.to_sql("posting", conn, if_exists="append", index=False)  # empty salaries become NULL

print(f"Loaded {len(posting)} postings and {len(category)} categories.")

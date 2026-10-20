"""Create the fakejobs database (if needed), build the schema and import the CSV.

Usage:  python load_data.py
"""
import psycopg2
from psycopg2 import sql

import db


def main():
    settings = db.pg_settings()

    # CREATE DATABASE cannot run inside a transaction, so use the maintenance db
    admin = psycopg2.connect(**{**settings, "dbname": "postgres"})
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings["dbname"],))
        if cur.fetchone() is None:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings["dbname"])))
            print(f"created database {settings['dbname']}")
    admin.close()

    conn = psycopg2.connect(**settings)
    with conn.cursor() as cur:
        cur.execute(db.SCHEMA_PATH.read_text())
    conn.commit()

    df = db.read_csv()
    db.insert_rows(conn, df, "%s")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), SUM(scam), COUNT(*) - COUNT(salary_min) FROM posting")
        n, scams, no_salary = cur.fetchone()
    print(f"loaded {n} postings ({scams} scams, {no_salary} without a salary range)")
    conn.close()


if __name__ == "__main__":
    main()

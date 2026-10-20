-- Fake Jobs Detector: moderation backlog schema.
-- One row per reviewed job posting; the 8 job categories live in their own
-- table so a category name is stored once (no update anomalies).
-- Written in portable SQL so the same file also runs on SQLite (demo fallback).

DROP TABLE IF EXISTS posting;
DROP TABLE IF EXISTS category;

CREATE TABLE category (
    category_id INTEGER PRIMARY KEY,
    name        VARCHAR(20) NOT NULL UNIQUE
);

CREATE TABLE posting (
    posting_id                INTEGER PRIMARY KEY,
    -- pay range: both NULL when the employer published no range
    salary_min                INTEGER,
    salary_max                INTEGER,
    -- text of the posting
    required_experience_years DOUBLE PRECISION NOT NULL,
    skills_listed             INTEGER NOT NULL,
    vague_word_count          INTEGER NOT NULL,
    exclamation_marks         INTEGER NOT NULL,
    allcaps_ratio             DOUBLE PRECISION NOT NULL,
    posting_hour              INTEGER NOT NULL CHECK (posting_hour BETWEEN 0 AND 23),
    mentions_trial_or_payment SMALLINT NOT NULL CHECK (mentions_trial_or_payment IN (0, 1)),
    -- behaviour of the employer
    employer_response_delay_h DOUBLE PRECISION NOT NULL,
    is_remote                 SMALLINT NOT NULL CHECK (is_remote IN (0, 1)),
    free_mail_contact         SMALLINT NOT NULL CHECK (free_mail_contact IN (0, 1)),
    -- properties of the company
    company_age_months        INTEGER NOT NULL,
    company_size              INTEGER NOT NULL,
    category_id               INTEGER NOT NULL REFERENCES category (category_id),
    -- outcome of the trust and safety review: 1 = confirmed scam
    scam                      SMALLINT NOT NULL CHECK (scam IN (0, 1)),
    CHECK ((salary_min IS NULL) = (salary_max IS NULL)),
    CHECK (salary_min IS NULL OR salary_min <= salary_max)
);

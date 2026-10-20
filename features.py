"""Turn a raw posting (CSV/database row) into the numeric features KNN uses.

The same function is applied to the stored backlog and to a new posting, so a
new posting is always cleaned exactly like the postings it is compared with.
"""
import numpy as np
import pandas as pd

CATEGORIES = ["admin", "finance", "healthcare", "hospitality", "it",
              "logistics", "marketing", "sales"]

# Final feature set, chosen in evaluate.py with cross-validation on the training set.
FEATURES = [
    "salary_width",              # salary_max / salary_min, 0 when no range was published
    "no_salary",                 # 1 when the employer published no pay range
    "required_experience_years",
    "skills_listed",
    "allcaps_ratio",
    "hour_sin", "hour_cos",      # posting hour on a circle: 23:00 is next to 00:00
    "vague_word_count",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with every candidate feature (raw + engineered)."""
    out = pd.DataFrame(index=df.index)
    no_salary = df["salary_min"].isna() | df["salary_max"].isna()
    out["no_salary"] = no_salary.astype(int)
    out["salary_width"] = (df["salary_max"] / df["salary_min"]).where(~no_salary, 0.0)
    # median fill keeps raw salaries usable for the comparison experiments
    out["salary_min"] = df["salary_min"].fillna(df["salary_min"].median() if df["salary_min"].notna().any() else 0)
    out["salary_max"] = df["salary_max"].fillna(df["salary_max"].median() if df["salary_max"].notna().any() else 0)
    for c in ["required_experience_years", "skills_listed", "vague_word_count",
              "exclamation_marks", "allcaps_ratio", "posting_hour",
              "employer_response_delay_h", "is_remote", "free_mail_contact",
              "company_age_months", "company_size", "mentions_trial_or_payment"]:
        out[c] = df[c].astype(float)
    angle = 2 * np.pi * df["posting_hour"] / 24
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    for cat in CATEGORIES:
        out[f"category_{cat}"] = (df["category"] == cat).astype(int)
    return out


def prepare(df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    """Raw postings -> unscaled feature matrix (scaling is learned by the model)."""
    return build_features(df)[features].to_numpy(dtype=float)

"""Turn postings into the 8 numbers the model uses.

The same function is used for the 400 reviewed postings and for a new posting,
so both are cleaned in exactly the same way.
"""
import numpy as np
import pandas as pd

FEATURES = ["salary_width", "no_salary", "required_experience_years", "skills_listed",
            "allcaps_ratio", "hour_sin", "hour_cos", "vague_word_count"]


def prepare(df):
    out = pd.DataFrame(index=df.index)

    # no pay range: flag it and use 0 as the width
    out["salary_width"] = (df["salary_max"] / df["salary_min"]).fillna(0)
    out["no_salary"] = df["salary_min"].isna().astype(int)

    out["required_experience_years"] = df["required_experience_years"]
    out["skills_listed"] = df["skills_listed"]
    out["allcaps_ratio"] = df["allcaps_ratio"]

    # hour on a circle, so 23:00 and 00:00 are close to each other
    angle = 2 * np.pi * df["posting_hour"] / 24
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)

    out["vague_word_count"] = df["vague_word_count"]
    return out[FEATURES].to_numpy(dtype=float)

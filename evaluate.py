"""Choose the model with cross-validation, test it once, then train and save it.

Run:  python evaluate.py
Writes results/metrics.json (shown in the app) and model/knn.pkl (used by the app).
"""
import json
import os
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import db
from features import prepare
from knn import KNN


def scores(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def cross_validate(X, y, k=3, scaling="standard"):
    """5-fold CV; returns the scores over all out-of-fold predictions."""
    pred = np.zeros_like(y)
    for train, val in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        pred[val] = KNN(k, scaling).fit(X[train], y[train]).predict(X[val])
    return scores(y, pred)


# ---- data
posting, category, source = db.load_tables()
df = db.run(db.ALL_POSTINGS, posting, category)
y = df["scam"].to_numpy()
print(f"{len(df)} postings from {source}")

# ---- feature sets to compare
raw = df[["salary_min", "salary_max"]].fillna(df[["salary_min", "salary_max"]].median())
raw["no_salary"] = df["salary_min"].isna().astype(int)
for col in ["required_experience_years", "skills_listed", "vague_word_count", "exclamation_marks",
            "allcaps_ratio", "posting_hour", "employer_response_delay_h", "is_remote",
            "free_mail_contact", "company_age_months", "company_size", "mentions_trial_or_payment"]:
    raw[col] = df[col]
raw_without_category = raw.to_numpy(dtype=float)
for name in category["name"]:
    raw["category_" + name] = (df["category"] == name).astype(int)
raw_all = raw.to_numpy(dtype=float)

final = prepare(df)
width = final[:, [0]]
final_plain_hour = np.column_stack([final[:, [0, 1, 2, 3, 4, 7]], df["posting_hour"]])

feature_sets = {
    "all raw columns": raw_all,
    "raw, without category": raw_without_category,
    "raw + salary width": np.column_stack([raw_all, width]),
    "final (8 features)": final,
    "final, hour as plain number": final_plain_hour,
}

# ---- hold out 20% as a test set; every choice below uses only the other 80%
train, test = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)

results = {"source": source, "n_train": len(train), "n_test": len(test)}
results["feature_sets"] = {name: cross_validate(X[train], y[train]) for name, X in feature_sets.items()}
results["scaling_raw"] = {str(s): cross_validate(raw_all[train], y[train], scaling=s)
                          for s in [None, "minmax", "standard"]}
results["scaling_final"] = {str(s): cross_validate(final[train], y[train], scaling=s)
                            for s in [None, "minmax", "standard"]}
results["k_values"] = {str(k): cross_validate(final[train], y[train], k=k) for k in [1, 3, 5, 7, 9, 15]}

# ---- final setup, tested once on the test set
model = KNN(k=3, scaling="standard").fit(final[train], y[train])
test_pred = model.predict(final[test])
results["test"] = scores(y[test], test_pred)
results["baseline"] = scores(y[test], np.zeros_like(y[test]))  # always "legitimate"

# check our KNN against scikit-learn's
scaler = StandardScaler().fit(final[train])
sk = KNeighborsClassifier(3).fit(scaler.transform(final[train]), y[train])
results["same_as_sklearn"] = bool((sk.predict(scaler.transform(final[test])) == test_pred).all())

for name, r in results["feature_sets"].items():
    print(f"CV  {name:28s} accuracy {r['accuracy']:.3f}  recall {r['recall']:.3f}")
t = results["test"]
print(f"TEST accuracy {t['accuracy']:.3f}  precision {t['precision']:.3f}  recall {t['recall']:.3f}")
print(f"Baseline accuracy {results['baseline']['accuracy']:.3f}, same as sklearn: {results['same_as_sklearn']}")

# ---- the model the app uses: same setup, trained on all 400 postings
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)
pickle.dump(KNN(k=3, scaling="standard").fit(final, y), open("model/knn.pkl", "wb"))
json.dump(results, open("results/metrics.json", "w"), indent=2)
print("Saved model/knn.pkl and results/metrics.json")

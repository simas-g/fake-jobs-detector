"""Evaluate the KNN model and justify every modelling choice.

1. Hold out 20% of the postings as a test set (stratified, fixed seed).
2. On the remaining 80%, compare scaling methods, feature sets and k with
   5-fold cross-validation.
3. Train the chosen setup on the full 80% and report it once on the test set.

Writes results/metrics.json, which the app's "Model performance" tab reads.
Usage:  python evaluate.py
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import db
from features import CATEGORIES, FEATURES, build_features
from knn import KNN

SEED = 42

RAW = ["salary_min", "salary_max", "no_salary", "required_experience_years",
       "skills_listed", "vague_word_count", "exclamation_marks", "allcaps_ratio",
       "posting_hour", "employer_response_delay_h", "is_remote", "free_mail_contact",
       "company_age_months", "company_size", "mentions_trial_or_payment"] + \
      [f"category_{c}" for c in CATEGORIES]

FEATURE_SETS = {
    "all raw columns": RAW,
    "raw, without category": [f for f in RAW if not f.startswith("category_")],
    "raw + salary width": RAW + ["salary_width"],
    "final (8 features)": FEATURES,
    "final, hour as plain number": [f for f in FEATURES if not f.startswith("hour_")] + ["posting_hour"],
}


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def cross_validate(X, y, k=3, scaling="standard"):
    """Pooled out-of-fold predictions; the scaler is refit inside every fold."""
    pred = np.zeros_like(y)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        pred[va] = KNN(k, scaling).fit(X[tr], y[tr]).predict(X[va])
    return metrics(y, pred)


def main():
    conn, backend = db.connect()
    df = db.run_query(conn, db.load_queries()["all_postings"]["sql"])
    feats = build_features(df)
    y = df["scam"].to_numpy(dtype=int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"data from {backend}: {len(df)} postings, train {len(train_idx)}, test {len(test_idx)}")

    def X_of(cols, idx):
        return feats[cols].to_numpy(dtype=float)[idx]

    results = {"backend": backend, "n_train": len(train_idx), "n_test": len(test_idx)}

    print("\n5-fold CV on the training set, k=3")
    results["scaling"] = {}
    for scaling in [None, "minmax", "standard"]:
        m = cross_validate(X_of(RAW, train_idx), y_tr, 3, scaling)
        results["scaling"][str(scaling or "none")] = m
        print(f"  scaling={str(scaling):9s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f}")

    results["feature_sets"] = {}
    for name, cols in FEATURE_SETS.items():
        m = cross_validate(X_of(cols, train_idx), y_tr)
        results["feature_sets"][name] = m
        print(f"  {name:28s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f}")

    results["k_values"] = {}
    for k in [1, 3, 5, 7, 9, 15]:
        m = cross_validate(X_of(FEATURES, train_idx), y_tr, k)
        results["k_values"][str(k)] = m
        print(f"  k={k:2d} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f}")

    # final model, evaluated once on the untouched test set
    model = KNN(3, "standard").fit(X_of(FEATURES, train_idx), y_tr)
    test_pred = model.predict(X_of(FEATURES, test_idx))
    test = metrics(y_te, test_pred)
    baseline = metrics(y_te, np.zeros_like(y_te))
    results["test"] = test
    results["baseline_always_legit"] = baseline

    # sanity check: scikit-learn's KNN must give the same predictions
    sk = KNeighborsClassifier(3).fit(StandardScaler().fit_transform(X_of(FEATURES, train_idx)), y_tr)
    sk_pred = sk.predict(StandardScaler().fit(X_of(FEATURES, train_idx)).transform(X_of(FEATURES, test_idx)))
    results["matches_sklearn"] = bool((sk_pred == test_pred).all())

    print(f"\nTEST SET (n={len(test_idx)}), k=3, final features")
    print(f"  accuracy={test['accuracy']:.3f} precision={test['precision']:.3f} "
          f"recall={test['recall']:.3f} f1={test['f1']:.3f}")
    print(f"  confusion: TN={test['tn']} FP={test['fp']} FN={test['fn']} TP={test['tp']}")
    print(f"  baseline always-legit accuracy={baseline['accuracy']:.3f}")
    print(f"  identical to scikit-learn: {results['matches_sklearn']}")

    missed = df.iloc[test_idx][test_pred != y_te]
    results["test_errors"] = missed.to_dict(orient="records")
    if len(missed):
        print("\nmisclassified test postings:")
        print(missed.to_string())

    db.ROOT.joinpath("results").mkdir(exist_ok=True)
    (db.ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nwrote results/metrics.json")


if __name__ == "__main__":
    main()

"""
Train-test contamination experiment on a synthetic dataset.

Setup:
  * Features and target are random and binary, and the target is independent of
    the features. A model therefore cannot learn anything real, and the honest
    test accuracy is 50% (AUC 0.5).
  * The training set is "contaminated" by replacing a share of its rows with
    copies of test rows (features and labels). Contamination levels run from 0%
    (clean) to 100% (the training set is the test set).
  * Any accuracy above 50% on the test set can only come from the leaked rows.
  * Models range from limited capacity (logistic regression, shallow tree,
    AdaBoost on stumps) to high capacity (deep tree, random forest, neural network).

Outputs: results_raw.csv, results_summary.csv, accuracy_vs_contamination.png
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N_TRAIN, N_TEST, N_FEATURES = 1000, 1000, 20
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
SEEDS = range(20)


def models(seed):
    return {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Shallow tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "AdaBoost (stumps)": AdaBoostClassifier(n_estimators=100, random_state=seed),
        "Deep tree (unlimited)": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "Neural network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed),
    }


rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    X_train = rng.integers(0, 2, size=(N_TRAIN, N_FEATURES))
    y_train = rng.integers(0, 2, size=N_TRAIN)
    X_test = rng.integers(0, 2, size=(N_TEST, N_FEATURES))
    y_test = rng.integers(0, 2, size=N_TEST)

    for p in LEVELS:
        k = int(round(p * N_TEST))
        leaked = rng.choice(N_TEST, size=k, replace=False)         # test rows that leak
        replaced = rng.choice(N_TRAIN, size=k, replace=False)      # train rows they replace
        Xc, yc = X_train.copy(), y_train.copy()
        Xc[replaced], yc[replaced] = X_test[leaked], y_test[leaked]
        is_leaked = np.zeros(N_TEST, dtype=bool)
        is_leaked[leaked] = True

        for name, m in models(seed).items():
            m.fit(Xc, yc)
            pred = m.predict(X_test)
            proba = m.predict_proba(X_test)[:, 1]
            rows.append({
                "seed": seed, "contamination": p, "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "auc": roc_auc_score(y_test, proba),
                "acc_leaked_rows": accuracy_score(y_test[is_leaked], pred[is_leaked]) if k else np.nan,
                "acc_clean_rows": accuracy_score(y_test[~is_leaked], pred[~is_leaked]) if k < N_TEST else np.nan,
            })
    print(f"seed {seed} done", flush=True)

raw = pd.DataFrame(rows)
raw.to_csv(f"{OUT}/results_raw.csv", index=False)
summary = (raw.groupby(["model", "contamination"])
           .agg(accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                acc_leaked_rows=("acc_leaked_rows", "mean"), acc_clean_rows=("acc_clean_rows", "mean"))
           .reset_index())
summary.to_csv(f"{OUT}/results_summary.csv", index=False)
print(summary.round(3).to_string())

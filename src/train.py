import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def load_data(path, sample=None):
    """Load CSV and optionally sample rows for faster training."""
    data = pd.read_csv(path)
    if sample and sample < len(data):
        data = data.sample(n=sample, random_state=42)
        print(f"[INFO] Using a sample of {sample} rows instead of full dataset.")
    else:
        print(f"[INFO] Using full dataset with {len(data)} rows.")
    return data

def evaluate(y_true, y_pred, y_score):
    return {
        'roc_auc': float(roc_auc_score(y_true, y_score)),
        'pr_auc': float(average_precision_score(y_true, y_score)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

def plot_curves(y_true, scores, labels, outdir):
    # ROC
    plt.figure()
    for s, l in zip(scores, labels):
        fpr, tpr, _ = roc_curve(y_true, s)
        plt.plot(fpr, tpr, label=l)
    plt.plot([0,1],[0,1],'k--')
    plt.legend(); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.savefig(os.path.join(outdir,'roc.png'))
    plt.close()
    # PR
    plt.figure()
    for s, l in zip(scores, labels):
        prec, rec, _ = precision_recall_curve(y_true, s)
        plt.plot(rec, prec, label=l)
    plt.legend(); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.savefig(os.path.join(outdir,'pr.png'))
    plt.close()

def main(args):
    print("[INFO] Loading data...")
    data = load_data(args.data, args.sample)
    X = data.drop('Class', axis=1)
    y = data['Class']

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("[INFO] Handling imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(),
        'svm': SVC(probability=True, random_state=42)
    }

    results, scores, labels = {}, [], []
    best_preds, best_scores = None, None

    for name, model in models.items():
        print(f"[INFO] Training model: {name}...")
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:,1]
        # ⚡ use threshold 0.3 instead of 0.5
        y_pred = (y_score >= 0.3).astype(int)
        metrics = evaluate(y_test, y_pred, y_score)
        results[name] = metrics
        scores.append(y_score); labels.append(name)
        if name == 'rf':
            best_preds, best_scores = y_pred, y_score

    # Ensemble: average RF + GB
    print("[INFO] Building ensemble (RF + GB)...")
    avg_score = (best_scores + scores[labels.index('gb')]) / 2
    y_pred = (avg_score >= 0.3).astype(int)
    metrics = evaluate(y_test, y_pred, avg_score)
    results['dboost'] = metrics

    # Save results
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out,'metrics.json'),'w') as f:
        json.dump(results,f,indent=2)

    with open(os.path.join(args.out,'classification_report.txt'),'w') as f:
        f.write(classification_report(y_test,y_pred,zero_division=0))

    cm = confusion_matrix(y_test,y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(args.out,'confusion_matrix.csv'))

    plot_curves(y_test, scores+[avg_score], labels+['dboost'], args.out)
    print("✅ Training finished. Results saved to", args.out)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/creditcard.csv')
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--sample', type=int, default=None, help="Use a sample of N rows for faster training")
    args = parser.parse_args()
    main(args)

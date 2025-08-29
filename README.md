# Credit Card Fraud Detection

A Machine Learning project to detect fraudulent credit card transactions using the popular [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
The dataset is highly imbalanced, containing **284,807 transactions** with only **492 fraud cases (0.172%)**.

## Project Structure
```bash
creditcard-fraud-project/
├── data/                # Dataset (not included in repo due to large size)
├── outputs/             # Results (metrics, plots, confusion matrix)
├── src/
│   ├── train.py         # Training script (with SMOTE + models + evaluation)
│   ├── view_results.py  # Script to view saved metrics and reports
├── .gitignore           # Ignore large dataset files
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## Features
- Handles **imbalanced dataset** using **SMOTE**.
- Implements multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
- Evaluates using:
  - ROC-AUC
  - PR-AUC
  - F1-Score
- Saves outputs:
  - `metrics.json` → Performance metrics
  - `classification_report.txt` → Detailed precision/recall/F1
  - `confusion_matrix.csv` → Confusion matrix values
  - `roc.png` and `pr.png` → Curves for model comparison

## Installation & Usage

### 1. Clone Repository
```bash
git clone https://github.com/prasannaedu/creditcard-fraud-detection.git
cd creditcard-fraud-detection
```

### 2. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Linux/Mac

pip install -r requirements.txt
```

### 3. Download Dataset
- Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Place it in the `data/` directory.

### 4. Train Models
```bash
python src/train.py --data data/creditcard.csv --out outputs
```

(Optional: use a sample of 10,000 rows to train faster)
```bash
python src/train.py --data data/creditcard.csv --out outputs --sample 10000
```

### 5. View Results
```bash
python src/view_results.py
```

## Results (Sample Run on 10,000 Rows)
| Model               | ROC-AUC | PR-AUC | F1-Score |
|---------------------|---------|--------|----------|
| Logistic Regression | 0.7168  | 0.2527 | 0.0784   |
| Random Forest       | 0.9995  | 0.8509 | 0.6667   |
| Gradient Boosting   | 0.9991  | 0.7688 | 0.4348   |
| SVM                 | 0.9918  | 0.3722 | 0.3333   |
| **Ensemble (RF+GB)**| **0.9998** | **0.8767** | **0.5000** |

## Notes
- The dataset is **too large** for GitHub (>100MB). It is excluded using `.gitignore`.  
- Download it separately from [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## References
- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)

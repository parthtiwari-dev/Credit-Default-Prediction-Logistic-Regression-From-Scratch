# Credit Default Prediction — Logistic Regression From Scratch

Project goal:
Build a full classification pipeline that predicts whether a customer will default next month using logistic regression implemented from scratch and compare with sklearn.

Dataset:
UCI "Default of Credit Card Clients Dataset" (Kaggle mirror name: default of credit card clients dataset). Target: `default.payment.next.month`.

Folder structure:
credit-default-logistic/
├── data/
│ ├── raw.csv
│ └── processed.csv
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling_logistic.ipynb
├── src/
│ ├── scratch_logistic.py
│ ├── preprocess.py
│ └── evaluation.py
├── outputs/
│ ├── figures/
│ └── results/
├── README.md
└── requirements.txt


Pipeline steps:
1. Load raw.csv, sanity checks, inspect imbalance.
2. EDA: distribution plots, correlations, selected pairplot.
3. Feature engineering: encoding, scaling, train-test split (stratify).
4. Imbalance handling: baseline, class weights, oversampling.
5. Train vectorized logistic from scratch (gradient descent). Save cost curve.
6. Train sklearn LogisticRegression (solver='liblinear', class_weight='balanced') as benchmark.
7. Manual evaluation: implement confusion matrix, precision, recall, F1, ROC, PR curves, threshold tuning.
8. Interpret coefficients: odds ratios, feature importance.
9. Produce final experiments table and visuals for LinkedIn + resume.

Deliverables:
- `src` with full implementations
- Notebooks documenting EDA, FE, modeling and visuals exported to outputs/figures
- `outputs/results/experiments.csv` with experiment summary

LinkedIn Hook:
“Everyone starts with Titanic. I built logistic regression from scratch to predict credit card defaults.”

Next steps to run:
1. `pip install -r requirements.txt`
2. Place dataset as `data/raw.csv`
3. Run `python src/preprocess.py` to generate `data/processed.csv`
4. Open notebooks to run EDA and modeling

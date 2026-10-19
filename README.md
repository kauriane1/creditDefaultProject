# Credit Card Payment Default Prediction: Traditional ML vs Deep Learning

Summative assignment (Machine Learning): binary classification of next-month credit-card payment default using Scikit-learn and TensorFlow/Keras (`tf.data`).

## Problem
Predict whether a credit-card customer will default on the following month's payment (`Y` = 1) from demographic, credit-limit, repayment-status, bill-amount and payment-amount variables. The task is imbalanced (22.12% default), so the project evaluates macro-F1, per-class precision/recall, confusion matrices and ROC-AUC rather than accuracy alone.

## Dataset
**Default of Credit Card Clients** (Yeh & Lien, 2009), UCI Machine Learning Repository, dataset id 350: <https://archive.ics.uci.edu/dataset/350/default>

- 30,000 rows, 23 predictors (`X1`-`X23`), binary target `Y`; no missing values
- 35 exact duplicate rows and 21 feature groups with conflicting labels (documented; no rows removed)
- Stratified split 70/10/20: 21,000 train / 3,000 validation / 6,000 test; `StandardScaler` fitted on the training set only
- **The raw data is not committed to this repository.** The notebook downloads it with `ucimlrepo.fetch_ucirepo(id=350)`; a zipped copy is also in the Google Drive folder.

## Experiments (validation set; seed 42)

| Exp | Approach | Model | Key settings | Val. accuracy | Val. macro-F1 |
|---|---|---|---|---|---|
| 1 | Traditional | Logistic Regression | max_iter=1000 | 0.8130 | 0.6402 |
| 2 | Traditional | Random Forest (default) | 100 trees | 0.8217 | **0.6914** |
| 3 | Traditional | Random Forest (tuned) | 200 trees, max_depth=15, min_samples_split=10 | 0.8217 | 0.6833 |
| 4 | Traditional | Gradient Boosting | 100 estimators, lr=0.1 | 0.8217 | 0.6848 |
| 5 | Deep learning | Sequential DNN (2 layers) | 64-32, Dropout 0.3, Adam, 15 epochs | 0.8207 | 0.6828 |
| 6 | Deep learning | Sequential DNN (3 layers) | 128-64-32, Dropout 0.4/0.3, Adam lr=0.001, 20 epochs | 0.8197 | 0.6884 |
| 7 | Deep learning | Sequential DNN + BatchNorm | 128-64, BatchNorm, Dropout 0.3, Adam lr=0.0005, 20 epochs | 0.8190 | 0.6854 |

Experiment 6 is selected automatically as the best network (highest validation macro-F1 among the networks).

### Test-set results of the selected network (Exp. 6, threshold 0.5)

| Accuracy | Precision (default) | Recall (default) | F1 (default) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.8170 | 0.6395 | 0.3956 | 0.4888 | 0.6887 | 0.773 |

Confusion matrix `[[4377, 296], [802, 525]]`. Test ROC-AUC of the other models: Logistic Regression 0.708, Random Forest 0.756, Gradient Boosting 0.779.

**Main finding:** tree-based models and the networks perform within ~0.01 macro-F1 of each other (single runs, no significance testing); all clearly beat Logistic Regression. The selected network recovers only ~40% of actual defaulters at the 0.5 threshold, so threshold tuning, class weighting and fairness checks would be needed before any real use.

## Repository structure
```
.
├── notebooks/CreditCard_PaymentDefault_Prediction.ipynb   # full pipeline with outputs
├── report/Credit_Card_Default_Prediction_Report.pdf       # IEEE-style report
├── figures/                                               # figures used in the report
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducing the results
**Google Colab (recommended, how the results were produced):**
1. Open the notebook with the Colab badge above.
2. Run all cells top to bottom. The first cells install `ucimlrepo` and mount Google Drive; models, scaler and TensorBoard logs are written to `MyDrive/CreditDefaultProject/`.

**Local machine:**
```bash
git clone https://github.com/<YOUR-USERNAME>/credit-card-default-prediction.git
cd credit-card-default-prediction
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/CreditCard_PaymentDefault_Prediction.ipynb
```
The notebook mounts Google Drive through `google.colab`; when running locally, skip or edit that cell and change `PROJECT_DIR` / `MODEL_DIR` to a local folder.

Seeds are fixed (`SEED = 42` for Python, NumPy, Keras and all splits/models). Exact numbers can still differ slightly across hardware and library versions.

## Saved artifacts (Google Drive, not in Git)
- `feature_scaler.joblib`: fitted `StandardScaler`
- `best_traditional_model.joblib`: default Random Forest
- `best_deep_learning_model.keras`: selected network (Experiment 6)
- `tensorboard_logs/`: per-experiment TensorBoard logs (`tensorboard --logdir tensorboard_logs`)

## Known limitations
- Single run per configuration; no confidence intervals
- Fixed 0.5 threshold; no class weighting or resampling tried
- Duplicates retained; cross-partition duplicate leakage not checked
- ROC cell in the notebook labels the Exp. 6 curve "DNN + BatchNorm" (and omits Exp. 7)

## References
Yeh, I.-C. and Lien, C.-H., "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients," *Expert Systems with Applications*, vol. 36, no. 2, pp. 2473-2480, 2009. Full reference list is in the report.

## License / academic note
Coursework submission. Dataset © original authors / UCI Machine Learning Repository.
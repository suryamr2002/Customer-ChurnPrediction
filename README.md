# 📊 Customer Churn Prediction with XGBoost

Predict which customers are likely to churn using historical, consumption, and pricing data — built with XGBoost and explained through feature importance.


## ✅ Project Overview

This study focuses on an electricity utility company with high churn among SME customers, driven primarily by **price sensitivity**. The goal is to:

1. Combine customer, churn, and pricing data
2. Engineer features related to consumption, pricing trends, and account metadata
3. Handle imbalanced data and outliers
4. Build and tune an XGBoost model for binary churn prediction
5. Evaluate performance and interpret results
6. Recommend targeted retention strategies based on insights

---

## 🔧 Data

Data files in the repo:

* **customer\_data.csv** – customer attributes and activation dates
* **churn\_data.csv** – churn flag (`0`/`1`)
* **historical\_price\_data.csv** – historical electricity and gas pricing

These are merged and processed in the notebook.

---

## 📚 Requirements

Use the [`requirements.txt`](requirements.txt) provided


Key libraries used:

* Python 3.x
* `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `seaborn`, `matplotlib`

---

## 🚀 Reproduce Project

Clone the repository, then:

```bash
pip install -r requirements.txt
jupyter notebook "Churn_prediction_andAnalysis_github.ipynb"
```

The notebook walks through:

1. **Data Loading & Cleaning**
2. **Feature Engineering**

   * Tenure calculation
   * Consumption metrics (`cons_12m`, `cons_last_month`, etc.)
   * Price-based features over time
3. **Handling Skewness & Outliers**
4. **Dummy encoding**, **feature aggregation**
5. **Train-test split**, sampling
6. **Model tuning** via `GridSearchCV`, focusing on

   * `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `scale_pos_weight`
     
        | Parameter          | Meaning                                                               |
        | ------------------ | --------------------------------------------------------------------- |
        | `subsample`        | Fraction of rows used per tree (0.7 = 70%)                            |
        | `scale_pos_weight` | Used for imbalanced classes (1 = no weighting)                        |
        | `n_estimators`     | Total number of trees (boosting rounds)                               |
        | `min_child_weight` | Minimum sum of weights needed in a child node                         |
        | `max_depth`        | Maximum tree depth (controls model complexity)                        |
        | `learning_rate`    | How much the model adjusts per tree (lower = slower but more precise) |
        | `gamma`            | Minimum loss reduction to make a split (regularization)               |
        | `colsample_bytree` | Fraction of features used per tree                                    |

7. **Model evaluation**

   * Metrics: accuracy, precision, recall, F1, ROC-AUC
   * ROC curve plotting
8. **Feature interpretation** using XGBoost’s feature importances

---

## 📈 Results

* Final XGBoost model achieved **AUC close to 1.0** (on `unseen data `)
* Most important churn predictors:

  * **Price sensitivity** (`price_p*_var`)
  * **11-month/yearly consumption metrics**
  * **Net margin, tenure**, etc.
* Recommendation: offer **targeted incentives** to high-value customers predicted to churn

---

## 💡 Business Insight

* **Churn driven by pricing** and consumption patterns
* Monetary benefits should focus on **high-consumption, high-marginal-value customers**

---

## ✅ File Structure

```
├── Dataset
      ├──customer_data.csv
      ├── churn_data.csv
      ├── historical_price_data.csv
├── Churn_prediction_andAnalysis_github.ipynb
├── requirements.txt
└── README.md
```

---

## 📌 Tips

* Ensure aligned index structure before merging data
* Apply skew transforms (`log1p`, Yeo-Johnson) on consumption/price features
* Avoid multicollinearity by dropping one dummy column
* Tune XGBoost with `GridSearchCV` using updated packages
* Validate model on unseen or cross-validated data to check for overfitting

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and open a PR. Suggestions:

* Use **SHAP or LIME** for deeper interpretability
* Compare to **logistic regression** or other **tree-based models**
* Add **time-series cross-validation or custom validation strategies**


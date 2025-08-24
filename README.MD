# Employee Attrition Analysis & Prediction

## 📌 Overview
Employee attrition is a critical issue faced by organizations. High attrition rates increase hiring costs and impact productivity. This project analyzes an HR dataset to uncover patterns in employee attrition and builds predictive models to estimate the likelihood of employees leaving the company.

---

## 🎯 Project Goals
- Perform **data cleaning and preprocessing**.
- Conduct **Exploratory Data Analysis (EDA)** to identify attrition trends.
- Build and evaluate predictive models:
  - Logistic Regression
  - Decision Tree Classifier
- Explain model predictions using **SHAP (SHapley Additive Explanations)** for interpretability.

---

## 📂 Dataset Information
- **File Name**: `Employee Attrition_dataset.csv`
- **Rows**: ~1470 employees
- **Columns**:
  - **Demographic Features**: `Age`, `Gender`, `MaritalStatus`
  - **Job Features**: `Department`, `JobRole`, `JobLevel`
  - **Compensation**: `MonthlyIncome`, `SalaryBand` (engineered feature)
  - **Performance & Tenure**: `YearsAtCompany`, `YearsSinceLastPromotion`, `TrainingTimesLastYear`
  - **Target**: `Attrition` (`Yes`/`No`)

---

## 📁 Project Structure
FINAL_PROJECT/
│
├── Attrition_analysis.ipynb 
├── Employee Attrition_dataset.csv 
├── HR_Employee Attrition_dashboard.pbix
├── README.md 
├── logistic_regression_shap
└── decision_tree_shap.png

---

## 🔍 Exploratory Data Analysis Highlights
- **Attrition by Department**:
  - Highest attrition observed in certain departments.
- **Attrition by Salary Band**:
  - Lower salary bands have significantly higher attrition.
- **Promotion Impact**:
  - Employees with long gaps since last promotion tend to leave more.

**Key Visualizations**:
- Bar charts for Department-wise and Salary Band attrition.
- Distribution plots for tenure and age.
- Correlation heatmap for numeric features.

---

## 🛠 Tools & Libraries
- **Python** (3.x)
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Model Explainability**: `shap`

---

## ⚙️ Data Preprocessing
- Handled missing values and duplicates.
- Label encoding for categorical variables.
- Feature scaling (Standardization) for logistic regression.
- Created new feature: `SalaryBand` using quantile-based binning.

---

## 🤖 Models Implemented
### 1. **Logistic Regression**
    Scaled features using StandardScaler.
    Evaluation metrics:
    Accuracy: 87.75%
    Precision: 0.90 (Class 0), 0.69 (Class 1)
    Recall: 0.96 (Class 0), 0.43 (Class 1)
    F1-score: 0.93 (Class 0), 0.53 (Class 1)

### 2. **Decision Tree Classifier**
    Used Gini Index as the splitting criterion.
    Evaluation metrics:
    Accuracy: 84.35%
    Precision: 0.86 (Class 0), 0.53 (Class 1)
    Recall: 0.97 (Class 0), 0.19 (Class 1)
    F1-score: 0.91 (Class 0), 0.28 (Class 1)
---

## ✅ Model Performance
| Model                | Accuracy | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | ~85%    | ~0.87   |
| Decision Tree        | ~83%    | ~0.80   |


---

## 🔍 SHAP Analysis (Explainability)
SHAP is used to interpret model predictions and understand feature importance:
- **Summary Plot**: Shows which features have the most impact on predictions across the dataset.
- **Force Plot**: Explains why a specific employee is predicted to leave or stay.
- **Feature Importance**: Highlights top contributors like:
  - `MonthlyIncome`
  - `YearsSinceLastPromotion`
  - `JobLevel`

**Code snippet to add SHAP analysis**:
```python
import shap

# Train your model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# SHAP values
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot for one observation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

## 🧾 Conclusion 
Logistic Regression outperforms the Decision Tree overall (higher accuracy and ROC-AUC) and offers more stable, interpretable coefficients.

Key drivers (from SHAP and EDA) typically include MonthlyIncome, YearsSinceLastPromotion, and JobLevel, along with tenure-related variables.

However, recall for the attrition class remains limited (LogReg: 0.43, Tree: 0.19). If the business objective is catching potential leavers, you should lower the decision threshold, rebalance classes, and prioritize Recall / PR-AUC over raw Accuracy. This will likely reduce precision but provides earlier intervention opportunities.

---
# Health Insurance Premium Prediction

## Problem Statement

### Business Context

In today's healthcare landscape, rising costs and increasing demand for affordable insurance put significant pressure on insurance providers to price premiums accurately. Setting premiums too high drives customers away, while setting them too low leads to financial losses. This project aims to build a machine learning model that predicts health insurance charges based on personal and demographic factors.

### Objective

Develop a predictive model that accurately estimates health insurance premium charges, helping insurance companies make data-driven pricing decisions.

---

## Dataset

The dataset contains **1,338 records** with the following features:

| Feature    | Description                                              |
|------------|----------------------------------------------------------|
| `age`      | Age of the insured individual                            |
| `sex`      | Gender (male / female)                                   |
| `bmi`      | Body Mass Index                                          |
| `children` | Number of dependents covered                             |
| `smoker`   | Smoking status (yes / no)                                |
| `region`   | Residential region (northeast, northwest, southeast, southwest) |
| `charges`  | Insurance premium charges (target variable)              |

---

## Project Workflow

### 1. Data Overview

- Checked dataset shape, data types, and statistical summary
- Identified and removed **1 duplicate record** (1,338 → 1,337 rows)
- Confirmed **no missing values** in any column

### 2. Exploratory Data Analysis (EDA)

**Univariate Analysis:**
- **Age** — Higher concentration of younger individuals (20-25 age bracket)
- **BMI** — Normally distributed, centered around 30
- **Gender** — Nearly balanced between male and female
- **Smoker** — Majority are non-smokers
- **Region** — Fairly balanced, southeast slightly higher
- **Charges** — Right-skewed distribution with outliers above 50,000

**Bivariate Analysis:**
- Insurance charges **increase with age** (positive correlation)
- **Smokers pay significantly higher premiums** than non-smokers — the strongest differentiator
- Gender has a minor influence; males have slightly higher and more variable charges
- Number of children shows limited impact beyond 2 dependents
- Regional differences are minimal
- Correlation heatmap confirmed **age (0.30)** and **smoker status** as the most influential features

### 3. Data Preprocessing

- **Dummy encoding** — Converted categorical columns (`sex`, `smoker`, `region`) into numerical values using one-hot encoding with `drop_first=True`
- **Train-test split** — 70% training, 30% testing (`random_state=42`)

### 4. Model Evaluation Criteria

In insurance premium prediction, the model can make two types of errors:

- **Underestimating the price** — Predicted premium is lower than actual cost, leading to financial losses for the insurer
- **Overestimating the price** — Predicted premium is higher than actual cost, driving customers away

To capture both types of errors, we used:
- **R-squared (R2)** — Measures how well the model explains the variance in charges
- **Mean Squared Error (MSE)** — Penalizes larger prediction errors more heavily

### 5. Model Building — Linear Regression

We built three linear regression models incrementally to understand feature importance:

| Model   | Features Used          | Train R2 | Test R2 | Observation                          |
|---------|------------------------|----------|---------|--------------------------------------|
| Model 1 | Age only               | Low      | Low     | Underfitting — single feature is insufficient |
| Model 2 | Age + BMI              | Low      | Low     | Slight improvement, still underfitting |
| Model 3 | All features           | ~0.74    | ~0.77   | Significant improvement, generalizes well but room to grow |

**Takeaway:** Using all features (including encoded categorical variables) significantly improved the model, but linear regression alone could not fully capture the non-linear patterns in the data.

### 6. Decision Tree Regressor

- Built a Decision Tree model to capture non-linear relationships
- **Problem:** The model **overfitted** — near-perfect training performance but poor test performance
- This indicated the need for an ensemble approach

### 7. Random Forest Regressor

- Applied Random Forest to reduce overfitting through bagging (training multiple trees on random subsets)
- Default Random Forest already showed improvement over a single decision tree

**Hyperparameter Tuning:**

Tuned the following parameters to optimize performance:

| Parameter          | Purpose                                          |
|--------------------|--------------------------------------------------|
| `n_estimators`     | Number of trees in the forest                    |
| `max_depth`        | Maximum depth of each tree                       |
| `min_samples_split`| Minimum samples required to split a node         |
| `min_samples_leaf` | Minimum samples required at a leaf node          |

**Tuning methods used:**
- **GridSearchCV** — Exhaustive search over a defined parameter grid
- **RandomizedSearchCV** — Efficient random sampling from the parameter space

### 8. XGBoost Regressor

- Applied XGBoost, a gradient boosting ensemble method, for further improvement
- XGBoost builds trees sequentially, where each new tree corrects the errors of the previous one

**Hyperparameter Tuning:**

| Parameter          | Purpose                                          |
|--------------------|--------------------------------------------------|
| `n_estimators`     | Number of boosting rounds                        |
| `learning_rate`    | Step size for each boosting round                |
| `max_depth`        | Maximum depth of each tree                       |
| `subsample`        | Fraction of samples used per tree                |

Used **GridSearchCV** and **RandomizedSearchCV** to find the best combination.

### 9. Feature Importance

- Extracted feature importances from both the best Random Forest and XGBoost models
- **Smoker status** emerged as the most important predictor of insurance charges, followed by **age** and **BMI**

---

## Model Progression Summary

```
Linear Regression (Age only)        → Underfitting
Linear Regression (Age + BMI)       → Still underfitting
Linear Regression (All features)    → Decent baseline (~0.75 R2)
Decision Tree                       → Overfitting
Random Forest (Default)             → Reduced overfitting
Random Forest (Tuned)               → Improved generalization
XGBoost (Default)                   → Strong performance
XGBoost (Tuned)                     → Best performance
```

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- **Environment:** Google Colab / Jupyter Notebook

---

## Files in This Repository

| File | Description |
|------|-------------|
| `Case_Study_Health_Insurance_Premium_Prediction (1).ipynb` | Detailed solution with explanations (up to Linear Regression) |
| `HealthInsurance_PremiumPrediction.ipynb` | Complete solution including Decision Tree, Random Forest, and XGBoost |
| `insurance_prediction.csv` | Dataset used for the project |

---

## Key Findings

1. **Smoking status** is the single most important factor affecting insurance premiums
2. **Age** and **BMI** are the next most significant predictors
3. Linear regression provides a reasonable baseline but cannot capture non-linear patterns
4. Decision trees overfit easily without constraints
5. Ensemble methods (Random Forest, XGBoost) with hyperparameter tuning deliver the best results
6. XGBoost with tuned hyperparameters achieved the best overall performance

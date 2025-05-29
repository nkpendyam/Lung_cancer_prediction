# Lung Cancer Prediction

This repository contains a machine learning project for predicting lung cancer risk using survey data. The implementation is in a Jupyter Notebook (`MAYA_final.ipynb`) and uses the dataset `survey lung cancer.csv` to train and evaluate multiple models, including a custom stacking classifier called "Maya Hybrid." The project employs advanced feature engineering, handles class imbalance with SMOTE, and evaluates models using accuracy, precision, recall, and F1-score. The best-performing model is a Random Forest with an accuracy of 0.9126.

## Motivation

Lung cancer is a leading cause of cancer-related deaths globally, with early detection being critical for improving patient outcomes. This project aims to develop a predictive model to identify high-risk individuals based on survey data, enabling early screening and intervention. By leveraging features like smoking habits, age, and health indicators, the model supports clinicians and researchers in addressing this public health challenge.

## Project Overview

The project analyzes a dataset of patient attributes to predict lung cancer risk. It includes:
- Data preprocessing with categorical encoding and SMOTE for class imbalance.
- Feature engineering using polynomial features.
- Feature selection with mutual information.
- Training and evaluation of multiple machine learning models.
- A custom stacking classifier (Maya Hybrid) combining SVC, Decision Tree, and KNN.
- Visualizations like histograms and confusion matrices.

## Dataset

- **File**: `data/survey lung cancer.csv`
- **Description**: Contains 309 records with 16 columns (15 features + target variable `LUNG_CANCER`).
- **Class Distribution**:
  - `LUNG_CANCER = 1 (YES)`: 270 cases (87.4%)
  - `LUNG_CANCER = 2 (NO)`: 39 cases (12.6%)

### Features

| Feature                  | Description                                    |
|--------------------------|------------------------------------------------|
| `GENDER`                 | Gender (M = Male, F = Female; mapped to M=1, F=2) |
| `AGE`                    | Age of the patient (numeric)                   |
| `SMOKING`                | Smoking status (1 = No, 2 = Yes)               |
| `YELLOW_FINGERS`         | Yellow fingers (1 = No, 2 = Yes)               |
| `ANXIETY`                | Anxiety level (1 = No, 2 = Yes)                |
| `PEER_PRESSURE`          | Peer pressure influence (1 = No, 2 = Yes)      |
| `CHRONIC DISEASE`        | Chronic disease (1 = No, 2 = Yes)              |
| `FATIGUE`                | Fatigue symptoms (1 = No, 2 = Yes)             |
| `ALLERGY`                | Allergy status (1 = No, 2 = Yes)               |
| `WHEEZING`               | Wheezing symptoms (1 = No, 2 = Yes)            |
| `ALCOHOL CONSUMING`      | Alcohol consumption (1 = No, 2 = Yes)          |
| `COUGHING`               | Coughing symptoms (1 = No, 2 = Yes)            |
| `SHORTNESS OF BREATH`    | Shortness of breath (1 = No, 2 = Yes)          |
| `SWALLOWING DIFFICULTY`  | Difficulty swallowing (1 = No, 2 = Yes)        |
| `CHEST PAIN`             | Chest pain symptoms (1 = No, 2 = Yes)          |
| `LUNG_CANCER`            | Target variable (YES = 1, NO = 2)              |

### Preprocessing

- **Categorical Encoding**: `GENDER` (M=1, F=2) and `LUNG_CANCER` (YES=1, NO=2) are mapped to numeric values.
- **Feature Engineering**: Polynomial features (degree=2) capture variable interactions.
- **Class Imbalance**: SMOTE balances the dataset.
- **Feature Selection**: Mutual information selects top features.

## Methodology

1. **Data Loading**: Load `survey lung cancer.csv` using pandas.
2. **Preprocessing**: Encode categorical variables, apply SMOTE, generate polynomial features.
3. **Feature Selection**: Use mutual information to select relevant features.
4. **Model Training**: Train Logistic Regression, KNN, Decision Tree, SVM, Naive Bayes, Random Forest, and Maya Hybrid.
5. **Hyperparameter Tuning**: Optimize models with RandomizedSearchCV.
6. **Evaluation**: Assess models using accuracy, precision, recall, F1-score, and confusion matrices.
7. **Visualization**: Generate histograms and confusion matrices.

## Requirements

Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-optimize imbalanced-learn

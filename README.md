# Neoadjuvant Therapy Classification Project

## Project Overview

This project focuses on predicting the response to neoadjuvant therapy in cancer patients using various machine learning models. The dataset includes clinical and histopathological features of patients. The project includes data processing, feature engineering, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Feature Selection](#feature-selection)
- [Machine Learning Models and evaluation ](#machine-learning-models-and-evaluation)



### Data Description

The dataset used for this project is `Neoadj_base_luminaux - anonymise.xls`. It contains clinical data and histopathological information for patients undergoing neoadjuvant therapy.

- **Columns:** The dataset has multiple columns including patient identifiers, clinical data, histopathological features, and outcome variables.

## Setup

### Requirements

To run this project, you'll need the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- tensorflow

You can install the required packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow
```
## File Structure
```
/path/to/project
│
├── Neoadj_base_luminaux - anonymise.xls # Raw dataset
├── perso_mec_imp.csv # tailored imputed dataset
├── data_preprocessing.py # Data processing and feature engineering script
├── model_building.py # Machine learning model training and evaluation script
├── neural_network.py # Neural network model implementation and training script
├── README.md # This README file
└── requirements.txt # List of Python dependencies
```
### Data Preparation
In the data_preprocessing.py script, the following steps are performed:

1. Loading Data: Load the dataset and additional features.
2. Data Cleaning: Drop irrelevant columns and encode categorical variables.
3. Feature Engineering: Merge features from different sources and select relevant features.
4. Handling Imbalanced Data: Apply SMOTE to balance the classes.
```
# Load the dataset
df = pd.read_excel('/path/to/Neoadj_base_luminaux - anonymise.xls')

# Data Cleaning and Feature Selection
df1 = df.drop([...], axis=1)  # Drop irrelevant columns

# Encoding categorical variables
sata_n_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df1['sata_n'] = df1['sata_n'].replace(sata_n_map)

# Filtering data based on 'chim_typ'
filtered_df = df1.loc[(df1['chim_typ'].isin([1, 2, 5]))]
grouped_df = filtered_df.groupby('anap_kc').apply(lambda x: x.reset_index(drop=True))

# Load additional features
X_df = pd.read_csv('/path/to/perso_mec_imp.csv')
```
### Feature Selection
Different feature selection methods are employed:

- ANOVA: anova_top8 function selects features based on ANOVA test results.
- Gini Index: GiniRF_top8 function selects features based on the Gini index from a Random Forest classifier.
- SVM ANOVA: svm_anova function selects features relevant for SVM classification.

### Machine Learning Models and evaluation

Various classifiers are trained and evaluated, including:

- Random Forest Classifier
- Support Vector Classifier
- Logistic Regression
- Gradient Boosting Classifier
- XGBoost Classifier

The performance of the models is evaluated using cross-validation and metrics such as accuracy, specificity, sensitivity, and confusion matrices.


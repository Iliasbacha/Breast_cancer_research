#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from imblearn.metrics import specificity_score
from imblearn.metrics import sensitivity_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


# In[3]:


df = pd.read_excel('/home/ilias.elbacha/Téléchargements/Neoadj_base_luminaux - anonymise.xls')

df1 = df.drop(['IDENTITE', 'Id','dec_caus','Unnamed: 52','dt_naiss', 'centre', 'dt_diag','dt_pfs','dt_rec','dt_chir','dt_ddn','dt_deb','dt_der','dt_follow','dec_caus'], axis=1)
#sata encoding
sata_n_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
sata_t_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df1['sata_n'] = df1['sata_n'].replace(sata_n_map)
df1['sata_t'] = df1['sata_t'].replace(sata_t_map)

# group by chem typ (1,2,5)
selected_columns = ['chim_typ']
selected_values = [1, 2, 5]

# Filter the data frame based on selected values
filtered_df = df1.loc[(df1[selected_columns[0]].isin(selected_values))]

grouped_df = filtered_df.groupby('anap_kc').apply(lambda x: x.reset_index(drop=True))

X_df = pd.read_csv('/home/ilias.elbacha/perso_mec_imp.csv')


# In[4]:


columns_to_add = ['age_diag', 'adp_clin1', 'tum_histo', 'tum_re', 'tum_rp', 'chim_typ', 'chim_tax', 'chim_ant']

X_df.reset_index(drop=True, inplace=True)
grouped_df.reset_index(drop=True, inplace=True)

features = pd.concat([X_df, grouped_df[columns_to_add]], axis=1)


# In[142]:


# Define the classifiers
clf_1 = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1, random_state=42)
clf_2 = SVC(kernel="linear", probability=True, random_state=42)
clf_3 = LogisticRegression(max_iter=100, random_state=42)
clf_4 = GradientBoostingClassifier()
clf_5 = XGBClassifier()

#def anova_top8(df):
 #   return ['age_diag', 't_clin1', 'tum_histo', 'tum_sbr', 'tum_nucl', 'tum_mito', 'tum_re', 'tum_rp']

# Define the list of feature selection methods and classifiers to use
feature_selection = features  
Kresults_dfs=[]
specificity_scorer = make_scorer(specificity_score)
sensitivity_scorer = make_scorer(sensitivity_score)
auc_scorer = make_scorer(accuracy_score)
scorers = { 'specificity': specificity_scorer, 'sensitivity': sensitivity_scorer, 'auc': auc_scorer}
classifiers = [clf_1, clf_2, clf_3, clf_4, clf_5]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)#pour la grantir la presentation des repartions de class dans chaque fold

X = features
Y = grouped_df["anap_kc"]
for clf in classifiers:
    for scorer_name, scorer in scorers.items():
        cv_results = cross_validate(clf, X, Y, cv=cv, scoring=scorers, return_estimator=True)
        
        clf_names = clf.__class__.__name__
        clf_results = pd.DataFrame({
                
            
            'classifier': clf_names,
            'metric': scorer_name,
            'score': cv_results[f'test_{scorer_name}'].mean(),
            'estimators': cv_results['estimator']})
            
        Kresults_dfs.append(clf_results)
Kresults3 = pd.concat(Kresults_dfs, ignore_index=True)


# In[141]:


Kresults3.head(60)


# In[14]:


Kresults3.to_csv('/home/ilias.elbacha/Bureau/PFE_bacha/Results_perso_impu_all_features.csv', index=False)


# In[19]:


# Define the classifiers
clf_1 = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1, random_state=42)
clf_2 = SVC(kernel="linear", probability=True, random_state=42)
clf_3 = LogisticRegression(max_iter=120, random_state=42)
clf_4 = GradientBoostingClassifier()
clf_5 = XGBClassifier()

# Define the feature selection methods to use
def stackRFGBRL(df):
    return ['adp_clin1', 'tum_multi', 't_clin1', 'age_diag']

def stackSVMRF(df):
    return ['tum_multi', 't_clin1', 'adp_clin1', 'age_diag']

def svm_anova(df):
    return ['age_diag', 'tum_sbr', 'tum_mito', 'tum_re', 'tum_rp']

def GiniRF_top8(df):
    return ['age_diag', 't_clin1','tum_rp','tum_re','t_irm1','t_ech1','t_mam1','tum_mito']


def anova_top8(df):
    return ['age_diag', 't_clin1', 'tum_histo', 'tum_sbr', 'tum_nucl', 'tum_mito', 'tum_re', 'tum_rp']

# Define the list of feature selection methods and classifiers to use
feature_selection_methods = [stackRFGBRL, stackSVMRF, svm_anova, GiniRF_top8,  anova_top8]


# Define the list of feature selection methods and classifiers to use

Kresults_dfs=[]
specificity_scorer = make_scorer(specificity_score)
sensitivity_scorer = make_scorer(sensitivity_score)
auc_scorer = make_scorer(accuracy_score)
scorers = { 'specificity': specificity_scorer, 'sensitivity': sensitivity_scorer, 'auc': auc_scorer}
classifiers = [clf_1, clf_2, clf_3, clf_4, clf_5]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)#pour la grantir la presentation des repartions de class dans chaque fold
for j, select_features in enumerate(feature_selection_methods):
    features_selection = select_features(features)
    X = features[features_selection]
    Y = grouped_df["anap_kc"]
    for clf in classifiers:
        for scorer_name, scorer in scorers.items():
            cv_results = cross_validate(clf, X, Y, cv=cv, scoring=scorers, return_estimator=True)
            clf_names = clf.__class__.__name__
            clf_results = pd.DataFrame({
                
                'feature_selection': select_features.__name__,
                'classifier': clf_names,
                'metric': scorer_name,
                'score': cv_results[f'test_{scorer_name}'].mean(),
                'estimators': cv_results['estimator']})
            
            Kresults_dfs.append(clf_results)
Kresults = pd.concat(Kresults_dfs, ignore_index=True)


# In[21]:


Kresults.to_csv('/home/ilias.elbacha/Bureau/PFE_bacha/stratified_results_perso_impu.csv', index=False)


# In[20]:


print(Kresults)


# In[94]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grids for the classifiers
param_grid_clf_1 = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15]
}

param_grid_clf_2 = {
    'C': [0.1, 1, 10]
}

param_grid_clf_3 = {
    'C': [0.1, 1, 10]
}

param_grid_clf_4 = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7]
}

param_grid_clf_5 = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7]
}

param_grids = [param_grid_clf_1, param_grid_clf_2, param_grid_clf_3, param_grid_clf_4, param_grid_clf_5]

Kresults_dfs = []
specificity_scorer = make_scorer(specificity_score)
scorers = {'specificity': specificity_scorer}


X = features
Y = grouped_df["anap_kc"]
for i, clf in enumerate(classifiers):
    param_grid = param_grids[i]
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring=specificity_scorer, cv=cv, refit=False)
    grid_search.fit(X, Y)
        
    clf_names = clf.__class__.__name__
    clf_results = pd.DataFrame({
        
        'classifier': clf_names,
        'metric': 'specificity',
        'score': grid_search.best_score_,
        'best_params': [grid_search.best_params_],
        'estimator': [None]  # Not refitting, so no best estimator
    })
        
    Kresults_dfs.append(clf_results)

Kresults2 = pd.concat(Kresults_dfs, ignore_index=True)


# In[118]:


Kresults2


# In[33]:


def svm_anova(df):
    return ['age_diag', 'tum_sbr', 'tum_mito', 'tum_re', 'tum_rp']

def anova_top8(df):
    return ['age_diag', 't_clin1', 'tum_histo', 'tum_sbr', 'tum_nucl', 'tum_mito', 'tum_re', 'tum_rp']

# Update the feature_selection_methods list with function names as strings
feature_selection_methods = [svm_anova, anova_top8]


classifier_instances = [clf_1, clf_2, clf_3, clf_4, clf_5]

confusion_matrices = []

for index, row in Kresults2.iterrows():
    clf_name = row['classifier']
    params = row['best_params']
    feature_selection_method_name = row['feature_selection']
    feature_selection_method = next(method for method in feature_selection_methods if method.__name__ == feature_selection_method_name)
    
    X = features[feature_selection_method(features)]
    Y = grouped_df["anap_kc"]
    
    # Find the actual classifier instance based on its name
    clf_instance = classifier_instances[[clf.__class__.__name__ for clf in classifier_instances].index(clf_name)]
    
    clf_instance.set_params(**params)  # Set the best parameters for the classifier
    clf_instance.fit(X, Y)
    y_pred = clf_instance.predict(X)
    cm = confusion_matrix(Y, y_pred)
    confusion_matrices.append(cm)

Kresults2['confusion_matrix'] = confusion_matrices


# In[54]:


for index, row in Kresults2.iterrows():
    clf_name = row['classifier']
    params = row['best_params']
    #feature_selection_method_name = row['feature_selection']
    #feature_selection_method = next(method for method in feature_selection_methods if method.__name__ == feature_selection_method_name)
    
    X = features
    Y = grouped_df["anap_kc"]
    
    # Find the actual classifier instance based on its name
    clf_instance = classifier_instances[[clf.__class__.__name__ for clf in classifier_instances].index(clf_name)]
    
    clf_instance.set_params(**params)  # Set the best parameters for the classifier
    clf_instance.fit(X, Y)
    y_pred = clf_instance.predict(X)
    cm = confusion_matrix(Y, y_pred)
    
    plt.figure()
    plt.title(f'Confusion Matrix - {clf_name}')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[44]:


for index, row in Kresults3.iterrows():
    clf_name = row['classifier']
    #params = row['best_params']
    #feature_selection_method_name = row['feature_selection']
    #feature_selection_method = next(method for method in feature_selection_methods if method.__name__ == feature_selection_method_name)
    
    X = features
    Y = grouped_df["anap_kc"]
    
    # Find the actual classifier instance based on its name
    clf_instance = classifier_instances[[clf.__class__.__name__ for clf in classifier_instances].index(clf_name)]
    
    #clf_instance.set_params(**params)  # Set the best parameters for the classifier
    clf_instance.fit(X, Y)
    y_pred = clf_instance.predict(X)
    cm = confusion_matrix(Y, y_pred)
    
    plt.figure()
    plt.title(f'Confusion Matrix - {clf_name}')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[97]:


# Define the parameter grids for the classifiers
param_grid_clf_1 = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15]
}

param_grid_clf_2 = {
    'C': [0.1, 1, 10]
}

param_grid_clf_3 = {
    'C': [0.1, 1, 10]
}

param_grid_clf_4 = {
    'max_depth': [3, 5, 7]
}

param_grid_clf_5 = {
    'max_depth': [3, 5, 7]
}

param_grids = [param_grid_clf_1, param_grid_clf_2, param_grid_clf_3, param_grid_clf_4, param_grid_clf_5]

# Define scoring functions (sensitivity_scorer, auc_scorer, specificity_scorer) if not shown in your code

# Define sensitivity, specificity, and auc scorers
def sensitivity_score(y_true, y_pred):
    # Calculate sensitivity (true positive rate)
    true_positive = sum((y_true == 1) & (y_pred == 1))
    total_positive = sum(y_true == 1)
    return true_positive / total_positive

def specificity_score(y_true, y_pred):
    # Calculate specificity (true negative rate)
    true_negative = sum((y_true == 0) & (y_pred == 0))
    total_negative = sum(y_true == 0)
    return true_negative / total_negative

def auc_score(y_true, y_pred):
    # Calculate AUC using an appropriate method
    pass  # Replace with your AUC calculation

# Create lists to hold metrics and scorers
metric_names = ['specificity', 'sensitivity', 'auc']
scorers = {
    'specificity': make_scorer(specificity_score),
    'sensitivity': make_scorer(sensitivity_score),
    'auc': make_scorer(auc_score)  # Update this based on your AUC calculation
}

Kresults_dfs = []

# Loop over classifiers and parameter grids
for i, (clf, param_grid) in enumerate(zip(classifiers, param_grids)):
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring=scorers, cv=5, refit=False)
    grid_search.fit(X, Y)
        
    clf_names = clf.__class__.__name__
    
    clf_results = pd.DataFrame({
        'classifier': [clf_names] * len(param_grid),
        'metric': metric_names,
        'score': [grid_search.cv_results_[f"mean_test_{scorer}"][idx] for scorer in metric_names for idx in range(len(param_grid))],
        'best_params': grid_search.cv_results_['params'],
        'estimator': [None] * len(param_grid)
    })
        
    Kresults_dfs.append(clf_results)

Kresults2 = pd.concat(Kresults_dfs, ignore_index=True)


# In[73]:


Kresults3.head(50)


# In[101]:


specificity = 0.08
sensitivity = 0.90    
total_positive = 97
total_negative = 23
total_instances = 120

# Calculate True Negatives (TN)
TN = total_negative * specificity

# Calculate False Positives (FP)
FP = total_negative - TN

# Calculate False Negatives (FN)
FN = total_positive * (1 - sensitivity)

# Calculate True Positives (TP)
TP = total_positive - FN

confusion_matrix = [[TN, FP], [FN, TP]]
print("Confusion Matrix:")
print(confusion_matrix)


# In[102]:


import numpy as np
import matplotlib.pyplot as plt

# Define the confusion matrices for each classifier
confusion_matrices = {
    'GradientBoostingClassifier': np.array([[9, 14], [11, 86]]),
    'LogisticRegression': np.array([[5, 18], [8, 89]]),
    'RandomForestClassifier': np.array([[2, 21], [4, 93]]),
    'SVC': np.array([[1, 22], [3, 94]]),
    'XGBClassifier': np.array([[2, 21], [10, 87]])
}

# Define class labels
class_labels = ['Negative', 'Positive']

# Create a separate heatmap-style plot for each classifier
for classifier, confusion_matrix in confusion_matrices.items():
    plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {classifier}')
    plt.colorbar()

    # Add class labels to the plot
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Add the actual values in the cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='white' if confusion_matrix[i, j] > (0.5 * confusion_matrix.max()) else 'black')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    plt.show()


# In[149]:


import joblib

classifiers = [clf_1, clf_2, clf_3, clf_4, clf_5]
classifier_names = ['RandomForest', 'SVC', 'LogisticRegression', 'GradientBoosting', 'XGB']

for clf, clf_name in zip(classifiers, classifier_names):
    # Save the model to a file
    model_filename = f"{clf_name}_model.pkl"
    joblib.dump(clf, model_filename)
    print(f"Saved {clf_name} model to {model_filename}")


# In[150]:


loaded_classifiers = []

for clf_name in classifier_names:
    # Load the model from the file
    model_filename = f"{clf_name}_model.pkl"
    loaded_clf = joblib.load(model_filename)
    loaded_classifiers.append(loaded_clf)
    print(f"Loaded {clf_name} model from {model_filename}")


# In[151]:


from sklearn.utils import shuffle

# Shuffle your feature matrix (X) and target vector (Y) simultaneously
X_shuffled, Y_shuffled = shuffle(X, Y, random_state=42)


# In[152]:


Kresults_dfs_after_reshuffle = []

for clf in classifiers:
    for scorer_name, scorer in scorers.items():
        cv_results = cross_validate(clf, X_shuffled, Y_shuffled, cv=cv, scoring=scorers, return_estimator=True)
        
        clf_names = clf.__class__.__name__
        clf_results = pd.DataFrame({
            'classifier': clf_names,
            'metric': scorer_name,
            'score': cv_results[f'test_{scorer_name}'].mean(),
            'estimators': cv_results['estimator']})
            
        Kresults_dfs_after_reshuffle.append(clf_results)

Kresults_after_reshuffle = pd.concat(Kresults_dfs_after_reshuffle, ignore_index=True)


# In[162]:


Kresults_after_reshuffle.head(90)


# In[ ]:





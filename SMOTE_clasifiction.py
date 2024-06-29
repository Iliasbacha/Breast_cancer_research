#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix


# In[2]:


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
columns_to_add = ['age_diag', 'adp_clin1', 'tum_histo', 'tum_re', 'tum_rp', 'chim_typ', 'chim_tax', 'chim_ant']

X_df.reset_index(drop=True, inplace=True)
grouped_df.reset_index(drop=True, inplace=True)

features = pd.concat([X_df, grouped_df[columns_to_add]], axis=1)


# In[3]:


# Define the classifiers
clf_1 = RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=1, random_state=42)
clf_2 = SVC(kernel="linear", probability=True, random_state=42,C=10)
clf_3 = LogisticRegression(max_iter=120, random_state=42,C=1)
clf_4 = GradientBoostingClassifier(max_depth=3,n_estimators=100)
clf_5 = XGBClassifier(max_depth=3,n_estimators=100)


# In[5]:


def svm_anova(df):
    return ['age_diag', 'tum_sbr', 'tum_mito', 'tum_re', 'tum_rp']

def GiniRF_top8(df):
    return ['age_diag', 't_clin1','tum_rp','tum_re','t_irm1','t_ech1','t_mam1','tum_mito']

def anova_top8(df):
    return ['age_diag', 't_clin1', 'tum_histo', 'tum_sbr', 'tum_nucl', 'tum_mito', 'tum_re', 'tum_rp']


# In[4]:


feature_selection_methods = features   
results_dfs=[]
classifiers = [clf_1, clf_2, clf_3, clf_4, clf_5]


# In[5]:




specificity_scorer = make_scorer(specificity_score)
sensitivity_scorer = make_scorer(sensitivity_score)
auc_scorer = make_scorer(accuracy_score)
scorers = { 'specificity': specificity_scorer, 'sensitivity': sensitivity_scorer, 'auc': auc_scorer}

K=5

results_dfs = []


X = features
Y = grouped_df["anap_kc"]

# Apply SMOTE for minority class '0' in 'anap_kc'
smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X, Y)

for clf in classifiers:
    clf.fit(X_res, Y_res) 

    cv_results = cross_validate(clf, X_res, Y_res, cv=K, scoring=scorers, return_estimator=True)

    clf_names = clf.__class__.__name__
    clf_results = []

    # Add scores from cross-validation
    for scorer_name in scorers.keys():
        clf_result = {
            
            'classifier': clf_names,
            'metric': scorer_name,
            'score': cv_results[f'test_{scorer_name}'].mean(),
            'estimators': cv_results['estimator']
        }
        clf_results.append(clf_result)

    # Calculate confusion matrix, TP, TN, FP, FN, specificity, sensitivity, and accuracy
    y_pred = clf.predict(X_res)
    cm = confusion_matrix(Y_res, y_pred)
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Add confusion matrix, TP, TN, FP, FN, specificity, sensitivity, and accuracy to results
    clf_result = {
        
        'classifier': clf_names,
        'metric': 'confusion_matrix',
        'score': cm,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'accuracy': accuracy,
        'estimators': cv_results['estimator']
    }
    clf_results.append(clf_result)

    results_dfs.extend(clf_results)

results = pd.DataFrame(results_dfs)


# In[8]:


for model_results in results_dfs:
    
    classifier = model_results['classifier']
    metric = model_results['metric']
    score = model_results['score']

    if metric == 'confusion_matrix':
        confusion_matrix = model_results['score']
        TP = model_results['TP']
        TN = model_results['TN']
        FP = model_results['FP']
        FN = model_results['FN']
        specificity = model_results['specificity']
        sensitivity = model_results['sensitivity']
        accuracy = model_results['accuracy']

        # Plot the confusion matrix
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
        plt.title(f'Confusion matrix (  {classifier})')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Display other metrics
        print(f'{metric} for  {classifier}:')
        print(f'TP: {TP}')
        print(f'TN: {TN}')
        print(f'FP: {FP}')
        print(f'FN: {FN}')
        print(f'Specificity: {specificity}')
        print(f'Sensitivity: {sensitivity}')
        print(f'Accuracy: {accuracy}')


# In[2]:


#Import libriries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, Conv1D, Conv2D,MaxPooling2D, MaxPooling1D, Input, Dropout
from tensorflow.keras.optimizers import Adam
import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[31]:


Y = pd.DataFrame(Y,columns=grouped_df['anap_kc'].columns)

print(Y)


# In[3]:


Y = Y_res["anap_kc"]
X_res["Y"] = Y
train_df, test_df = train_test_split(X_res, test_size=0.2, stratify=X_res['Y'])
# Define the feature selection functions
def svm_anova(df):
    return ['age_diag', 'tum_sbr', 'tum_mito', 'tum_re', 'tum_rp']

def anova_top8(df):
    return ['age_diag', 't_clin1', 'tum_histo', 'tum_sbr', 'tum_nucl', 'tum_mito', 'tum_re', 'tum_rp']

# Select the desired features
selected_features = anova_top8(train_df)  # or anova_top8(train_df)

# Form np arrays of labels and selected features.
train_labels = np.array(train_df.pop('Y'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Y'))
test_labels = np.array(test_df.pop('Y'))

train_features = np.array(train_df[selected_features])
val_features = np.array(val_df[selected_features])
test_features = np.array(test_df[selected_features])


# In[39]:


scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)

print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Test features shape:', test_features.shape)


# In[40]:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model


# In[1]:


EPOCHS = 100
BATCH_SIZE = 32

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='tp', 
    verbose=1,
    patience=50,
    mode='max',
    restore_best_weights=True)
model = make_model()
model.summary()


# In[42]:


model.predict(train_features[:10])


# In[43]:


results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


# In[52]:


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')
]

from sklearn.metrics import accuracy_score, confusion_matrix

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(train_features.shape[-1],)))
    model.add(Dense(16, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid",bias_initializer=output_bias))
    
    model.compile(optimizer="adam", loss='binary_crossentropy',metrics=METRICS)
    
    return model

# Adjust the input shape to match your train_features
input_shape = (train_features.shape[1],)

# Create and compile the model
model = make_model(input_shape)

# Train the model
history = model.fit(train_features, train_labels, epochs=100, batch_size=2028, validation_split=0.20)

# Make predictions
predictions = model.predict(test_features)
pred_labels = np.round(predictions)

# Calculate accuracy, confusion matrix, TP, TN, FP, FN
score = accuracy_score(test_labels, pred_labels)
conf_mx = confusion_matrix(test_labels, pred_labels)

TP = conf_mx[0, 0]
TN = conf_mx[1, 1]
FP = conf_mx[0, 1]
FN = conf_mx[1, 0]

# Calculate specificity and sensitivity
specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

print("Specificity:", specificity)
print("Sensitivity:", sensitivity)


# In[56]:


initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)


# In[57]:


model = make_model(input_shape)
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)


# In[58]:


model = make_model(input_shape)
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)


# In[66]:


def plot_loss(history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  plt.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')


# In[67]:


colors = ['blue', 'red', 'green']
plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)


# In[68]:


model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels))


# In[69]:


def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();
plot_metrics(baseline_history)   


# In[70]:


train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)


# In[76]:


def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Positive responses correctly predicted (True Positive): ', cm[1][1])  # Corrected
  print('Negative responses Incorrectly predicted (False Negative): ', cm[0][1])
  print('Positive responses Incorrectly predicted  (False Positive): ', cm[1][0])  # Corrected
  print('Negative responses correctly predicted (True Negatives): ', cm[0][0])
  print('Total of positive responses ', np.sum(cm[1]))


# In[77]:


baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()
plot_cm(test_labels, test_predictions_baseline)


# In[ ]:


Training labels shape: (124,)
Validation labels shape: (31,)
Test labels shape: (39,)


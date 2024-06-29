#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libriries
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix


# In[3]:


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[7]:


file = tf.keras.utils
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

#imputed_df_MF = pd.read_csv('/home/ilias.elbacha/Bureau/PFE_bacha/imputed_data_miceforest.csv')
Y = grouped_df["anap_kc"].reset_index(drop=True)
#imputed_df_MF["Y"] = Y
#imputed_df_MF.head()
X_df = pd.read_csv('/home/ilias.elbacha/perso_mec_imp.csv')

columns_to_add = ['age_diag', 'adp_clin1', 'tum_histo', 'tum_re', 'tum_rp', 'chim_typ', 'chim_tax', 'chim_ant']

X_df.reset_index(drop=True, inplace=True)
grouped_df.reset_index(drop=True, inplace=True)

features = pd.concat([X_df, grouped_df[columns_to_add]], axis=1)


# In[10]:


Y.count()


# In[11]:


pos, neg = np.bincount(Y)
total = neg + pos
print('Examples:\n    Total: {}\n   Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# In[14]:


# Use a utility from sklearn to split and shuffle your dataset.


# Define the feature selection functions
#"def svm_anova(df):
    #return ['age_diag', 'tum_sbr', 'tum_mito', 'tum_re', 'tum_rp']

#def anova_top8(df):
    #return ['age_diag', 't_clin1', 'tum_histo', 'tum_sbr', 'tum_nucl', 'tum_mito', 'tum_re', 'tum_rp']

# Select the desired features
#selected_features = anova_top8(train_df)  # or anova_top8(train_df)

# Form np arrays of labels and selected features.
# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(features, test_size=0.2)
train_df, val_df = train_test_split(features, test_size=0.3)

# Form np arrays of labels and features.
train_labels = np.array(train_df)
bool_train_labels = train_labels != 0
val_labels = np.array(val_df)
test_labels = np.array(test_df)

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)


# In[15]:


scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


# In[16]:


pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

sns.jointplot(x=pos_df['tum_re'], y=pos_df['tum_rp'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
plt.suptitle("Positive distribution")

sns.jointplot(x=neg_df['tum_re'], y=neg_df['tum_rp'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
_ = plt.suptitle("Negative distribution")


# In[17]:


import keras as K
import tensorflow as tf



K.metrics.BinaryAccuracy(name='accuracy')

def recall(y_true, y_pred, weight=1.0):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall_keras = (weight * true_positives) / (possible_positives + tf.keras.backend.epsilon())
    return recall_keras



def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision_keras

def specificity(y_true, y_pred):
    tn = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + tf.keras.backend.epsilon())


def negative_predictive_value(y_true, y_pred, weight=1):
    tn = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * (1 - y_pred), 0, 1)))
    npv = (weight * tn) / (tn + fn + tf.keras.backend.epsilon())
    return npv


def positive_predictive_value(y_true, y_pred, weight=1):
    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    fp = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * y_pred, 0, 1)))
    ppv = (weight * tp) / (tp + fp + tf.keras.backend.epsilon())
    return ppv


def fbeta(y_true, y_pred, beta=2, weight=1):
    y_pred = tf.clip_by_value(y_pred, 0, 1)

    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axis=1)
    fp = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred - y_true, 0, 1)), axis=1)
    fn = tf.reduce_sum(tf.round(tf.clip_by_value(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = (weight * tp) / (tp + fn + tf.keras.backend.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + tf.keras.backend.epsilon())
    return tf.reduce_mean(num / den)



import numpy as np


def make_model5(learning_rate=0.0001, output_bias=None, random_seed=40):
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(train_features.shape[1],)))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid", bias_initializer=output_bias))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',
                                                                        specificity,
                                                                        negative_predictive_value,
                                                                        positive_predictive_value,
                                                                        fbeta,
                                                                        recall,
                                                                        precision
                                                                        ])

    # Fit the model to training data and store the history object
    #history = model.fit(train_features, train_labels, epochs=100, batch_size=64, validation_data=(val_features, val_labels))

    return model, #history


# In[18]:


model_tuple = make_model5()
model1 = model_tuple[0]

model1.fit(
    train_features,
    train_labels,
    batch_size=46,
    epochs=100,
    
    validation_data=(val_features, val_labels)
)


# In[333]:



def make_scorer_tf(metric):
    def scorer(y_true, y_pred):
        score = metric(y_true, y_pred)
        return np.array(score)

    return make_scorer(scorer)

# Define the parameter grid for grid search
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [100, 500, 1000],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer_tf(negative_predictive_value), cv=3, error_score='raise')

# Fit the grid search to your data
grid_search.fit(train_features, train_labels)

# Retrieve the best model and its performance
best_model1 = grid_search.best_estimator_
best_score = grid_search.best_score_
best_params = grid_search.best_params_


# In[381]:


print('best parameters',best_params)
print('best model',best_model)
print('best score',best_score)


# In[25]:


def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions )
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
plot_cm(test_labels, model.predict(test_features))  
plot_cm(train_labels, model.predict(train_features))  
# Extract the TP and TN values
tn = cm[1, 1]  # True negatives
tp = cm[0, 0]  # True positives

# Print the TP and TN values
print("True Negatives (TN):", tn)
print("True Positives (TP):", tp)


# In[303]:


print(history1.history.keys())


# In[380]:


def plot_metrics(history):
  metrics = ['negative_predictive_value', 'loss','accuracy','specificity']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history1.history[metric], color=colors[0], label='Train')
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();
plot_metrics(history)    


# In[281]:



EPOCHS = 500
BATCH_SIZE = 16

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor= specificity, 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)


# In[382]:


best_model1.predict(train_features[:10])


# In[383]:


results = best_model1.evaluate(train_features, train_labels, batch_size=32, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


# In[200]:


initial_bias = np.log([pos/neg])
initial_bias


# In[201]:


model_init = make_model5(output_bias=initial_bias)
model_init.predict(train_features[:10])


# In[202]:


results = model1.evaluate(train_features, train_labels, batch_size=16, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


# In[283]:


history = best_model.fit(
    train_features,
    train_labels,
    batch_size=46,
    epochs=100,
    callbacks=[early_stopping],
    validation_data=(test_features, test_labels)
)


# In[204]:


initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)


# In[ ]:


# Create an instance of the model
model = make_model5()

# Load initial weights (if available)
model.load_weights(initial_weights)

# Set bias of the last layer to 0.0
model.layers[-1].bias.assign([0.0])

# Train the model
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=16,
    epochs=500,
    validation_data=(test_features, test_labels),
    verbose=0
)


# In[ ]:


model = model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=16,
    epochs=100,
    validation_data=(test_features, test_labels), 
    verbose=0)


# def plot_loss(history, label, n):
#   plt.semilogy(history.epoch, history.history['loss'],
#                color=colors[n], label='Train ' + label)
#   plt.semilogy(history.epoch, history.history['val_loss'],
#                color=colors[n], label='Val ' + label,
#                linestyle="--")
#   plt.xlabel('Epoch')
#   plt.ylabel('Loss')

# In[ ]:


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)


# In[ ]:



model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=16,
    epochs=500,
    callbacks=[early_stopping],
    validation_data=(test_features, test_labels))


# In[1]:


def plot_metrics(history):
  metrics = ['loss', 'prc', 'accuracy', 'recall']
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


# In[ ]:


train_predictions_baseline = model.predict(train_features, batch_size=32)
test_predictions_baseline = model.predict(test_features, batch_size=32)
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('(True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))


# In[ ]:



baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=32, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)


# # Weight class

# In[205]:


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 1)
weight_for_1 = (1 / pos) * (total / 6.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# In[206]:


weighted_model = model
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=16,
    epochs=100,
    callbacks=[early_stopping],
    validation_data=(test_features, test_labels),
    # The class weights go here
    class_weight=class_weight)


# In[207]:


plot_metrics(weighted_history)


# In[208]:


train_predictions_weighted = weighted_model.predict(train_features, batch_size=32)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=32)


# In[209]:


weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=32, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)


# In[ ]:





# In[310]:


#oversampling


# In[91]:


pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]


# In[92]:


ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape


# In[ ]:


# Create and compile the model
model = make_model()

# Train the model
history = model.fit(selected_features, Y, epochs=50, batch_size=32, validation_split=0.20)

# Make predictions
predictions = model.predict(x_test)
pred_labels = np.round(predictions)

# Calculate accuracy, confusion matrix, TP, TN, FP, FN
score = accuracy_score(y_test, pred_labels)
conf_mx = confusion_matrix(y_test, pred_labels)

TP = conf_mx[0, 0]
TN = conf_mx[1, 1]
FP = conf_mx[0, 1]
FN = conf_mx[1, 0]

# Calculate specificity and sensitivity
specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

print("Specificity:", specificity)
print("Sensitivity:", sensitivity)


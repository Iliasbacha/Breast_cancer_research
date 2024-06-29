#!/usr/bin/env python
# coding: utf-8

# In[14]:


pip install miceforest


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from scipy.stats import wasserstein_distance
import miceforest as mf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from impyute.imputation.cs import mice


# In[18]:


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

Y = grouped_df["anap_kc"]
X = grouped_df.drop(columns=["gg_env","t_irm3","rec_typ","t_irm2","anap_kc", "t_histo","sata_t","sata_n","rec_loc","rec_cont","recidive","ttt_hormo","ttt_rt","etat","rec_meta","gg_cura","t_clin2","t_clin3",'CHIR', 'chir_cons', 'Chim_typ2','adp_clin2', 'adp_clin3 '])


# In[19]:


X.describe()


# In[20]:


X.count()


# In[21]:


# Step 1: Identify MAR and MNAR Variables
MAR_var = ['tum_sbr', 'tum_nucl', 'tum_mito', 'tum_multi']
MNAR_var = ['t_clin1', 't_irm1', 't_mam1', 't_ech1']
No_MV_var = ['age_diag','adp_clin1','tum_histo','tum_re','tum_rp','chim_typ','chim_tax','chim_ant']

# Step 2: Impute MAR Variables using MICEforest
imputed_data_mar = mice(X[MAR_var].values, method='miceforest', min_value=0, max_value=3, max_iter=10, verbose=True)

# Step 3: Impute MNAR Variables using MICE
imputed_data_mnar = mice(X[MNAR_var].values,min_value=0, max_value=100, max_iter=100)

# Step 4: Combine Imputed Data
imputed_data = pd.DataFrame(imputed_data_mar, columns=X[MAR_var].columns)
imputed_data[MNAR_var] = imputed_data_mnar


# In[22]:


for variable in imputed_data.columns:
    # Plot density distribution of the original data
    sns.kdeplot(X[variable], label='Original Data')

    # Plot density distribution of the imputed data
    sns.kdeplot(imputed_data[variable], label='Imputed Data')

    # Set plot title and labels
    plt.title(f'Density Distribution Comparison for {variable}')
    plt.xlabel(variable)
    plt.ylabel('Density')

    # Show the plot
    plt.legend()
    plt.show()


# In[23]:


# Flatten the imputed data and original data into series
imputed_data_flattened = pd.Series(imputed_data.values.flatten())
x_flattened = pd.Series(X[MAR_var + MNAR_var].values.flatten())

# Plot the density distributions of the imputed data and original data
sns.kdeplot(imputed_data_flattened, label='Imputed Data')
sns.kdeplot(x_flattened, label='Original Data')

# Set plot title and labels
plt.title('Overall Density Distribution')
plt.xlabel('Value')
plt.ylabel('Density')

# Show the plot
plt.legend()
plt.show()


# In[24]:


from scipy.stats import wasserstein_distance

# Combine original and imputed data into one DataFrame
data = pd.concat([X[MAR_var + MNAR_var], imputed_data[MAR_var + MNAR_var]], ignore_index=True)
data['Data'] = ['Original'] * len(X) + ['Imputed'] * len(imputed_data)

# Compute the Wasserstein distance for each variable
w_distances = {}
for variable in MAR_var + MNAR_var:
    # Select the data for the variable
    variable_data = data.loc[data[variable].notna(), variable]
    
    # Compute the Wasserstein distance
    w_distance = wasserstein_distance(variable_data[:len(X)], variable_data[len(X):])
    
    # Store the Wasserstein distance
    w_distances[variable] = w_distance


# In[25]:


data.count()


# In[26]:


print( w_distances)


# In[27]:


# Compute mean, standard deviation, and median of Wasserstein distances
mean_distance = np.mean(list(w_distances.values()))
std_distance = np.std(list(w_distances.values()))
median_distance = np.median(list(w_distances.values()))

# Print the results
print(f"Mean Wasserstein Distance: {mean_distance:.4f}")
print(f"Standard Deviation of Wasserstein Distances: {std_distance:.4f}")
print(f"Median Wasserstein Distance: {median_distance:.4f}")


# In[28]:


imputed_data.to_csv('perso_mec_imp.csv', index=False)


# In[13]:


# Assuming X is your original DataFrame with missing values
# and MAR_var, MNAR_var are the variables with missing values

# Step 1: Create a mask for missing values in the original data
mask = X.isnull()

# Step 2: Apply imputation method to masked data
imputed_data_mar = mice(X[MAR_var].values, method='miceforest', min_value=0, max_value=3, max_iter=10, verbose=True)
imputed_data_mnar = mice(X[MNAR_var].values, min_value=0, max_value=3, max_iter=100)

# Step 3: Create DataFrames for imputed data
imputed_data_mar = pd.DataFrame(imputed_data_mar, columns=X[MAR_var].columns)
imputed_data_mnar = pd.DataFrame(imputed_data_mnar, columns=X[MNAR_var].columns)

# Step 4: Combine imputed data
imputed_data = pd.concat([imputed_data_mar, imputed_data_mnar], axis=1)

# Step 5: Create true values DataFrame by copying original data and updating missing values with imputed data
true_values = X.copy()
true_values.loc[mask[MAR_var], MAR_var] = imputed_data_mar
true_values.loc[mask[MNAR_var], MNAR_var] = imputed_data_mnar

# Step 6: Evaluate the imputation method using MAE and MSE
mae = mean_absolute_error(true_values, X)
mse = mean_squared_error(true_values, X)

print("MAE:", mae)
print("MSE:", mse)


# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 03:47:33 2023

@author: cara-lifarrell
"""

# Individual Project
## Cara-Li Farrell (261051787)

###############################################################################
###############################################################################
############################ Classification Model #############################
############################################################################### 
###############################################################################

###############################################################################
################################ Test Dataset #################################
############################################################################### 

################################### Set Up ####################################

# Importing libraries
import pandas as pd
    
# Importing dataset
kickstarter = pd.read_excel("Kickstarter.xlsx")

############################### Preprocessing ################################# 

# 0. Moving the y variable as the first column + setting up y variable

## state ##
columns_order = ['state'] + [col for col in kickstarter.columns if col != 'state']
kickstarter = kickstarter[columns_order]

# Dropping 'suspended' and 'canceled' state values
kickstarter = kickstarter.drop(kickstarter[kickstarter['state'] == 'suspended'].index)
kickstarter = kickstarter.drop(kickstarter[kickstarter['state'] == 'canceled'].index)

# Checking the remaining possible values
kickstarter.shape
state_counts = kickstarter['state'].value_counts()
print(state_counts)

# Making it a binary variable, where 0: failed and 1: successful
for index, row in kickstarter.iterrows():
    if row['state'] == 'successful':
        kickstarter.at[index, 'state'] = 1
    else:
        kickstarter.at[index, 'state'] = 0

# Check unique state values
unique_states = kickstarter['state'].unique()
print(unique_states)

# 1. Duplicates

kickstarter[kickstarter.duplicated()].shape # there are no duplicates

# Drop duplicates
kickstarter = kickstarter.drop_duplicates()
kickstarter.shape

# 2. Irrelevant columns

# Drop unecessary columns
kickstarter = kickstarter.drop(columns=['id', 
                                        'name',
                                        'pledged',
                                        'disable_communication',
                                        'currency',
                                        'deadline',
                                        'state_changed_at',
                                        'created_at',
                                        'launched_at',
                                        'staff_pick',
                                        'backers_count',
                                        'usd_pledged',
                                        'spotlight',
                                        'name_len_clean',
                                        'blurb_len_clean',
                                        'state_changed_at_weekday',
                                        'state_changed_at_month',
                                        'state_changed_at_day',	
                                        'state_changed_at_yr',
                                        'state_changed_at_hr',
                                        'launch_to_state_change_days'], axis=1)

# Drop columns with insignificant scores from the feature importance score
kickstarter = kickstarter.drop(columns=['country',
                                        'deadline_weekday',
                                        'created_at_weekday',
                                        'launched_at_weekday'], axis=1)

# 3. Missing values

# Checking missing values
print(kickstarter.isnull().sum()) # category, name_len, and blurb_len have missing values

# Drop missing values
kickstarter = kickstarter.dropna()

kickstarter.shape
print(kickstarter.isnull().sum())
    
# 4. Miscellaneous

## goal ##
# Create a new column called goal_usd with the USD value of the goal
kickstarter['goal_usd'] = kickstarter['goal']*kickstarter['static_usd_rate']

# Drop the old goal column and static_usd_rate
kickstarter = kickstarter.drop(columns=['goal'], axis=1)  
kickstarter = kickstarter.drop(columns=['static_usd_rate'], axis=1)    

# 5. Dummify the categorical variables
# Remaining categorical columns
categorical_columns = ['category']
                       #'country',
                       #'deadline_weekday',
                       #'created_at_weekday',
                       #'launched_at_weekday']

# Create dummy variables for the specified columns
kickstarter = pd.get_dummies(kickstarter, columns=categorical_columns)

# Make sure all columns are numeric
kickstarter.dtypes
for column in kickstarter.columns:
    kickstarter[column] = pd.to_numeric(kickstarter[column], errors='coerce')
    
kickstarter.dtypes

############################ Classification Model ############################# 

########################## Setting up the Variables ########################### 

# Constructing the variables
X = kickstarter.iloc[:,1:]
y = kickstarter['state']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

###################### Gradient Boosting Algorithm Model ######################

# Building the model
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state = 5)
model_gbt = gbt.fit(X_train, y_train)

# Print feature importance
feature_importance_df = pd.DataFrame(list(zip(X.columns, model_gbt.feature_importances_)), columns = ['predictor','feature importance'])
print(feature_importance_df)

'''
# Finding the optimal combinations of hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid_gbt = {
    'n_estimators': [50, 100, 150, 200],      # Vary from 50 to 200 with increments of 50
    'max_features': [3, 4, 5, 6],             # Vary from 3 to 6
    'min_samples_leaf': [1, 2, 3, 4]          # Vary from 1 to 4
    }

# Create the GridSearchCV object
grid_search_gbt = GridSearchCV(estimator=gbt, param_grid=param_grid_gbt, scoring='accuracy', cv=5, verbose=True)

# Fit the GridSearchCV object to your training data
gbt_model = grid_search_gbt.fit(X_train, y_train)

# Retrieve the best hyperparameters and model
best_params_gbt = grid_search_gbt.best_params_
best_model_gbt = grid_search_gbt.best_estimator_

# Print the best combination of hyperparameters
print(best_params_gbt)

# Evaluate the best model on your test data
accuracy_gbt = grid_search_gbt.best_score_

# Print the accuracy score
print(accuracy_gbt) # 0.7629604672829723
'''

# Final model with optimal parameters
gbt = GradientBoostingClassifier(n_estimators=200, max_features=4, min_samples_leaf=1, random_state=5)
model_gbt = gbt.fit(X_train, y_train)

# Calculate accuracy score
y_test_pred = model_gbt.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_test_pred)





###############################################################################
############################### Grading Dataset ###############################
############################################################################### 

################################### Set Up ####################################

# Importing libraries
import pandas as pd
    
# Importing dataset
kickstarter_grading = pd.read_excel("Kickstarter-Grading-Sample.xlsx")

############################### Preprocessing ################################# 

# 0. Moving the y variable as the first column + setting up y variable

## state ##
columns_order2 = ['state'] + [col for col in kickstarter_grading.columns if col != 'state']
kickstarter_grading = kickstarter_grading[columns_order2]

# Dropping 'suspended' and 'canceled' state values
kickstarter_grading = kickstarter_grading.drop(kickstarter_grading[kickstarter_grading['state'] == 'suspended'].index)
kickstarter_grading = kickstarter_grading.drop(kickstarter_grading[kickstarter_grading['state'] == 'canceled'].index)

# Checking the remaining possible values
kickstarter_grading.shape
state_counts2 = kickstarter_grading['state'].value_counts()
print(state_counts2)

# Making it a binary variable, where 0: failed and 1: successful
for index, row in kickstarter_grading.iterrows():
    if row['state'] == 'successful':
        kickstarter_grading.at[index, 'state'] = 1
    else:
        kickstarter_grading.at[index, 'state'] = 0

# Check unique state values
unique_states2 = kickstarter_grading['state'].unique()
print(unique_states2)

# 1. Duplicates

kickstarter_grading[kickstarter_grading.duplicated()].shape # there are no duplicates

# Drop duplicates
kickstarter_grading = kickstarter_grading.drop_duplicates()
kickstarter_grading.shape

# 2. Irrelevant columns

# Drop unecessary columns
kickstarter_grading = kickstarter_grading.drop(columns=['id', 
                                                        'name',
                                                        'pledged',
                                                        'disable_communication',
                                                        'currency',
                                                        'deadline',
                                                        'state_changed_at',
                                                        'created_at',
                                                        'launched_at',
                                                        'staff_pick',
                                                        'backers_count',
                                                        'usd_pledged',
                                                        'spotlight',
                                                        'name_len_clean',
                                                        'blurb_len_clean',
                                                        'state_changed_at_weekday',
                                                        'state_changed_at_month',
                                                        'state_changed_at_day',	
                                                        'state_changed_at_yr',
                                                        'state_changed_at_hr',
                                                        'launch_to_state_change_days'], axis=1)

# Drop columns with insignificant scores from the feature importance score
kickstarter_grading = kickstarter_grading.drop(columns=['country',
                                                        'deadline_weekday',
                                                        'created_at_weekday',
                                                        'launched_at_weekday'], axis=1)

# 3. Missing values

# Checking missing values
print(kickstarter_grading.isnull().sum()) # category, name_len, and blurb_len have missing values

# Drop missing values
kickstarter_grading = kickstarter_grading.dropna()

kickstarter_grading.shape
print(kickstarter_grading.isnull().sum())
    
# 4. Miscellaneous

## goal ##
# Create a new column called goal_usd with the USD value of the goal
kickstarter_grading['goal_usd'] = kickstarter_grading['goal']*kickstarter_grading['static_usd_rate']

# Drop the old goal column and static_usd_rate
kickstarter_grading = kickstarter_grading.drop(columns=['goal'], axis=1)  
kickstarter_grading = kickstarter_grading.drop(columns=['static_usd_rate'], axis=1)    

# 5. Dummify the categorical variables
# Remaining categorical columns
categorical_columns2 = ['category']
                       #'country',
                       #'deadline_weekday',
                       #'created_at_weekday',
                       #'launched_at_weekday']

# Create dummy variables for the specified columns
kickstarter_grading = pd.get_dummies(kickstarter_grading, columns=categorical_columns2)

# Make sure all columns are numeric
kickstarter_grading.dtypes
for column in kickstarter_grading.columns:
    kickstarter_grading[column] = pd.to_numeric(kickstarter_grading[column], errors='coerce')
    
kickstarter_grading.dtypes

############################ Classification Model ############################# 

########################## Setting up the Variables ########################### 

# Constructing the variables
X_grading = kickstarter_grading.iloc[:,1:]
y_grading = kickstarter_grading['state']

###################### Gradient Boosting Algorithm Model ######################

# Apply the model previously trained to the grading data
y_grading_pred = model_gbt.predict(X_grading)

# Calculate the accuracy score
from sklearn.metrics import accuracy_score
accuracy_grading = accuracy_score(y_grading, y_grading_pred)












###############################################################################
###############################################################################
############################## Clustering Model ###############################
############################################################################### 
###############################################################################

################################### Set Up ####################################

# Importing libraries
import pandas as pd
    
# Importing dataset
kickstarter3 = pd.read_excel("Kickstarter.xlsx")

############################### Preprocessing ################################# 

# 0. Moving the y variable as the first column + setting up y variable

## state ##
columns_order = ['state'] + [col for col in kickstarter3.columns if col != 'state']
kickstarter3 = kickstarter3[columns_order]

# Dropping 'suspended' and 'canceled' state values
kickstarter3 = kickstarter3.drop(kickstarter3[kickstarter3['state'] == 'suspended'].index)
kickstarter3 = kickstarter3.drop(kickstarter3[kickstarter3['state'] == 'canceled'].index)

# Checking the remaining possible values
kickstarter3.shape
state_counts = kickstarter3['state'].value_counts()
print(state_counts)

# Making it a binary variable, where 0: failed and 1: successful
for index, row in kickstarter3.iterrows():
    if row['state'] == 'successful':
        kickstarter3.at[index, 'state'] = 1
    else:
        kickstarter3.at[index, 'state'] = 0

# Check unique state values
unique_states = kickstarter3['state'].unique()
print(unique_states)

# 1. Duplicates

kickstarter3[kickstarter3.duplicated()].shape # there are no duplicates

# Drop duplicates
kickstarter3 = kickstarter3.drop_duplicates()
kickstarter3.shape

# 2. Irrelevant columns

# Drop unecessary columns
kickstarter3 = kickstarter3.drop(columns=['id', 
                                        'name',
                                        'pledged',
                                        'disable_communication',
                                        'currency',
                                        'deadline',
                                        'state_changed_at',
                                        'created_at',
                                        'launched_at',
                                        'name_len_clean',
                                        'blurb_len_clean'], axis=1)

# 3. Missing values

# Checking missing values
print(kickstarter3.isnull().sum()) # category have missing values

# Drop missing values
kickstarter3 = kickstarter3.dropna()

kickstarter3.shape
print(kickstarter3.isnull().sum())
    
# 4. Miscellaneous

## goal ##
# Create a new column called goal_usd with the USD value of the goal
kickstarter3['goal_usd'] = kickstarter3['goal']*kickstarter3['static_usd_rate']

# Drop the old goal column and static_usd_rate
kickstarter3 = kickstarter3.drop(columns=['goal'], axis=1)  
kickstarter3 = kickstarter3.drop(columns=['static_usd_rate'], axis=1)    

# 5. Dummify the categorical variables
# Remaining categorical columns
categorical_columns = ['country',
                       'staff_pick',
                       'category',
                       'spotlight',
                       'deadline_weekday',
                       'state_changed_at_weekday',
                       'created_at_weekday',
                       'launched_at_weekday']

# Create dummy variables for the specified columns
kickstarter3 = pd.get_dummies(kickstarter3, columns=categorical_columns)

############################## Clustering Model ############################### 

########################## Setting up the Variables ########################### 

# Constructing the variable
X3 = kickstarter3.iloc[:,:]

# Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler3 = StandardScaler() 
X_std3 = scaler3.fit_transform(X3)

################################ K-Means Model ################################

# Finding the optimal K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for i in range (2,8):    
    kmeans = KMeans(n_clusters=i, random_state=5)
    kmeans_model = kmeans.fit(X_std3)
    kmeans_labels = kmeans_model.labels_
    print(i,':',silhouette_score(X_std3,kmeans_labels))
    
# Optimal k is 6 with the highest silhouette score (0.08393927635646167)

# Run kmeans with k=6
kmeans = KMeans(n_clusters=6, random_state=5)
kmeans_model = kmeans.fit(X_std3)
kmeans_labels = kmeans_model.labels_

# Counting the number of observations in each cluster
# Turning into a dataframe
df_kmeans_labels = pd.DataFrame({'Cluster_Labels': kmeans_labels})

# Counting observations
for i in sorted(df_kmeans_labels['Cluster_Labels'].unique()):
    cluster_count = (df_kmeans_labels['Cluster_Labels'] == i).sum()
    print(f"Cluster {i+1}: {cluster_count} observations")
    
# Generate insights
kmeans_centers = kmeans_model.cluster_centers_
df_kmeans_centers = pd.DataFrame(kmeans_centers, columns=kickstarter3.columns)

# Create a dataframe with the category and the labels
kickstarter3['Cluster_Labels'] = df_kmeans_labels['Cluster_Labels']

# Moving the cluster label column to the first column
columns_order3 = ['Cluster_Labels'] + [col for col in kickstarter3.columns if col != 'Cluster_Labels']
kickstarter3 = kickstarter3[columns_order3]

# Getting the summary statistics of each cluster

# Unique
unique_clusters = kickstarter3['Cluster_Labels'].unique()

# Get actual cluster numbers (they start at 0)
kickstarter3['Cluster_Labels'] = kickstarter3['Cluster_Labels']+1

# Get the summary statistics of each cluster

# Defining a function that will take in the cluster label number we are looking 
# for and getting its summary statistics

def cluster_summary_statistics(data, cluster_label):
    cluster_data = data[data['Cluster_Labels'] == cluster_label] # for the rows we want
    summary_statistics = cluster_data.describe().transpose()
    return summary_statistics

cluster1 = cluster_summary_statistics(kickstarter3, 1)
cluster2 = cluster_summary_statistics(kickstarter3, 2)
cluster3 = cluster_summary_statistics(kickstarter3, 3)
cluster4 = cluster_summary_statistics(kickstarter3, 4)
cluster5 = cluster_summary_statistics(kickstarter3, 5)
cluster6 = cluster_summary_statistics(kickstarter3, 6)

# Counting the number of each status (success or failure) in each cluster

def count_states_in_clusters(dataset):
    # Check if required columns are present
    required_columns = ["state", "Cluster_Labels"]
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError("Required columns 'state' and 'Cluster_Labels' are missing.")

    # Count the number of each 'state' for each 'Cluster_Labels'
    count_table = dataset.groupby(['Cluster_Labels', 'state']).size().unstack(fill_value=0)

    return count_table

state_counts = count_states_in_clusters(kickstarter3)
print(state_counts)

# Success ratio
success_ratios = state_counts[1]/(state_counts[0]+state_counts[1])

# Plotting average goal and pledged
import matplotlib.pyplot as plt

# Data averages
pledged = [18575, 15603, 15747, 14979, 17454, 12769]
goal = [68574, 67971, 115635, 73750, 57919, 85465]
cluster_labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6"]

# Plotting
plt.figure(figsize=(10, 6))
for i, (g, p, label) in enumerate(zip(goal, pledged, cluster_labels)):
    plt.scatter(g, p, label=label, marker='o')
    plt.text(g, p, f'{label}\n({g}, {p})', fontsize=8, ha='right', va='bottom')

plt.title('Pledged vs Goal with Cluster Labels')
plt.xlabel('Goal')
plt.ylabel('Pledged')
plt.grid(True)
plt.legend()
plt.show()

'''  
################### Hierarchical Agglomerative Clustering #####################

# Finding the optimal k
from sklearn.cluster import AgglomerativeClustering

for i in range (2,8):    
    hac = AgglomerativeClustering(n_clusters=i, affinity="euclidean", linkage="complete")
    hac_model = hac.fit(X_std3)
    hac_labels = hac_model.labels_
    print(i,':',silhouette_score(X_std3,hac_labels))

# Optimal K is 2 with the highest silhouette score (0.8803809459407886)

# Run hirearchical with k=2
hac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete') 
hac_model = hac.fit_predict(X_std3) 
hac_labels = hac.labels_

# Counting the number of observations in each cluster
# Turning into a dataframe
df_hac_labels = pd.DataFrame({'Cluster_Labels': hac_labels})

# Counting the number of observations in each cluster
for i in sorted(df_hac_labels['Cluster_Labels'].unique()):
    cluster_count = (df_hac_labels['Cluster_Labels'] == i).sum()
    print(f"Cluster {i+1}: {cluster_count} observations")

# There is only 1 observation in cluster 2
'''



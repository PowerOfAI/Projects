# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:55:59 2022

@author: Pawan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import copy



# loading dataset
df1 = pd.read_csv("CreditAnalysis_data.csv")

# keeping original data intact
df = copy.deepcopy(df1)

# Preprosseing 
df.isna().sum() #ordereditem_product_id has missing values

# mode imputation as product id is categorical data
df.ordereditem_product_id = df.ordereditem_product_id.fillna(statistics.mode(df.ordereditem_product_id))
df.isna().sum()

# unique values count coulumwise
df.master_order_status.value_counts()

df.dtypes

# changing datatypes of 'created' column from 'object' to 'datatime'
df['created'] = pd.to_datetime(df['created'])
df.dtypes

# dumping unwanted columns
df.drop(['Unnamed: 0', 'master_order_id', 'master_order_status', 'ordereditem_product_id'], axis=1, inplace=True)
df.head()

from datetime import timedelta
snapshot_date = df['created'].max()+ timedelta(days=1)

rfm_df = df.groupby('retailer_names', as_index=False).agg({'created': lambda x: (snapshot_date-x.max()).days,
                                           'order_id':'count',
                                           'value':'sum'})
rfm_df.columns = 'retailer_names','recency', 'frequency', 'monetary'

plt.figure(figsize=(12,10))
plt.subplot(3,1,1); sns.distplot(rfm_df['recency']) # Plotting distribution of recency
plt.subplot(3,1,2); sns.distplot(rfm_df['frequency']) # Plotting distribution of frequency
plt.subplot(3,1,3); sns.distplot(rfm_df['monetary']) # Plotting distribution of monetory
plt.show()

# =============================================================================

# K-Means Clustering

from sklearn.cluster import KMeans

km_df = copy.deepcopy(rfm_df)
km_df.describe()

#normalizing the data as the scale diff is seen

from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()

km_df[['recency', 'frequency', 'monetary']] = scaler_std.fit_transform(km_df[['recency', 'frequency', 'monetary']])

twss = []
n = list(range(2,16,1))
for number in n:
    km = KMeans(n_clusters=number, init='k-means++').fit(km_df.iloc[:,1:])
    twss.append(km.inertia_)

plt.plot(n, twss)
plt.xlabel('n_clusters')
plt.ylabel('twss')
plt.title('Elbow Curve')
plt.show()

# as per elbow curve of standardized data, the optimum n_clusters are 10

km_mod = KMeans(n_clusters=10, init='k-means++').fit(km_df.iloc[:,1:])
rfm_df['km_labels'] = km_mod.labels_

unique_kmlabel_count = rfm_df.km_labels.value_counts().reset_index()

plt.pie(unique_kmlabel_count.km_labels, labels=unique_kmlabel_count.index, autopct='%.2f')

# =============================================================================

# verifing the fact "20% of the customers give 80% of the business" checking upon km label accuracy

top_spender = rfm_df.groupby('retailer_names', as_index=False).agg({'monetary':'sum', 'frequency':'sum'}).sort_values(ascending=False, by='monetary')

plt.scatter(top_spender.monetary, top_spender.frequency)

top_spender.iloc[:,1:].corr()
sns.pairplot(top_spender.iloc[:,1:])

phenomina_80_20 = top_spender.monetary.sum(), top_spender.frequency.sum(), top_spender.monetary[:(int(len(top_spender.monetary)*20/100))].sum(), top_spender.frequency[:(int(len(top_spender.frequency)*20/100))].sum()
phenomina_80_20 = pd.DataFrame(phenomina_80_20).transpose()
phenomina_80_20.columns = 'Total_sum_of_monetary', 'total_sum_of_frequency', 'sum_of_monetary_from_top_20%_spenders', 'sum_of_orders_from_top20_spenders'

phenomina_80_20['monetary%of_top20%spenders'] = phenomina_80_20['sum_of_monetary_from_top_20%_spenders']*100/phenomina_80_20.Total_sum_of_monetary
phenomina_80_20['ordcount%of_top20%spenders'] = phenomina_80_20['sum_of_orders_from_top20_spenders']*100/phenomina_80_20.total_sum_of_frequency
phenomina_80_20 = (phenomina_80_20).transpose().reset_index()
phenomina_80_20.columns = 'labels', 'value'

km_label_accu_check = rfm_df.sort_values(ascending=False, by='monetary')

# top 20% spendars are contributing 96% to total revenue and 72% to total order count
# but those 20% of the customers are clustered in multiple diff groups by kmeans, which should not be the case
# hence creating ratings based on ".rank" function on recency, frequency and monetary
# =============================================================================


# creating additional ranking columns for r, f, m using .rank function
rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

# normalizing the rank of the customers
rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*10
rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*10
rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*10
rfm_df.columns

# final ranking 
rfm_df['final_ranking'] = round(rfm_df[['r_rank_norm', 'f_rank_norm', 'm_rank_norm']].mean(axis=1))

final_ranking_accu_check = rfm_df[['retailer_names', 'recency', 'frequency', 'monetary', 'final_ranking']].sort_values(by='monetary', ascending=False)
# =============================================================================

# Above rating machanism is also not accurate as the final ranking were not based 
# on distribution density and were derived as the mean of all r, f and m scores
# 1 unit change in recency giving huge variance in final ranking, while ord frequency and monetary values are almost similar (this should not be the case)


# Hence, Creating custom defined ratings with more weightage to monetary than to frequency and least to recency
# as there might be some customers who just purchase once in a while with good amount
# =============================================================================

new_r_rank = []

for recency in rfm_df['recency']:
    if recency >= 1 and recency <= 5:
        new_r_rank.append(10)
    elif recency >= 6 and recency <= 10:
        new_r_rank.append(9)
    elif recency >= 11 and recency <= 15:
        new_r_rank.append(8)
    elif recency >= 16 and recency <= 20:
        new_r_rank.append(7)
    elif recency >= 21 and recency <= 25:
        new_r_rank.append(6)
    elif recency >= 26 and recency <= 30:
        new_r_rank.append(5)
    elif recency >= 31 and recency <= 35:
        new_r_rank.append(4)
    elif recency >= 36 and recency <= 40:
        new_r_rank.append(3)
    elif recency >= 41 and recency <= 45:
        new_r_rank.append(2)
    elif recency >= 46:
        new_r_rank.append(1)
    else:
        new_r_rank.append('Invalid Recency')
        
rfm_df['new_r_rank'] = new_r_rank 

# =============================================================================

# creating score for frequency

g = rfm_df.groupby('frequency').frequency.count().sort_values(ascending=False) 
# 1/3rd of the total frequency is under 20 order
 

new_f_rank = []       
for frequency in rfm_df['frequency']:
    if frequency >= 1 and frequency <= 2:
        new_f_rank.append(1)
    elif frequency >= 3 and frequency <= 5:
        new_f_rank.append(2)
    elif frequency >= 6 and frequency <= 10:
        new_f_rank.append(3)
    elif frequency >= 11 and frequency <= 15:
        new_f_rank.append(4)
    elif frequency >= 16 and frequency <= 20:
        new_f_rank.append(5)
    elif frequency >= 21 and frequency <= 30:
        new_f_rank.append(6)
    elif frequency >= 31 and frequency <= 50:
        new_f_rank.append(7)
    elif frequency >= 50 and frequency <= 100:
        new_f_rank.append(8)
    elif frequency >= 101 and frequency <= 500:
        new_f_rank.append(9)
    elif frequency > 500: 
        new_f_rank.append(10)
    else:
        new_r_rank.append('Invalid Frequency')
        
rfm_df['new_f_rank'] = new_f_rank

# =============================================================================
# creating score for monetary

# rfm_df.monetary.describe() 
# '''
# count    2.150000e+02
# mean     1.238127e+05
# std      6.567191e+05
# min      4.250000e+01
# 25%      1.815555e+03
# 50%      5.229330e+03
# 75%      2.204974e+04
# max      8.286501e+06
# '''
# =============================================================================

sns.boxplot(rfm_df.monetary)

gb = rfm_df.groupby('retailer_names').monetary.sum().sort_values(ascending=False)

# The range of monetary is quite long, hence creating amounts' buckets

m_above100k = []
m_under100k = []
m_under10k = []
m_under1k = []
for amount in rfm_df['monetary']:
    if amount >=100000:
        m_above100k.append(amount)    
    if amount <=100000 and amount >10000:
        m_under100k.append(amount)
    if amount <=10000 and amount >1000:
        m_under10k.append(amount)
    if amount <=1000:
        m_under1k.append(amount)
    else:
        pass

# weitage of each monetary_group in overall data
weitage_dist_above100k = (len(m_above100k)*100)/215 # 8.37%
weitage_dist_under100k = (len(m_under100k)*100)/215 # 27.90%
weitage_dist_under10k = (len(m_under10k)*100)/215 #49.30%
weitage_dist_under1k = (len(m_under1k)*100)/215 # 14.41%

# Based on above distribution weitage, assigning score from 10 to 1

new_m_rank = []
for i in rfm_df['monetary']:
    
    # Assigning 1 score for above 100k bucket (As group weightage is 8.37%)
    
    if i >=100000:
        new_m_rank.append(10)
        
        #Assigning 3 scores for 100k to 10k bucket (As group weightage is 27.90%)
        
    elif i >= 70000 and i< 100000:
        new_m_rank.append(9)
    elif i >= 35000 and i< 70000:
        new_m_rank.append(8)
    elif i >= 10000 and i< 35000:
        new_m_rank.append(7)
        
        # Assigning 5 scores for 10k to 1k bucket (As group weightage is 49.30%)
        
    elif i >= (10000-1800) and i< 10000:
        new_m_rank.append(6)
    elif i >= (10000-3600) and i< (10000-1800):
        new_m_rank.append(5)
    elif i >= (10000-5400) and i< (10000-3600):
        new_m_rank.append(4)
    elif i >= (10000-7200) and i< (10000-5400):
        new_m_rank.append(3)
    elif i >= (10000-9000) and i< (10000-7200):
        new_m_rank.append(2)
    
    # Assigning 1 scores for 10k to 1k bucket (As group weightage is 14.41%)
    
    elif i< 1000:
        new_m_rank.append(1)
    else:
        new_m_rank.append('Invalid Monetary')
    
rfm_df['new_m_rank'] = new_m_rank

# =============================================================================
# Giving this time 50% weightage to monetary, 30% to frequency and 20% to recency in final scores
# as amount spent plays a vital role than frequency comes at 2nd place and recency comes last
# =============================================================================

r_weightaged20 = []
for rank  in rfm_df['new_r_rank']:
    r_weightaged20.append(rank*.2)

f_weightaged30 = []
for rank  in rfm_df['new_f_rank']:
    f_weightaged30.append(rank*.3)
    
m_weightaged50 = []  
for rank  in rfm_df['new_m_rank']:
    m_weightaged50.append(rank*.5)
    
rfm_df['r_weightaged20'] = r_weightaged20
rfm_df['f_weightaged30'] = f_weightaged30
rfm_df['m_weightaged50'] = m_weightaged50

rfm_df['new_final_ranking'] = round(rfm_df[['r_weightaged20', 'f_weightaged30', 'm_weightaged50']].sum(axis=1))
# =============================================================================

labels = rfm_df.new_final_ranking.value_counts().index.to_list()
plt.pie(rfm_df.new_final_ranking.value_counts(), labels=labels, autopct='%.0f%%', shadow=True, counterclock=False)   

plt.hist(rfm_df.new_final_ranking)
plt.xlabel('cluster_ID')
plt.ylabel('No. of Retailers')
plt.show()

# =============================================================================

# Important insights

# 50% of the retailers belong to cluster number 9, 4 and 5 (A total of 51%)
# 36% of the retailers belong to cluster 8,9,10 and can be considered premium and credit worthy. These retailer have good score in all 3 categories, i.e. Recency, frequency and monetary
# 6% of the retailers belong to cluster 1 and 2. These set of retailers are not frequent buyers and have spend least in their life time, hence, are credit risky
# Retailers from cluster 4,5 contribute to 1/3 of the total pupulation. These sit on an average point in terms of giving credit
# =============================================================================
# taking final features for predictive model building from rfm_df

df2 = rfm_df.iloc[:,[0,1,2,3,-1]]     
# =============================================================================
# 

# =============================================================================
# predictive modelling (supervised learning)
#
# k-Nearest Neighbors
# 
# Decision Trees
# 
# Naive Bayes
# 
# Random Forest.
# 
# Gradient Boosting
# 
# Extreme Gradient Boosting
# =============================================================================

# creating dataset

# train test split
from sklearn.model_selection import train_test_split


train_x, test_x, train_y, test_y = train_test_split(df2.iloc[:,1:-1], df2['new_final_ranking'], test_size=.2, random_state=1)


#================================================================

# knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_x, train_y)
knn_pred_test = knn.predict(test_x)
knn_accu_test = accuracy_score(test_y, knn_pred_test)

knn_pred_train = knn.predict(train_x)
knn_accu_train = accuracy_score(train_y, knn_pred_train)
knn_accu_test, knn_accu_train

accuracy_dict = {}
accuracy_dict['knn_test', 'knn_train'] = knn_accu_test, knn_accu_train
#==============================================================================

# Desicion Tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=4,
                             splitter='best')
dtc.fit(train_x, train_y)

dtc_pred_test = dtc.predict(test_x)
dtc_accu_test = accuracy_score(test_y, dtc_pred_test)

dtc_pred_train = dtc.predict(train_x)
dtc_accu_train = accuracy_score(train_y, dtc_pred_train)

dtc_accu_test, dtc_accu_train

accuracy_dict['dtc_test', 'dtc_train'] = dtc_accu_test, dtc_accu_train
#==============================================================================

# Naive Bayes

from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB()
gnb.fit(train_x, train_y)

gnb_pred_test = gnb.predict(test_x)
gnb_accu_test = accuracy_score(test_y, gnb_pred_test)

gnb_pred_train = gnb.predict(train_x)
gnb_accu_train = accuracy_score(train_y, gnb_pred_train)

gnb_accu_test, gnb_accu_train

accuracy_dict['NB_test', 'NB_train'] = gnb_accu_test, gnb_accu_train

#==============================================================================

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# =============================================================================
# from sklearn.model_selection import GridSearchCV
# 
# rf = RandomForestClassifier()
# params = {'n_estimators':[50, 100, 200, 500, 1000, 5000], 
#           'criterion': ['gini', 'entropy'],
#           'min_samples_split': [2, 4, 6],
#           }
# 
# gcv = GridSearchCV(rf, params)
# gcv.fit(train_x, train_y)
# 
# gcv.best_score_
# gcv.best_params_
# =============================================================================

rf = RandomForestClassifier(criterion ='entropy', min_samples_split= 2, 
                            n_estimators= 20, n_jobs=-1, random_state=1)
rf.fit(train_x, train_y)

rf_pred_test = rf.predict(test_x)
rf_accu_test = accuracy_score(test_y, rf_pred_test)

rf_pred_train = rf.predict(train_x)
rf_accu_train = accuracy_score(train_y, rf_pred_train)

rf_accu_test, rf_accu_train

accuracy_dict['RF_test', 'RF_train'] = rf_accu_test, rf_accu_train

#=============================================================================

# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=30,                                 
                                 max_depth=1, random_state=1)
gbc.fit(train_x, train_y)

gbc_pred_test = gbc.predict(test_x)
gbc_accu_test = accuracy_score(test_y, gbc_pred_test)

gbc_pred_train = gbc.predict(train_x)
gbc_accu_train = accuracy_score(train_y, gbc_pred_train)

gbc_accu_test, gbc_accu_train

accuracy_dict['GBC_test','GBC_train'] = gbc_accu_test, gbc_accu_train

#==============================================================================

# Xtream Gradient Boosting

# pip install xgboost==0.90

import xgboost
#max_depths = 3, n_estimators = 5000, 
                                #learning_rate = .2, n_jobs = -1
# n_estimators=50, 
#                                min_samples_split=3, min_samples_leaf=2, 
 #                                max_depth=7, random_state=1
xgb_clf = xgboost.XGBClassifier(n_estimators=20, max_depth=1, 
                                n_jobs=-1, random_state=1)
xgb_clf.fit(train_x, train_y)

xgb_pred_test = xgb_clf.predict(test_x)
xgb_accu_test = accuracy_score(test_y, xgb_pred_test)

xgb_pred_train = xgb_clf.predict(train_x)
xgb_accu_train = accuracy_score(train_y, xgb_pred_train)

xgb_accu_test, xgb_accu_train

accuracy_dict['XGB_test','XGB_train'] = xgb_accu_test, xgb_accu_train

#==============================================================================

# SVM Classifier

from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear')
svm_model_linear.fit(train_x, train_y)
svm_pred_test = svm_model_linear.predict(test_x)
svm_accu_test = accuracy_score(test_y, svm_pred_test)

svm_pred_train = svm_model_linear.predict(train_x)
svm_accu_train = accuracy_score(train_y, svm_pred_train)

svm_accu_test, svm_accu_train

accuracy_dict['svm_linear_test', 'svm_linear_train'] = svm_accu_test, svm_accu_train
# =============================================================================

# Multiclass classification using Deep Learning Algorithms

# ANN

#pip install --upgrade h5py

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_x_norm = scaler.fit_transform(train_x)
test_x_norm = scaler.fit_transform(test_x)

ann = Sequential()
ann.add(Dense(33,input_dim =3,activation="ReLU", kernel_initializer='random_normal'))
ann.add(Dense(11,activation="softmax"))
ann.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

history = ann.fit(train_x_norm, train_y, validation_data=(test_x_norm, test_y), epochs=100, verbose=1)

train_loss, ann_test_acc = ann.evaluate(test_x, test_y, verbose=0)
test_loss, ann_train_acc = ann.evaluate(train_x, train_y, verbose=0)

accuracy_dict['ANN_test', 'ANN_train'] = ann_test_acc, ann_train_acc

# =============================================================================
# Finding best parameters for XGBoost to improve test score

# from sklearn.model_selection import GridSearchCV

# =============================================================================
# xgb_model = xgboost.XGBClassifier()
# optimization_dict = {'max_depth': [3,4,5,6,7],
#                      'min_child_weight': [1, 5, 10],
#                      'gamma': [0.5, 1, 1.5, 2, 5],
#                      'colsample_bytree': [0.6, 0.8, 1.0],
#                      'n_estimators': [50,100,200, 500],
#                      'learning_rate':[0.1, 0.2, 0.3]}
#  
# gcv = GridSearchCV(xgb_model, optimization_dict, 
#                      scoring='accuracy', verbose=1)
# 
# gcv.fit(train_x, train_y)
# 
# print(gcv.best_score_) #8956302521008404
# print(gcv.best_params_) #{'colsample_bytree': 0.8, 'gamma': 1, 
# 'learning_rate': 0.3, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 50}
# 
# 
# =============================================================================

#splitter='random', random_state=42
# Ran above grid search in Google Colab and came out with best estimator as mentioned in below model

dt_pre_pru = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=3,
                                    random_state=2)
dt_pre_pru.fit(train_x, train_y)

dt_train_pred = dt_pre_pru.predict(train_x)
dt_test_pred = dt_pre_pru.predict(test_x)

dt_prepru_accu_test = accuracy_score(train_y, dt_train_pred)
dt_prepru_accu_train = accuracy_score(test_y, dt_test_pred)

dt_prepru_accu_test # 0.95% accurate on test data
dt_prepru_accu_train # 0.86% accurate on train data

accuracy_dict['DT_Pre_Prunning_Test', 'DT_Pre_Prunning_Train'] = dt_prepru_accu_test, dt_prepru_accu_train

#==============================================================================

# DT Post Prunning

dt_clf = DecisionTreeClassifier()
path = dt_clf.cost_complexity_pruning_path(train_x, train_y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


# For each alpha we will append our model to a list
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(train_x, train_y)
    clfs.append(clf)
    
# We will remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node.

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.show()

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(train_x)
    y_test_pred = c.predict(test_x)
    train_acc.append(accuracy_score(y_train_pred,train_y))
    test_acc.append(accuracy_score(y_test_pred,test_y))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


post_pru = DecisionTreeClassifier(criterion='entropy', ccp_alpha=.02, random_state=42)
post_pru.fit(train_x, train_y)
post_pru_train_pred = post_pru.predict(train_x)
post_pru_test_pred = post_pru.predict(test_x)
post_pru_train_accu = accuracy_score(train_y, post_pru_train_pred)
post_pru_test_accu = accuracy_score(test_y, post_pru_test_pred)

accuracy_dict['DT_post_pru_test', 'DT_post_pru_train'] = post_pru_test_accu, post_pru_train_accu 

# =============================================================================

# Final model is svm_model_linear with test, train accuracy as 93.6, 86 respectively with low bias and low variance.


# =============================================================================

# Creating pickle file

import pickle
pickle.dump(svm_model_linear, open("model.pkl", "wb"))

# =============================================================================





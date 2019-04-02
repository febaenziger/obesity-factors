# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:19:20 2019

@author: Fiona Baenziger
"""

import csv, copy
import numpy as np
import pandas as pd
import time
import math
import sys
from operator import itemgetter

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.externals import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale


#Handle warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


########## GLOBAL PARAMETERS ###########

bmi_idx=1                                           #   Index of BMI variable
income_idx=1                                        #   Index of Income Variable

cross_val=1                                         #   Control Switch for CV   
technique=5                                         #   1 - Decision Tree, 2 - Ada Boost, 3 - Random Forest, 4 - Bagging, 5 - Gradient Boosting, 6 - MLP, 7 - SVR                                                                                                                                               
norm_target=0                                       #   Normalize target switch
norm_features=0                                     #   Normalize features switch
feat_select=1                                       #   Control Switch for Feature Selection  
fs_type=2                                           #   1 - Stepwise Recursion, Backwards ; 2 - Wrapper Select via Model ; 3 - Univariate Feature Selection - Chi-squared ; 4 - Full-blown Wrapper Select                                                                         
feat_start=2                                        #   Start column of features
k_cnt=5                                             #   Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3

#Set global model parameters
rand_st=1                                           #   Set Random State variable for randomizing splits on runs


#Recursive Function for searching thru feature space
def feat_space_search(arr, curr_idx):
    '''Setup currently as exhuastive search, but could be changed to use
       greedy search, random search, genetic algorithms, etc. ... also
       no regularization, so probably selects more features than necessary'''
    global roll_idx, combo_ctr, best_score, sel_idx
    
    if curr_idx==feat_cnt:
        #If end of feature array, roll thru combinations
        roll_idx=roll_idx+1
        print ("Combos Searched so far:", combo_ctr, "Current Best Score:", best_score)
        for i in range(roll_idx, len(arr)):
            arr[i]=0
        if roll_idx<feat_cnt-1:
            feat_space_search(arr, roll_idx+1)                                                                      #Recurse till end of rolls
        
    else:
        #Else setup next feature combination and calc performance
        arr[curr_idx]=1
        data=data_np#_wrap                                                                                          #Temp array to hold data
        temp_del=[i for i in range(len(arr)) if arr[i]==0]                                                          #Pick out features not in this combo, and remove
        data = np.delete(data, temp_del, axis=1)
        data_train, data_test, target_train, target_test = train_test_split(data, target_np, test_size=0.35)                

        scorers = {'Neg_MSE': 'neg_mean_squared_error', 'expl_var': 'explained_variance'}
        scores = cross_validate(rgr, data, target_np, scoring=scorers, cv=5)    
        score = np.asarray([math.sqrt(-x) for x in scores['test_Neg_MSE']]).mean()                              #RMSE
        print('Random Forest RMSE:', curr_idx, feat_arr, len(data[0]), score)
        if score<best_score:                                                                                    #Compare performance and update sel_idx and best_score, if needed
            best_score=score
            sel_idx=copy.deepcopy(arr) 

        #move to next feature index and recurse
        combo_ctr+=1  
        curr_idx+=1
        feat_space_search(arr, curr_idx)                                                                            #Recurse till end of iteration for roll
        
########## LOAD THE DATA ##########

file1= csv.reader(open('ehresp.csv'), delimiter=',', quotechar='"')

#Read Header Line
header=next(file1)            

#Read data
data=[]
target=[]
for row in file1:
    
    #Load Target
    if row[bmi_idx]=='' or float(row[bmi_idx]) <= 0: #If BMI is empty or <= 0, SKIP                   
        continue
#    elif row[income_idx]=='' or int(row[income_idx]) <= 0: #If INCOME is empty or <= 0, SKIP
#       continue
    else:
        target.append(float(row[bmi_idx])) #If pre-binned class, change float to int

    #Load row into temp array, cast columns  
    temp=[]
                 
    for j in range(feat_start,len(header)):
        if row[j]=='':
            temp.append(float())
        else:
            temp.append(float(row[j]))
        #    if float(row[j]) < 0:
         #       temp.append(float())
          #  else:
           #     temp.append(float(row[j]))

    #Load temp into Data array
    data.append(temp)

########## MANUALLY PREPROCESSED DATA ##########

# Create bins for Income
#b = pd.IntervalIndex.from_tuples([(.5, 1.5), (1.75, 5.25)])
#pd.cut(data_np[income_idx], b, labels=["Below 130%", "Above 130%"])
    
# Create bins for BMI
#bins = pd.IntervalIndex.from_tuples([(0, 18.4), (18.5, 24.9), (25, 29.9), (30, 100)])
#pd.cut(target_np, bins, labels=["Underweight", "Normal Weight", "Overweight", "Obese"], retbins=True)

# Do the undersampling thing
minorityTotal = 0
majority_indices = []
minority_indices = []
count = 0
for row in data:

    if int(row[income_idx]) >= 2 and int(row[income_idx]) <= 5:
        minorityTotal+=1
        minority_indices.append(count)
        
    else:
        majority_indices.append(count)
    
    count+=1
        
random_indices = np.random.choice(majority_indices, minorityTotal, replace=False)
under_sample_indices = np.concatenate([minority_indices, random_indices])

d=[]
t=[]
for l in random_indices:
    d.append(data[l])
    t.append(target[l])
    

data_np=np.asarray(d)
target_np=np.asarray(t)

##Test Print
print(header)
print(len(target),len(data))
print(len(target_np),len(data_np))
print(target[0], data[0])
print('\n')

print()

########## NORMALIZING THE DATA ##########

if norm_target==1:
    #Target normalization for continuous values
    target_np=scale(target_np)

if norm_features==1:
    #Feature normalization for continuous values
    data_np=scale(data_np)

########## FEATURE SELECTION ##########

#Feature Selection
if feat_select==1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''
    
    print('--FEATURE SELECTION ON--', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        rgr = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=3, criterion='mse', random_state=rand_st)
        sel = RFE(rgr, n_features_to_select=k_cnt, step=.1)
        print('Stepwise Recursive Backwards - Random Forest: ')
            
        fit_mod=sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx=fit_mod.get_support()      

    if fs_type==2:
        #Wrapper Select via model
        rgr= SVR(kernel='linear', gamma=0.1, C=1.0)
        sel = SelectFromModel(rgr, prefit=False, threshold='mean', max_features=5)
        print ('Wrapper Select: ')
            
        fit_mod=sel.fit(data_np, target_np)    
        sel_idx=fit_mod.get_support()

    if fs_type==3:
        #Univariate Feature Selection - Mutual Info Regression
        sel=SelectKBest(mutual_info_regression, k=k_cnt)
        fit_mod=sel.fit(data_np, target_np)
        print ('Univariate Feature Selection - Mutual Info: ')
        sel_idx=fit_mod.get_support()

        #Print ranked variables out sorted
        temp=[]
        scores=fit_mod.scores_
        for i in range(feat_start, len(header)):            
            temp.append([header[i], float(scores[i-feat_start])])

        print('Ranked Features')
        temp_sort=sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')

    if fs_type==4:
        #Full-blown Wrapper Select (from any kind of ML model)        

        start_ts=time.time()
        sel_idx=[]
        best_score=sys.float_info.max
        feat_cnt=len(data_np[0])
        #Create Wrapper model
        rgr= SVR(kernel='linear', gamma=0.1, C=1.0)                    #This could be any kind of regressor model         
        
        #Loop thru feature sets
        roll_idx=0
        combo_ctr=0
        feat_arr=[0 for col in range(feat_cnt)]                                         #Initialize feature array
        for idx in range(feat_cnt):
            roll_idx=idx
            feat_space_search(feat_arr, idx)                                           #Recurse
            feat_arr=[0 for col in range(feat_cnt)]                                     #Reset feature array after each iteration
        
        print('# of Feature Combos Tested:', combo_ctr)
        print(best_score, sel_idx, len(data_np[0]))
        print("Wrapper Feat Sel Runtime:", time.time()-start_ts)

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np[0]), len(temp))
    print('\n')
            
                
    ##3) Filter selected columns from original dataset #########
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index)
    
    


########## Training the Model ##########

print('-- Machine Learning Model Output --', '\n')

#Test/Train split
data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.30)
                 
#### Regressors ####
if cross_val==0:
    #SciKit Bagging Regressor - Cross Val
    
    if technique == 1:
        # Deciscion Tree
        start_ts=time.time()
        rgr = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=rand_st)
        bag = BaggingRegressor(rgr, max_samples=0.6, random_state=rand_st) 
        bag.fit(data_train, target_train)
        
    elif technique == 2:
        # Ada Boost
        start_ts=time.time()
        rgr= AdaBoostRegressor(n_estimators=100, base_estimator=None, loss='linear', learning_rate=0.5, random_state=rand_st)
        bag = BaggingRegressor(rgr, max_samples=0.6, random_state=rand_st) 
        bag.fit(data_train, target_train)                                                                                                
    
    elif technique == 3:
        # Random Forest
        start_ts=time.time()
        rgr = RandomForestRegressor(n_estimators=100, max_features=.33, max_depth=None, min_samples_split=3, random_state=rand_st)  
        bag = BaggingRegressor(rgr, max_samples=0.6, random_state=rand_st) 
        bag.fit(data_train, target_train)

    elif technique == 5:
        # Gradient Boosting
        start_ts=time.time()
        rgr= GradientBoostingRegressor(n_estimators=100, loss='ls', max_depth=None, min_samples_split=3, random_state=rand_st)
        bag = BaggingRegressor(rgr, max_samples=0.6, random_state=rand_st) 
        bag.fit(data_train, target_train)
    
    elif technique == 6:
        # MLP
        start_ts=time.time()
        rgr= MLPRegressor(activation='logistic', solver='lbfgs', alpha=0.0001, hidden_layer_sizes=10, random_state=rand_st)
        bag = BaggingRegressor(rgr, max_samples=0.6, random_state=rand_st) 
        bag.fit(data_train, target_train)
    
    elif technique == 7:
        # SVR
        start_ts=time.time()
        rgr= SVR(kernel='rbf', gamma=0.1, C=1.0)
        bag = BaggingRegressor(rgr, max_samples=0.6, random_state=rand_st) 
        bag.fit(data_train, target_train) 
        
    scores_RMSE = math.sqrt(metrics.mean_squared_error(target_test, bag.predict(data_test)))
    print('Decision Tree RMSE:', scores_RMSE)
    scores_Expl_Var = metrics.explained_variance_score(target_test, bag.predict(data_test))
    print('Decision Tree Expl Var:', scores_Expl_Var)                                                                                               
    print("CV Runtime:", time.time()-start_ts)
    
#### Cross-Val Regressors ####
if cross_val==1:
    #Setup Crossval regression scorers
    scorers = {'Neg_MSE': 'neg_mean_squared_error', 'expl_var': 'explained_variance'} 
    
    if technique == 1:
        # Deciscion Tree
        start_ts=time.time()
        rgr = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=rand_st)
        scores = cross_validate(rgr, data_np, target_np, scoring=scorers, cv=7)
        
    elif technique == 2:
        # Ada Boost
        start_ts=time.time()
        rgr= AdaBoostRegressor(n_estimators=100, base_estimator=None, loss='linear', learning_rate=0.5, random_state=rand_st)
        scores = cross_validate(rgr, data_np, target_np, scoring=scorers, cv=5)                                                                                                 
    
    elif technique == 3:
        # Random Forest
        start_ts=time.time()
        rgr = RandomForestRegressor(n_estimators=100, max_features=.33, max_depth=None, min_samples_split=3, random_state=rand_st)  
        scores = cross_validate(rgr, data_np, target_np, scoring=scorers, cv=7)

    elif technique == 5:
        # Gradient Boosting
        start_ts=time.time()
        rgr= GradientBoostingRegressor(n_estimators=80, loss='ls', max_depth=2, min_samples_split=3, random_state=rand_st)
        scores=cross_validate(rgr, data_np, target_np, scoring=scorers, cv=5) 
    
    elif technique == 6:
        # MLP
        start_ts=time.time()
        rgr= MLPRegressor(activation='logistic', solver='lbfgs', alpha=0.0001, hidden_layer_sizes=10, random_state=rand_st)
        scores=cross_validate(rgr, data_np, target_np, scoring=scorers, cv=5)
    
    elif technique == 7:
        # SVR
        start_ts=time.time()
        rgr= SVR(kernel='rbf', gamma=0.1, C=1.0)
        scores= cross_validate(rgr, data_np, target_np, scoring=scorers, cv=7)                                                                                       

    scores_RMSE = np.asarray([math.sqrt(-x) for x in scores['test_Neg_MSE']])                                       #Turns negative MSE scores into RMSE
    scores_Expl_Var = scores['test_expl_var']
    print("MODEL RMSE:: %0.2f (+/- %0.2f)" % ((scores_RMSE.mean()), (scores_RMSE.std() * 2)))
    print("MODEL Expl Var: %0.2f (+/- %0.2f)" % ((scores_Expl_Var.mean()), (scores_Expl_Var.std() * 2)))
    print("CV Runtime:", time.time()-start_ts)

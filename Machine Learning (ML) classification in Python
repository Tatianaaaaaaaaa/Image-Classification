# -*- coding: utf-8 -*-
# =============================================================================
# This file was created to run Machine Learning (ML) classification by 
# Support Vector Machine (SVM), Random Forest (RF),Artificial Netural Network (ANN),
# K-Nearest Neighbors (KNN), and Ensamble analysis (EA)
# Updates: 10/30/2021
# =============================================================================

# =============================================================================
# 1.Import required libraries and packages
# =============================================================================
# In[] 
# Import required libraries and packages 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import os #import operating system
cwd = os.getcwd() # Set up the current working directory
print(cwd) # current working directory

# =============================================================================
# 2.Load in required training data in csv file 
# =============================================================================
# In[] Load in required training data in csv file format 
raw = pd.read_csv('Training_WestApopka.csv') # read file
raw_clean = raw.copy() # make a copy of raw data to prevent changing the original file

# =============================================================================
# 3. Data preprocessing for training
# =============================================================================
# In[] 
# set up dependent value (classification) and independent value (attributions of satellite imagery)
y = raw_clean['Class'] # set dependent value column 
X = raw_clean.drop(['Class'],axis = 1) # set independent value column

# Data scaling
# Standardize the training dataset in a specific range
from sklearn.preprocessing import StandardScaler
Stand = StandardScaler()
X_train = Stand.fit_transform(X)
# =============================================================================
# 4.Call machine learning (ML) models for training 
# =============================================================================
# In[] 
# Fitting RF to the training dataset
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state = 3)
classifier_rf.fit(X_train,y)

# Fitting Kernel SVM to the training dataset
from sklearn.svm import SVC
classifier_svm = SVC(C=3033,gamma= 0.005)
classifier_svm.fit(X_train,y)

# Fitting ANN to the training dataset
from sklearn.neural_network import MLPClassifier
classifier_MLP = MLPClassifier(hidden_layer_sizes = 300, max_iter=2000)
classifier_MLP.fit(X_train, y)

# Fitting KNN to the training dataset
# feature selection --PCA
from sklearn.decomposition import PCA
# make an instance 
pca = PCA(0.95)
pca.fit(X_train)
X_train_knn = pca.transform(X_train)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(X_train_knn, y)
# =============================================================================
# 5.Generate error matrix to evaluate the performance of models
# =============================================================================
# In[]
def y_pred(classifier,X_train,y): # predictions from trained ML model
    from sklearn.model_selection import cross_val_predict  
    y_pred = cross_val_predict(classifier, X_train, y, cv =10)
    return y_pred

class Error_matrix(): 
    def __init__(self,classifier,y_pred,y):
        self.class_name= classifier.classes_
        from sklearn.metrics import confusion_matrix
        self.cm = confusion_matrix(y, y_pred) 
        self.cm_df = pd.DataFrame(self.cm,  
                index = self.class_name,
                columns = self.class_name)
        
        self.col_sum = list(self.cm.sum(axis = 1))# a list of sum across column
        self.col_sum_df= pd.DataFrame (self.col_sum,
                                       columns = ['Column Total'],
                                       index = self.class_name)
        self.row_sum = list(self.cm.sum(axis = 0)) # a list of sum across row
        self.row_sum_df = pd.DataFrame([self.row_sum],
                                       columns = self.class_name,
                                       index = ['Row total'])
        self.diag= np.diag(self.cm) # the diag of confusion matrix

        self.ua = self.diag/self.col_sum # user's accuracy and producer's accuracy
        self.ua = [np.round(x,3) for x in self.ua]
        self.ua_df = pd.DataFrame (self.ua,columns=["UA"],index =self.class_name)
        self.pa = self.diag/self.row_sum
        self.pa = [np.round(x,3) for x in self.pa]
        self.pa_df = pd.DataFrame ([self.pa],columns = self.class_name,index = ['PA'])

        from sklearn.metrics import cohen_kappa_score # calculate kappa value for confusion matrix 
        self.kappa_value = cohen_kappa_score(y, y_pred) 
        self.kappa_value_df = pd.DataFrame(self.kappa_value,
                                           index=['Kappa value'],
                                           columns =['Kappa value'])
    def add_to_cm(self): # add producer's accuracy, user's accuracy and kappa value to confusion matrix  
        self.cm_final = self.cm_df
        self.cm_final['Column total'] = self.col_sum_df
        self.cm_final['UA'] = self.ua_df
        self.cm_final = self.cm_final.append(self.row_sum_df)
        self.cm_final = self.cm_final.append(self.pa_df)
        self.cm_final = self.cm_final.append(self.kappa_value_df) 
        return self.cm_final
    def get_ua(self):
        return self.ua
    
# main function for error matrix
def error_matrix_main(classifier,X_train,y):
    pred_y = y_pred(classifier,X_train,y)
    cm = Error_matrix(classifier,pred_y,y)
    cm_final = cm.add_to_cm()
    return cm_final   
# In[]
# call error_matrix_main（） function to export the error matrix table to the current working directory
cm_final_rf = error_matrix_main(classifier_rf,X_train,y)    
cm_final_rf_csv = cm_final_rf.to_csv('classifier_rf_cm_test.csv')
    
cm_final_svm = error_matrix_main(classifier_svm,X_train,y)    
cm_final_svm_csv = cm_final_svm.to_csv('classifier_svm_cm_test.csv')

cm_final_ann = error_matrix_main(classifier_MLP,X_train,y)    
cm_final_ann_csv = cm_final_ann.to_csv('classifier_ann_cm_test.csv') 

cm_final_knn = error_matrix_main(classifier_knn,X_train_knn,y)    
cm_final_knn_csv = cm_final_knn.to_csv('classifier_knn_cm_test.csv')       
# =============================================================================
# 6.Perform ensemble analysis of different classifiers, such as SVM, RF and ANN
# =============================================================================
# In[] ensemble analysis of different classifiers

# read predctions from RF, SVM, ANN by calling y_pred () function   
y_pred_rf = y_pred(classifier_rf,X_train,y)
y_pred_svm = y_pred(classifier_svm,X_train,y)
y_pred_ann = y_pred(classifier_MLP,X_train,y) 

# read classification code and user's accuracy (ua)
def read_code(classifier,y_pred,y):
    pa_code = list(classifier.classes_)
    cm = Error_matrix(classifier,y_pred,y)
    pa_ua = cm.get_ua()
    return pa_code,pa_ua

# sign producer's accuracy to each predictions 
def sign_pa(classifier,y_pred,y): 
    y_pred_PA = {} # empty list for producer's accuracy
    pa_code,pa_ua = read_code(classifier,y_pred,y)
    for i in range(len(y_pred)):
        if y_pred[i] in pa_code:
            y_pred_PA[i] = pa_ua[y_pred[i]-1]
    return y_pred_PA

y_pred_rf_PA = sign_pa(classifier_rf,y_pred_rf,y) # a list inclduing each prediction and its producer's accuracy by classifier RF
y_pred_svm_PA = sign_pa(classifier_svm,y_pred_svm,y) # a list inclduing each prediction and its producer's accuracy by classifier SVM
y_pred_ann_PA = sign_pa(classifier_MLP,y_pred_ann,y) # a list inclduing each prediction and its producer's accuracy by classifier ANN

# comparing list
def compare_max (y_pred_ann,y_pred_svm,y_pred_rf,
                       y_pred_svm_PA,y_pred_rf_PA,y_pred_ann_PA):
    final = {} # new list for ensemble analysis results
    for j in range(len(y_pred_ann)):
        if y_pred_svm[j] == y_pred_rf[j]:
            final[j] = y_pred_svm[j]
        elif y_pred_svm[j] == y_pred_ann[j]:
            final[j] = y_pred_svm[j]
        elif y_pred_rf[j] == y_pred_ann[j]:
            final[j] = y_pred_rf[j]   
        else:              
             # compare y_pred_svm_pa,y_pred_rf_pa,y_pred_ann_pa 
                if y_pred_svm_PA[j] >= y_pred_rf_PA[j]:
                    if y_pred_rf_PA[j] >= y_pred_ann_PA[j]:
                        final[j] = y_pred_svm[j] #a>=b>=c
                    else:
                        if y_pred_svm_PA[j] > y_pred_ann_PA[j]:
                            final[j] = y_pred_svm[j] #a>c>b
                        else:
                            final[j] = y_pred_ann[j] #c>a>b
                elif y_pred_svm_PA[j] < y_pred_rf_PA[j]:
                    if y_pred_rf_PA[j] < y_pred_ann_PA[j]:
                        final[j] = y_pred_ann[j] #c>b>a
                    else:
                        if y_pred_ann_PA[j] > y_pred_svm_PA[j]:
                            final[j] = y_pred_rf[j] #b>c>a
                        else:
                            final[j] = y_pred_rf[j] #b>a>c
    # predicted value from the training data based on ensemble analysis
    y_pred_em = list(final.values())
    return y_pred_em

# predicted y from ensemble analysis（training）
pred_y_em = compare_max (y_pred_rf,y_pred_svm,y_pred_ann,
             y_pred_rf_PA,y_pred_svm_PA,y_pred_ann_PA)
# In[] Error_matrix for ensemble analysis    
#  main function for error matrix_em
def error_matrix_em_main(y_pred_em,y):
    cm_em = Error_matrix(classifier_rf,y_pred_em,y)
    cm_em_final = cm_em.add_to_cm()
    return cm_em_final

error_matrix_em = error_matrix_em_main(pred_y_em,y)
error_matrix_em_csv = error_matrix_em.to_csv('classifier_em_cm_test.csv') # export confusion matrix to excel file  

# =============================================================================
# 7.Conduct McNemar test to compare the classifiers 
# =============================================================================
# In[]
# Conduct McNemar Test 
import numpy as np
from mlxtend.evaluate import mcnemar_table
from statsmodels.stats.contingency_tables import mcnemar 

# The correct target (class) labels
y_target = np.array(y)

# read pred_y for each classifier by function 
y_pred_rf = y_pred(classifier_rf,X_train,y)
y_pred_svm = y_pred(classifier_svm,X_train,y)
y_pred_ann = y_pred(classifier_MLP,X_train,y)
y_pred_knn = y_pred(classifier_knn,X_train_knn,y)

def mcnemar_test(y_target,y_model1,y_model2):
    tb = mcnemar_table(y_target=y_target, 
                   y_model1=y_model1, 
                   y_model2=y_model2)
    result = mcnemar(tb, exact= False)
    zvalue = result.statistic
    print('statistic=%.2f' % (result.statistic))
    return zvalue

# zvalue for different classifiers
zvalue_svm_rf = mcnemar_test(y_target,y_pred_svm,y_pred_rf)
zvalue_svm_ann = mcnemar_test(y_target,y_pred_svm,y_pred_ann)
zvalue_svm_knn = mcnemar_test(y_target,y_pred_svm,y_pred_knn)

zvalue_rf_ann = mcnemar_test(y_target,y_pred_rf,y_pred_ann)
zvalue_rf_knn = mcnemar_test(y_target,y_pred_rf,y_pred_knn)
zvalue_ann_knn = mcnemar_test(y_target,y_pred_ann,y_pred_knn)

# =============================================================================
# 8.Deploy ML models to make predictions for all segments (prepared for generating maps)
# =============================================================================
# In[] 
# read all segments
raw_all = pd.read_csv('InputForPrediction_WestApopka.csv') # read file

y_all = raw_all ['Class'] # set dependent value column 
X_all = raw_all.drop(['Class'],axis = 1) # drop column 'class'

# feature Scaling
from sklearn.preprocessing import StandardScaler
Stand = StandardScaler()
X_all_scale = Stand.fit_transform(X_all)

y_all_pre_rf = classifier_rf.predict(X_all) # make predictions for all segments based on normalized features
y_all_pre_svm = classifier_svm.predict(X_all_scale) # make predictions for all segments based on normalized features
y_all_pre_ann = classifier_MLP.predict(X_all_scale) # make predictions for all segments based on normalized features

from sklearn.decomposition import PCA 
# make an instance 
pca = PCA(.93)
pca.fit(X_all_scale)
X_all_knn = pca.transform(X_all_scale)

y_all_pre_knn = classifier_knn.predict(X_all_knn) # make predictions for all segments based on selected features

# In[]
# export final classification results to the csv file (prepared for making maps)  
y_all_pred_df_rf = pd.DataFrame (y_all_pre_rf,index = X_all.index,columns=["rf_pred"])    
y_all_pred_df_rf.to_csv('RF_predall.txt') 
  
y_all_pred_df_svm = pd.DataFrame (y_all_pre_svm,index = X_all.index,columns=["svm_pred"]) 
y_all_pred_df_svm.to_csv('SVM_predall.txt')  
          
y_all_pred_df_ann = pd.DataFrame (y_all_pre_ann,index = X_all.index,columns=["ann_pred"]) 
y_all_pred_df_ann.to_csv('ANN_predall.txt')

y_all_pred_df_knn= pd.DataFrame (y_all_pre_knn,index = X_all.index,columns=["knn_pred"])    
y_all_pred_df_knn.to_csv('KNN_predall.txt')

# =============================================================================
# 9.Generate ensemble predictions by combining ANN, SVM, and RF predictions
# =============================================================================
# In[] 
# call sign_pa() function to sign producer's accuracy to each predictions 
y_all_pred_rf_PA = sign_pa(classifier_rf,y_all_pre_rf,y) # for RF
y_all_pred_svm_PA = sign_pa(classifier_svm,y_all_pre_svm,y) # for SVM
y_all_pred_ann_PA = sign_pa(classifier_MLP,y_all_pre_ann,y) # for ANN
 
# call compare_max() function to generate predicted y from ensemble analysis
y_all_pred_em = compare_max (y_all_pre_rf,y_all_pre_svm,y_all_pre_ann,
             y_all_pred_rf_PA,y_all_pred_svm_PA,y_all_pred_ann_PA)

# export ensemble predictions to excel file 
y_all_pred_em_df = pd.DataFrame(y_all_pred_em, columns=['em_pred']) 
y_all_pred_em_df.to_csv('em_pred_all_test.txt')

# Game over
#############################################



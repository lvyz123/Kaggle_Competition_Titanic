#==============
# Module Import
#==============
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import learning_curve as lc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

#===========
# Data Input
#===========
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
total_data = train_data.append(test_data,ignore_index=True)
total_data_statistics = total_data.describe()
total_data_statistics.to_csv('data_statistics.csv')
#sys.exit(0)

#===========================
# A Map of Aggregated Titles
#===========================
Title_Dict = {
                "Capt":       "Officer",
                "Col":        "Officer",
                "Major":      "Officer",
                "Jonkheer":   "Royalty",
                "Don":        "Royalty",
                "Sir" :       "Royalty",
                "Dr":         "Officer",
                "Rev":        "Officer",
                "the Countess":"Royalty",
                "Dona":       "Royalty",
                "Mme":        "Mrs",
                "Mlle":       "Miss",
                "Ms":         "Mrs",
                "Mr" :        "Mr",
                "Mrs" :       "Mrs",
                "Miss" :      "Miss",
                "Master" :    "Master",
                }

#=================
# Data Preparation
#=================

# Passenger Class
pclass = pd.get_dummies(total_data['Pclass'],prefix='Pclass')

# Title
total_data['Title'] = total_data['Name'].map(lambda n: n.split(', ')[1].split('. ')[0])
total_data['Title'] = total_data['Title'].map(Title_Dict)
title = pd.get_dummies(total_data['Title'],prefix='Title')

# Sex
sex = total_data['Sex'].map(lambda s: 0 if s=='male' else 1)

# Family Size
total_data['FamilySize'] = 1 + total_data['SibSp'] + total_data['Parch']
familysize = pd.get_dummies(total_data['FamilySize'],prefix='FamilySize')

# Age
age = total_data['Age'].fillna(total_data['Age'].mean())
age = age.map(lambda a: (a-age.mean())/(age.max()-age.min()))

# Ticket Type
#total_data['Ticket'] = total_data['Ticket'].map(lambda t: t.split()[0].replace('/','').replace('.','').strip())
#total_data['Ticket'] = total_data['Ticket'].map(lambda t: t[:-1] if t[-1].isdigit() else t)
total_data['Ticket'] = total_data['Ticket'].map(lambda t: t.upper().split()[0].replace('/','').replace('.','').strip())
total_data['Ticket'] = total_data['Ticket'].map(lambda t: t[0:2]+str(len(t)) if t.isdigit() else t)
total_data['Ticket'] = total_data['Ticket'].map(lambda t: 'STONO2' if t.startswith('STON') else t)
ticket = pd.get_dummies(total_data['Ticket'],prefix='Ticket')

# Fare
fare = total_data['Fare'].fillna(total_data['Fare'].mean())
fare = fare.map(lambda f: (f-fare.mean())/(fare.max()-fare.min()))

# Cabin
total_data['Cabin'].fillna('U',inplace=True)
total_data['Cabin'] = total_data['Cabin'].map(lambda c:str(c).split()[0][0])
cabin = pd.get_dummies(total_data['Cabin'],prefix='Cabin')

#======================================
# Feature Extraction/Redunduncy Removal
#======================================
full_X = pd.concat([pclass, title, sex, age, familysize, ticket, fare], axis=1)
#full_X.to_csv('transformed_data.csv', index=False)
#sys.exit(0)

#============================================
# Train/Cross-validation/Test Data Definition
#============================================
train_cv_X = full_X[:891]
train_cv_y = total_data['Survived'][:891]
test_X = full_X[891:]
stra_k_fold = StratifiedKFold(n_splits=3)
train_X,cv_X,train_y,cv_y = train_test_split(train_cv_X,train_cv_y,test_size=0.2)
plt.figure()

#==============================================
# Support Vector Machine Modelling and Solution
#==============================================

# Grid Searching and Fitting
Cs = np.logspace(0,2,20)
prediction_model_svc = GridSearchCV(estimator=SVC(gamma='auto'),param_grid=dict(C=Cs),cv=stra_k_fold,iid=False,n_jobs=-1)
prediction_model_svc.fit(train_X, train_y)
C_best = prediction_model_svc.best_estimator_.C
print(prediction_model_svc.best_score_, C_best)

# Learning Curve
axes = plt.subplot(2,2,1)
lc.learning_curve_plot(SVC(gamma='auto',C=C_best),train_X,train_y,cv_X,cv_y)
axes.set_title(label='Support Vector Machine')
#sys.exit(0)

# Cross Validation
print(cross_val_score(prediction_model_svc, cv_X, cv_y, cv=stra_k_fold, n_jobs=-1))
prediction_y_svc_cv = prediction_model_svc.predict(cv_X)
prediction_y_svc_cv = prediction_y_svc_cv.round().astype('bool')*1
prediction_result_svc_cv = pd.DataFrame({'PredictionCV': prediction_y_svc_cv, 'ActualCV': cv_y})
prediction_result_svc_cv.to_csv('prediction_submission_svc_cv.csv', index=False)

# Test Data Prediction
prediction_y_svc = prediction_model_svc.predict(test_X)
prediction_y_svc = prediction_y_svc.round().astype('bool')*1
prediction_result_svc = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_svc})
prediction_result_svc.to_csv('prediction_submission_svc.csv', index=False)

#===========================================
# K-Nearest Neighbors Modelling and Solution
#===========================================
#prediction_model_knn = KNeighborsClassifier(n_neighbors=10)
#prediction_model_knn.fit(train_X, train_y)
#lc.learning_curve_plot(KNeighborsClassifier(n_neighbors=10),train_X,train_y,cv_X,cv_y)
#sys.exit(0)
#print(cross_val_score(prediction_model_knn, cv_X, cv_y, cv=stra_k_fold, n_jobs=-1))
#prediction_y_knn_cv = prediction_model_knn.predict(cv_X)
#prediction_y_knn_cv = prediction_y_knn_cv.round().astype('bool')*1
#prediction_result_knn_cv = pd.DataFrame({'PredictionCV': prediction_y_knn_cv, 'ActualCV': cv_y})
#prediction_result_knn_cv.to_csv('prediction_submission_knn_cv.csv', index=False)
#prediction_y_knn = prediction_model_knn.predict(test_X)
#prediction_y_knn = prediction_y_knn.round().astype('bool')*1
#prediction_result_knn = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_knn})
#prediction_result_knn.to_csv('prediction_submission_knn.csv', index=False)

#================================================
# Random Forest Classifier Modelling and Solution
#================================================

# Grid Searching and Fitting
n_values = np.arange(5,105,5)
prediction_model_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=dict(n_estimators=n_values),cv=stra_k_fold,iid=False,n_jobs=-1)
prediction_model_rfc.fit(train_X, train_y)
n_best = prediction_model_rfc.best_estimator_.n_estimators
print(prediction_model_rfc.best_score_, n_best)

# Learning Curve
axes = plt.subplot(2,2,2)
lc.learning_curve_plot(RandomForestClassifier(random_state=0,n_estimators=n_best),train_X,train_y,cv_X,cv_y)
axes.set_title(label='Random Forest')
#sys.exit(0)

# Cross Validation
print(cross_val_score(prediction_model_rfc, cv_X, cv_y, cv=stra_k_fold, n_jobs=-1))
prediction_y_rfc_cv = prediction_model_rfc.predict(cv_X)
prediction_y_rfc_cv = prediction_y_rfc_cv.round().astype('bool')*1
prediction_result_rfc_cv = pd.DataFrame({'PredictionCV': prediction_y_rfc_cv, 'ActualCV': cv_y})
prediction_result_rfc_cv.to_csv('prediction_submission_rfc_cv.csv', index=False)

# Test Data Prediction
prediction_y_rfc = prediction_model_rfc.predict(test_X)
prediction_y_rfc = prediction_y_rfc.round().astype('bool')*1
prediction_result_rfc = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_rfc})
prediction_result_rfc.to_csv('prediction_submission_rfc.csv', index=False)

#===========================================
# Logistic Regression Modelling and Solution
#===========================================

# Grid Searching and Fitting
prediction_model_lr = LogisticRegressionCV(Cs=80,solver='liblinear',cv=stra_k_fold,scoring=accuracy_score,n_jobs=-1)
prediction_model_lr.fit(train_X, train_y)
C_best = prediction_model_lr.best_estimator_.C
print(prediction_model_lr.best_score_, C_best)

# Learning Curve
axes = plt.subplot(2,2,3)
lc.learning_curve_plot(LogisticRegression(solver='liblinear',C=C_best),train_X,train_y,cv_X,cv_y)
axes.set_title(label='Logistic Regression')
#sys.exit(0)

# Cross Validation
print(cross_val_score(prediction_model_lr, cv_X, cv_y, cv=stra_k_fold, n_jobs=-1))
prediction_y_lr_cv = prediction_model_lr.predict(cv_X)
prediction_y_lr_cv = prediction_y_lr_cv.round().astype('bool')*1
prediction_result_lr_cv = pd.DataFrame({'PredictionCV': prediction_y_lr_cv, 'ActualCV': cv_y})
prediction_result_lr_cv.to_csv('prediction_submission_lr_cv.csv', index=False)

# Test Data Prediction
prediction_y_lr = prediction_model_lr.predict(test_X)
prediction_y_lr = prediction_y_lr.round().astype('bool')*1
prediction_result_lr = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_lr})
prediction_result_lr.to_csv('prediction_submission_lr.csv', index=False)

#==============================================
# Naive Bayes Classifier Modelling and Solution
#==============================================
#prediction_model_nby = GaussianNB(priors=None)
#lc.learning_curve_plot(prediction_model_nby,train_cv_X,train_cv_y)
#sys.exit(0)
#prediction_model_nby.fit(train_X, train_y)
#print(cross_val_score(prediction_model_nby, cv_X, cv_y, cv=stra_k_fold, n_jobs=-1))
#prediction_y_nby_cv = prediction_model_nby.predict(cv_X)
#prediction_y_nby_cv = prediction_y_nby_cv.round().astype('bool')*1
#prediction_result_nby_cv = pd.DataFrame({'PredictionCV': prediction_y_nby_cv, 'ActualCV': cv_y})
#prediction_result_nby_cv.to_csv('prediction_submission_nby_cv.csv', index=False)
#prediction_y_nby = prediction_model_nby.predict(test_X)
#prediction_y_nby = prediction_y_nby.round().astype('bool')*1
#prediction_result_nby = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_nby})
#prediction_result_nby.to_csv('prediction_submission_nby.csv', index=False)

#======================================
# Neural Network Modelling and Solution
#======================================

# Grid Searching and Fitting
layer_sizes = np.arange(10,200,10)
activations = np.array(['identity','logistic','tanh','relu'])
prediction_model_nnr = GridSearchCV(estimator=MLPClassifier(solver='lbfgs'),param_grid=dict(hidden_layer_sizes=layer_sizes,activation=activations),cv=stra_k_fold,iid=False,n_jobs=-1)
prediction_model_nnr.fit(train_X, train_y)
best_layer_size = prediction_model_nnr.best_estimator_.hidden_layer_sizes
best_activation = prediction_model_nnr.best_estimator_.activation
print(prediction_model_nnr.best_score_, best_layer_size, best_activation)

# Learning Curve
axes = plt.subplot(2,2,4)
lc.learning_curve_plot(MLPClassifier(solver='lbfgs',hidden_layer_sizes=best_layer_size,activation=best_activation),train_X,train_y,cv_X,cv_y)
axes.set_title(label='Neural Network')
#sys.exit(0)

# Cross Validation
print(cross_val_score(prediction_model_nnr, cv_X, cv_y, cv=stra_k_fold, n_jobs=-1))
prediction_y_nnr_cv = prediction_model_nnr.predict(cv_X)
prediction_y_nnr_cv = prediction_y_nnr_cv.round().astype('bool')*1
prediction_result_nnr_cv = pd.DataFrame({'PredictionCV': prediction_y_nnr_cv, 'ActualCV': cv_y})
prediction_result_nnr_cv.to_csv('prediction_submission_nnr_cv.csv', index=False)

# Test Data Prediction
prediction_y_nnr = prediction_model_nnr.predict(test_X)
prediction_y_nnr = prediction_y_nnr.round().astype('bool')*1
prediction_result_nnr = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_nnr})
prediction_result_nnr.to_csv('prediction_submission_nnr.csv', index=False)

plt.show()

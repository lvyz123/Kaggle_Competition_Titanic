import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

# Data Input
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
total_data = train_data.append(test_data,ignore_index=True)

# A Map of Aggregated Titles
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

# Data Preparation
pclass = pd.get_dummies(total_data['Pclass'],prefix='Pclass')
total_data['Title'] = total_data['Name'].map(lambda n: n.split(', ')[1].split('. ')[0])
total_data['Title'] = total_data['Title'].map(Title_Dict)
title = pd.get_dummies(total_data['Title'],prefix='Title')
sex = total_data['Sex'].map(lambda s: 0 if s=='male' else 1)
total_data['FamilySize'] = 1 + total_data['SibSp'] + total_data['Parch']
familysize = pd.get_dummies(total_data['FamilySize'],prefix='FamilySize')
age = train_data['Age'].fillna(total_data['Age'].mean(),inplace=True)
total_data['Ticket'] = total_data['Ticket'].map(lambda t: t.split()[0].replace('/','').replace('.','').strip())
total_data['Ticket'] = total_data['Ticket'].map(lambda t: t[:-1] if t[-1].isdigit() else t)
ticket = pd.get_dummies(total_data['Ticket'],prefix='Ticket')
fare = total_data['Fare'].fillna(total_data['Fare'].mean())
total_data['Cabin'] = total_data['Cabin'].map(lambda c:str(c).split()[0][0])
total_data['Cabin'].fillna('U')
cabin = pd.get_dummies(total_data['Cabin'],prefix='Cabin')

# Feature Extraction/Redunduncy Removal
train_prepca = pd.concat([pclass, title, sex, age, familysize, ticket, fare, cabin], axis=1)
pca = PCA()
pca.fit(train_prepca)
print(pca.explained_variance_)
pca.n_components = 7
train_postpca = pca.fit_transform(train_prepca)

# Train/Test Data Definition
train_X = train_postpca[:891]
train_y = total_data['Survived'][:891]
test_X = train_postpca[891:]
stra_k_fold = StratifiedKFold(n_splits=3)

# Support Vector Machine Modelling and Solution
Cs = np.logspace(0,2,20)
prediction_model_svc = GridSearchCV(estimator=SVC(gamma='auto'),param_grid=dict(C=Cs),cv=stra_k_fold,iid=False,n_jobs=-1)
prediction_model_svc.fit(train_X[:-50], train_y[:-50])
print(prediction_model_svc.best_score_, prediction_model_svc.best_estimator_.C)
print(cross_val_score(prediction_model_svc, train_X[-50:], train_y[-50:], cv=stra_k_fold, n_jobs=-1))
prediction_y_svc_cv = prediction_model_svc.predict(train_X[-50:])
prediction_y_svc_cv = prediction_y_svc_cv.round().astype('bool')*1
prediction_result_svc_cv = pd.DataFrame({'PredictionCV': prediction_y_svc_cv, 'ActualCV': train_y[-50:]})
prediction_result_svc_cv.to_csv('prediction_submission_svc_cv.csv', index=False)
#prediction_y_svc = prediction_model_svc.predict(test_X)
#prediction_y_svc = prediction_y_svc.round().astype('bool')*1
#prediction_result_svc = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_svc})
#prediction_result_svc.to_csv('prediction_submission_svc.csv', index=False)

# K-Nearest Neighbors Modelling and Solution
prediction_model_knn = KNeighborsClassifier(n_neighbors=10)
prediction_model_knn.fit(train_X[:-50], train_y[:-50])
print(cross_val_score(prediction_model_knn, train_X[-50:], train_y[-50:], cv=stra_k_fold, n_jobs=-1))
prediction_y_knn_cv = prediction_model_knn.predict(train_X[-50:])
prediction_y_knn_cv = prediction_y_knn_cv.round().astype('bool')*1
prediction_result_knn_cv = pd.DataFrame({'PredictionCV': prediction_y_knn_cv, 'ActualCV': train_y[-50:]})
prediction_result_knn_cv.to_csv('prediction_submission_knn_cv.csv', index=False)
#prediction_y_knn = prediction_model_knn.predict(test_X)
#prediction_y_knn = prediction_y_knn.round().astype('bool')*1
#prediction_result_knn = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_knn})
#prediction_result_knn.to_csv('prediction_submission_knn.csv', index=False)

# Random Forest Classifier Modelling and Solution
prediction_model_rfc = RandomForestClassifier(random_state=0, n_estimators=10)
prediction_model_rfc.fit(train_X[:-50], train_y[:-50])
print(cross_val_score(prediction_model_rfc, train_X[-50:], train_y[-50:], cv=stra_k_fold, n_jobs=-1))
prediction_y_rfc_cv = prediction_model_rfc.predict(train_X[-50:])
prediction_y_rfc_cv = prediction_y_rfc_cv.round().astype('bool')*1
prediction_result_rfc_cv = pd.DataFrame({'PredictionCV': prediction_y_rfc_cv, 'ActualCV': train_y[-50:]})
prediction_result_rfc_cv.to_csv('prediction_submission_rfc_cv.csv', index=False)
#prediction_y_rfc = prediction_model_rfc.predict(test_X)
#prediction_y_rfc = prediction_y_rfc.round().astype('bool')*1
#prediction_result_rfc = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_rfc})
#prediction_result_rfc.to_csv('prediction_submission_rfc.csv', index=False)

# Logistic Regression Modelling and Solution
Cs = np.logspace(-1,1,20)
prediction_model_lr = GridSearchCV(estimator=LogisticRegression(solver='liblinear'),param_grid=dict(C=Cs),cv=stra_k_fold,iid=False,n_jobs=-1)
prediction_model_lr.fit(train_X[:-50], train_y[:-50])
print(prediction_model_lr.best_score_, prediction_model_lr.best_estimator_.C)
print(cross_val_score(prediction_model_lr, train_X[-50:], train_y[-50:], cv=stra_k_fold, n_jobs=-1))
prediction_y_lr_cv = prediction_model_lr.predict(train_X[-50:])
prediction_y_lr_cv = prediction_y_lr_cv.round().astype('bool')*1
prediction_result_lr_cv = pd.DataFrame({'PredictionCV': prediction_y_lr_cv, 'ActualCV': train_y[-50:]})
prediction_result_lr_cv.to_csv('prediction_submission_lr_cv.csv', index=False)
#prediction_y_lr = prediction_model_lr.predict(test_X)
#prediction_y_lr = prediction_y_lr.round().astype('bool')*1
#prediction_result_lr = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_lr})
#prediction_result_lr.to_csv('prediction_submission_lr.csv', index=False)

# Naive Bayes Classifier Modelling and Solution
#prediction_model_nby = GaussianNB(priors=None)
#prediction_model_nby.fit(train_X, train_y)
#print(cross_val_score(prediction_model_nby, train_X[-50:], train_y[-50:], cv=stra_k_fold, n_jobs=-1))
#prediction_y_nby_cv = prediction_model_nby.predict(train_X[-50:])
#prediction_y_nby_cv = prediction_y_nby_cv.round().astype('bool')*1
#prediction_result_nby_cv = pd.DataFrame({'PredictionCV': prediction_y_nby_cv, 'ActualCV': train_y[-50:]})
#prediction_result_nby_cv.to_csv('prediction_submission_nby_cv.csv', index=False)
#prediction_y_nby = prediction_model_nby.predict(test_X)
#prediction_y_nby = prediction_y_nby.round().astype('bool')*1
#prediction_result_nby = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': prediction_y_nby})
#prediction_result_nby.to_csv('prediction_submission_nby.csv', index=False)

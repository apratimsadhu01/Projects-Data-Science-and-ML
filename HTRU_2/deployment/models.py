import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('machine learning-deep learning/HTRU_2/HTRU_2.csv')

y=df['Class']
x=df.drop('Class',axis=1)

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=21)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x=np.array(x_test)[0]
print(x)

x[:5]
print(x)

# # model training 
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import RandomForestClassifier 
# from sklearn import tree
# from sklearn.svm import SVC
# #from xgboost import XGBClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

# # Logistic Regression
# lg=LogisticRegression()
# lg.fit(x_train,y_train)
# lg_pred=lg.predict(x_test)
# print("ACCURACY OF LOGISTIC REGRESSION:",accuracy_score(y_test,lg_pred))

# # K-Nearest Neighbor
# knn=KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2 )
# knn.fit(x_train,y_train)
# knn_pred=knn.predict(x_test)
# print("ACCURACY OF K-NEAREST NEIGHBOR:",accuracy_score(y_test,knn_pred))

# # Support Vector Machine
# svc=SVC(C=1.0,kernel='linear',probability=True)
# svc.fit(x_train,y_train)
# svc_pred=svc.predict(x_test)
# print("ACCURACY OF SUPPORT VECTOR CLASSIFIER:",accuracy_score(y_test,svc_pred))

# # Support Vector Machine
# dt=tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
# dt.fit(x_train,y_train)
# dt_pred=dt.predict(x_test)
# print("ACCURACY OF DECISION TREE:",accuracy_score(y_test,dt_pred))

# # Random Forest Classifier
# rf=RandomForestClassifier(n_estimators=100,max_depth=6)
# rf.fit(x_train,y_train)
# rf_pred=rf.predict(x_test)
# print("ACCURACY OF RANDOM FOREST CLASSIFIER:",accuracy_score(y_test,rf_pred))

# # BaggingClassifier
# bg=BaggingClassifier()
# bg.fit(x_train,y_train)
# bg_pred=bg.predict(x_test)
# print("ACCURACY OF BAGGING CLASSIFIER:",accuracy_score(y_test,bg_pred))

# # Adaboost Classifier
# ada=AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
# ada.fit(x_train,y_train)
# ada_pred=ada.predict(x_test)
# print("ACCURACY OF ADABOOST CLASSIFIER:",accuracy_score(y_test,ada_pred))

# # Gradient Boosting Classifier
# gd=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
# gd.fit(x_train,y_train)
# gd_pred=gd.predict(x_test)
# print("ACCURACY OF GRADIENT BOOSTING CLASSIFIER:",accuracy_score(y_test,gd_pred))

# # exporting the models
# pickle.dump(lg,open('machine learning-deep learning/HTRU_2/deployment/lg.pkl','wb'))
# pickle.dump(knn,open('machine learning-deep learning/HTRU_2/deployment/knn.pkl','wb'))
# pickle.dump(svc,open('machine learning-deep learning/HTRU_2/deployment/svc.pkl','wb'))
# pickle.dump(dt,open('machine learning-deep learning/HTRU_2/deployment/dt.pkl','wb'))
# pickle.dump(rf,open('machine learning-deep learning/HTRU_2/deployment/rf.pkl','wb'))
# pickle.dump(bg,open('machine learning-deep learning/HTRU_2/deployment/bg.pkl','wb'))
# pickle.dump(ada,open('machine learning-deep learning/HTRU_2/deployment/ada.pkl','wb'))
# pickle.dump(gd,open('machine learning-deep learning/HTRU_2/deployment/gd.pkl','wb')) 
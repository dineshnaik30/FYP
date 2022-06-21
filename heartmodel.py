import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import joblib
 

data = pd.read_csv('heart.csv')
data_points = data.loc[:,["age","sex","cp","trestbps","fbs","restecg","thalach"]]
labels = data.loc[:,'target']


x_train,x_test,y_train,y_test = train_test_split(data_points,labels,test_size=0.2)

svm = SVC(kernel='rbf', random_state=3, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

joblib.dump(svm, 'svm_heart_model.pkl')

svm_from_joblib = joblib.load('svm_heart_model.pkl')
 
# Use the loaded model to make predictions
result = svm_from_joblib.predict([[1,2,3,4,5,6,7]])

print(result)
print('Training data accuracy {:.2f}'.format(svm.score(x_train, y_train)*100))
print('Testing data accuracy {:.2f}'.format(svm.score(x_test, y_test)*100))

#knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
#knn.fit(x_train,y_train)
#print('Training data accuracy {:.2f}'.format(knn.score(x_train, y_train)*100))
#print('Testing data accuracy {:.2f}'.format(knn.score(x_test, y_test)*100))

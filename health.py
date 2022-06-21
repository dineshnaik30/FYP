import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import joblib
 

data = pd.read_csv('healthData.csv')
data_points = data.loc[:,["Bodymass","BP","BodyTemp","Glucose","HeartRate"]]
labels = data.loc[:,'Target']



x_train,x_test,y_train,y_test = train_test_split(data_points,labels,test_size=0.2)

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

joblib.dump(svm, 'svm_health_model.pkl')

svm_from_joblib = joblib.load('svm_health_model.pkl')

result = svm_from_joblib.predict([[1,2,3,4,5]])

print(result[0])

print('Training data accuracy {:.2f}'.format(svm.score(x_train, y_train)*100))
print('Testing data accuracy {:.2f}'.format(svm.score(x_test, y_test)*100))



import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import matplotlib.pyplot as plt 

#loading Dataset.
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

#Creating Features and Labels
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

#Training/Testing Model
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.8)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#Accuracy
accuracy = clf.score(X_test, y_test)
print('Accuracy:')
print(accuracy)

#Prediction using new array
prediction_array = np.array([6,4,6,4,6,4,6,4,6]) #if 1st value changed to 8, we get cancer.
pred = prediction_array.reshape(1, -1)
prediction = clf.predict(pred)
print("Prediction:")
print(prediction)

if prediction == 4:
	print ("Cancer Detected")

elif prediction == 2:
	print ("No Cancer Detected")

#Graphs
#df.plot(kind = 'density', subplots = True, layout = (4,4), sharex = False)
#plt.show()


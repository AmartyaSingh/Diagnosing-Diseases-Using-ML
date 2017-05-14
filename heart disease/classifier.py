#Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier


#To display Normal or Disease infected.
def display(predicted_y, accuracy):
	if predicted_y != 0:
		print ("Heart Disease :", accuracy)
	else:
		print ("Normal:", accuracy)
	print ("")
	return (0)


#Collecting Data
dataframe = pd.read_csv('binary_ds.csv')
dataframe.replace("?", -99999, inplace = True)
ys = np.array(dataframe['y'])
#print (ys)
xs = np.array(dataframe.drop(['y'], 1))
prediction_x_features = [56,1,4,125,0,1,0,103,1,1,2,2,7]
#normal:prediction_x_features = [34, 1, 0, 125, 0, 1, 0, 103, 1, 0, 2, 2, 7]
dataframe.plot(kind = 'density', subplots = True, layout=(4,4), sharex = False)
plt.show()


def predictor():
	#RandomForestClassifier
	print ("Using RandomForest Classifier...")
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(xs, ys, test_size=0.5)
	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	accuracy_rf = clf.score(X_test, y_test)
	predicted_y_rf = clf.predict(np.reshape(prediction_x_features, (1,-1))) #np.reshape added to remove warning in runtime.
	print (predicted_y_rf)
	display(predicted_y_rf, accuracy_rf)


	#KNNclassifier
	print ("Using KNearestNeighbors...")
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(xs, ys, test_size=0.8)
	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train, y_train)
	accuracy_knn = clf.score(X_test, y_test)
	predicted_y_knn = clf.predict(np.reshape(prediction_x_features, (1,-1)))
	print (predicted_y_knn)
	display(predicted_y_knn, accuracy_knn)


	#SVM
	print ("Using SVM...")
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(xs, ys, test_size=0.1)
	clf = svm.SVC()
	clf.fit(X_train, y_train)
	accuracy_svm = clf.score(X_test, y_test)
	predicted_y_svm = clf.predict(np.reshape(prediction_x_features, (1,-1)))
	print (predicted_y_svm)
	display(predicted_y_svm, accuracy_svm)


	#NeuralNetwork
	#print ("Using NN...")
	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(xs, ys, test_size=0.1)
	#clf = MLPClassifier()
	#clf.fit(X_train, y_train)
	#accuracy_nn = clf.score(X_test, y_test)
	#predicted_y_nn = clf.predict(np.reshape(prediction_x_features, (1,-1)))
	#print (predicted_y_nn)
	#display(predicted_y_nn, accuracy_nn)

predictor()













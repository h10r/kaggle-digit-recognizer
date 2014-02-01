import numpy as np    

from sklearn import cross_validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class Classifier():
	def __init__(self, features):
		self.features = features
		
		self.load_from_features()
		
		#self.set_up_classifier()
		
		self.cross_validation()

	def load_from_features(self):
		print( "load_from_features" )

		self.train_values, self.train_labels = self.features.load_train()

		self.test_values = self.features.load_test()

	def set_up_classifier(self):
		print( "set_up_classifier" )

		print( "SVC: " )
		self.clf = SVC(gamma=0.001).fit( self.train_values, self.train_labels )
		self.predict_testing_set()

	def cross_validation(self):
		print( "cross_validation" )

		X_train, X_test, y_train, y_test = cross_validation.train_test_split( self.train_values,self.train_labels, random_state=1)
		
		
		print( "SVC: " )
		self.clf = SVC(gamma=0.0001).fit(X_train, y_train)
		print( self.clf.score( X_test, y_test ) )
		
		self.print_hr()
		
		print( "LogisticRegression: " )
		self.clf = LogisticRegression(C=1e5).fit(X_train, y_train)
		print( self.clf.score( X_test, y_test ) )

		self.print_hr()

		print( "LDA: " )
		self.clf = LDA().fit(X_train, y_train)
		print( self.clf.score( X_test, y_test ) )
		
		"""
		print( "KNeighborsClassifier: " )
		self.clf = KNeighborsClassifier(3).fit(X_train, y_train)
		print( clf.score( X_test, y_test ) )

		print( "DecisionTreeClassifier: " )
		self.clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
		print( clf.score( X_test, y_test ) )

		print( "RandomForestClassifier: " )
		self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, y_train)
		print( clf.score( X_test, y_test ) )

		print( "GaussianNB: " )
		self.clf = GaussianNB().fit(X_train, y_train)
		print( clf.score( X_test, y_test ) )
		"""

	def print_hr(self):
		print("")
		print( "*" * 80 )
		print("")

	def predict_testing_set(self):
		print( "predict_testing_set" )

		clf_predict = self.clf.predict( self.X_test )
		print( clf_predict )


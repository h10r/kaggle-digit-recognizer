import numpy as np    

from sklearn import cross_validation
from sklearn import metrics

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

		self.X_train,self.y_train = self.features.load_train()
		self.X_test,self.y_test = self.features.load_test()

	def set_up_classifier(self):
		print( "set_up_classifier" )
		pass

	def cross_validation(self):
		print("cross_validation")

		print( "SVC: " )
		clf = SVC(gamma=0.001).fit(X_train, y_train)
		self.predict_testing_set()

		"""
		print( "KNeighborsClassifier: " )
		clf = KNeighborsClassifier(3).fit(X_train, y_train)
		self.predict_testing_set()

		print( "DecisionTreeClassifier: " )
		clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
		self.predict_testing_set()

		print( "RandomForestClassifier: " )
		clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, y_train)
		self.predict_testing_set()

		print( "LogisticRegression: " )
		clf = LogisticRegression(C=1e5).fit(X_train, y_train)
		self.predict_testing_set()

		print( "GaussianNB: " )
		clf = GaussianNB().fit(X_train, y_train)
		self.predict_testing_set()

		print( "LDA: " )
		clf = LDA().fit(X_train, y_train)
		self.predict_testing_set()
		"""

	def predict_testing_set(self):
		print( "predict_testing_set" )

		for test_sample in self.X_test:
			print( test_sample )


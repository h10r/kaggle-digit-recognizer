import numpy as np    

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

		print( "self.X_train" )
		print( len(self.X_train) )
		print( "self.y_train" )
		print( len(self.y_train) )

		self.X_test = self.features.load_test()

	def set_up_classifier(self):
		print( "set_up_classifier" )

		print( "SVC: " )
		self.clf = SVC(gamma=0.001).fit(self.X_train, self.y_train)
		self.predict_testing_set()

		"""
		print( "KNeighborsClassifier: " )
		self.clf = KNeighborsClassifier(3).fit(self.X_train, self.y_train)
		self.predict_testing_set()

		print( "DecisionTreeClassifier: " )
		self.clf = DecisionTreeClassifier(max_depth=5).fit(self.X_train, self.y_train)
		self.predict_testing_set()

		print( "RandomForestClassifier: " )
		self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(self.X_train, self.y_train)
		self.predict_testing_set()

		print( "LogisticRegression: " )
		self.clf = LogisticRegression(C=1e5).fit(self.X_train, self.y_train)
		self.predict_testing_set()

		print( "GaussianNB: " )
		self.clf = GaussianNB().fit(self.X_train, self.y_train)
		self.predict_testing_set()

		print( "LDA: " )
		self.clf = LDA().fit(self.X_train, self.y_train)
		self.predict_testing_set()
		"""

	def cross_validation(self):
		print( "cross_validation" )

		cv_X_train, cv_X_test, cv_y_train, cv_y_test = cross_validation.train_test_split( self.X_train,self.y_train, test_size=0.3, random_state=0 )

		print( "SVC: " )
		self.clf = SVC(gamma=0.001).fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )

		"""
		print( "KNeighborsClassifier: " )
		self.clf = KNeighborsClassifier(3).fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )

		print( "DecisionTreeClassifier: " )
		self.clf = DecisionTreeClassifier(max_depth=5).fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )

		print( "RandomForestClassifier: " )
		self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )

		print( "LogisticRegression: " )
		self.clf = LogisticRegression(C=1e5).fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )

		print( "GaussianNB: " )
		self.clf = GaussianNB().fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )

		print( "LDA: " )
		self.clf = LDA().fit(cv_X_train, cv_y_train)
		print( clf.score( cv_X_test, cv_y_test ) )
		"""

	def predict_testing_set(self):
		print( "predict_testing_set" )

		clf_predict = self.clf.predict( self.X_test )
		print( clf_predict )


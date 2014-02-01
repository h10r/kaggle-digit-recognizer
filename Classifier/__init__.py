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
		
		self.set_up_classifier()

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

	def predict_testing_set(self):
		print( "predict_testing_set" )

		for test_sample in self.X_test[3]:
			print( test_sample )

			clf_predict = self.clf.predict( test_sample )

			print( clf_predict )


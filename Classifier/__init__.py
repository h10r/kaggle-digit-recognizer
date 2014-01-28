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
		
		self.X = []
		self.Y = []

		self.clf = linear_model.LogisticRegression(C=1e5)

	def cross_validation(self):
		print("cross_validation")

		#self.clf.fit( self.X,self.Y )

		#cross_validation.cross_val_score( self.clf, self.X, self.Y, cv=5, scoring='f1' )

		X_train, X_test, y_train, y_test = cross_validation.train_test_split( self.X,self.Y, test_size=0.3, random_state=0 )

		print( X_train.shape )
		print( y_train.shape )
		
		print( X_test.shape )
		print( y_test.shape )

		print( "SVC: " )
		clf = SVC(gamma=0.001).fit(X_train, y_train)
		print( clf.score(X_test, y_test) )

		print( "KNeighborsClassifier: " )
		clf = KNeighborsClassifier(3).fit(X_train, y_train)
		print( clf.score(X_test, y_test) )
		
		print( "DecisionTreeClassifier: " )
		clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
		print( clf.score(X_test, y_test) )

		print( "RandomForestClassifier: " )
		clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, y_train)
		print( clf.score(X_test, y_test) )

		print( "LogisticRegression: " )
		clf = LogisticRegression(C=1e5).fit(X_train, y_train)
		print( clf.score(X_test, y_test) )

		print( "GaussianNB: " )
		clf = GaussianNB().fit(X_train, y_train)
		print( clf.score(X_test, y_test) )

		print( "LDA: " )
		clf = LDA().fit(X_train, y_train)
		print( clf.score(X_test, y_test) )


import numpy as np    

from sklearn import cross_validation

import csv
from random import randint

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class Classifier():

	KNN_BENCHMARK = "data/knn_benchmark.csv"
	RF_BENCHMARK = "data/rf_benchmark.csv"

	def __init__(self, features):
		self.features = features
		
		self.load_from_features()
		
		self.run_classifier()

	def load_from_features(self):
		print( "load_from_features" )

		self.train_values, self.train_labels = self.features.load_train()

		#self.test_values = self.features.load_test()

	def run_classifier(self):
		print( "run_classifier" )
		print()
		print( "SVC: " )
		
		cv_train_values, cv_test_values, cv_train_labels, cv_test_labels = cross_validation.train_test_split( self.train_values,self.train_labels, random_state=1)
		
		print( "train_values" )
		print( len( cv_train_values ) )
		
		print( "test_values" )
		print( len( cv_test_values ) )
		
		print( "train_labels" )
		print( len( cv_train_labels ) )
		
		print( "test_labels" )
		print( len( cv_test_labels ) )

		self.clf = SVC(gamma=0.001).fit( cv_train_values, cv_train_labels )
		predicted_set = list( map( self.clf.predict, cv_test_values ) )
		compare_score = self.compare_two_sets( predicted_set, cv_test_labels )

		print( "compare_score" )
		print( compare_score )

	def cross_validation(self):
		print( "cross_validation" )

		
		print( "SVC: " )
		#self.clf = SVC(gamma=0.0001).fit(X_train, y_train)
		self.clf = SVC().fit(X_train, y_train)
		
		
		#print( self.clf.score( X_test, y_test ) )
		
		"""
		
		print( "LogisticRegression: " )
		self.clf = LogisticRegression(C=1e5).fit(X_train, y_train)
		print( self.clf.score( X_test, y_test ) )


		print( "LDA: " )
		self.clf = LDA().fit(X_train, y_train)
		print( self.clf.score( X_test, y_test ) )
		
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

	def compare_two_sets( self, set_a, set_b ):
		set_len = len( set_a )

		if not ( len(set_a) == len(set_b) ):
			print( "validation_set and train_set don't match" )
			return

		matches = 0
		for i in range( set_len ):
			print( set_a[i][0] )
			print( set_b[i][0] )
			print( )
			if set_a[i][0] == set_b[i][0]:
				matches = matches + 1

		return matches/float(set_len)

	def validate_with_knn( self, train_set ):
		caption,validation_set = self.load_csv( self.KNN_BENCHMARK )
		return compare_two_sets( train_set, validation_set )

	def validate_with_rf( self, train_set ):
		caption,validation_set = self.load_csv( self.RF_BENCHMARK )
		return compare_two_sets( train_set, validation_set )

	def load_csv( self, filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption,csv_as_list

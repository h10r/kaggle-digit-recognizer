import numpy as np    

from sklearn import cross_validation

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

	# via http://peekaboo-vision.blogspot.se/2010/09/mnist-for-ever.html
	GAMMA = 0.00728932024638
	C = 2.82842712475

	def __init__(self, data_source):
		self.data_source = data_source
		
		self.clf = SVC(C=self.C, kernel="rbf", gamma=self.GAMMA)

		#self.clf = RandomForestClassifier(n_estimators=100, n_jobs=2)
		#self.clf = GaussianNB()

		self.train, self.target, self.test = self.data_source.load_train_target_and_test()

		#self.run_classifier()

		self.run_and_write_classifier_for_release()

	def run_and_write_classifier_for_release(self):
		results = []
		
		self.clf.fit( self.train, self.target )

		predicted_probs = self.clf.predict( self.test )
		
		print( predicted_probs )

		self.data_source.write_delimited_file( "kaggle/heuer_kaggle_release.csv", predicted_probs )

	"""
	def run_classifier(self):

		print( "run_classifier" )
		print()
		
		cv_train_values, cv_test_values, cv_train_labels, cv_test_labels = cross_validation.train_test_split( self.train_values,self.train_labels, random_state=1)
		
		self.clf = self.clf_model.fit( cv_train_values, cv_train_labels )
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
			if set_a[i][0] == set_b[i][0]:
				matches = matches + 1

		return matches/float(set_len)

	"""
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
	"""

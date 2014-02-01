import csv
import numpy as np

class DataSource():

	PATH_TO_TRAIN = "data/train20.csv"
	PATH_TO_TEST = "data/test20.csv"
	
	#PATH_TO_TRAIN = "data/train.csv"
	#PATH_TO_TEST = "data/test.csv"

	def load_train(self):
		train_set = self.load_csv( self.PATH_TO_TRAIN )
		"""
		header = train_set[0,:]
		"""

		all_labels = train_set[1:,0]
		all_photos_without_labels = train_set[1:,1:]

		return all_photos_without_labels, all_labels
	
	def load_test(self):
		test_set = self.load_csv( self.PATH_TO_TEST )

		all_photos_without_header = test_set[1:,:]

		return all_photos_without_header

	def load_csv(self, filename):
		return np.asarray( list( csv.reader( open( filename, 'rt') ) ) )
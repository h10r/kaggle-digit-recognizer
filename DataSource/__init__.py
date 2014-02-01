import csv
import numpy as np

class DataSource():
	PATH_TO_TRAIN = "data/train20.csv"
	PATH_TO_TEST = "data/test20.csv"

	def load_train(self):
		train_set = self.load_csv( self.PATH_TO_TRAIN )
		"""
		header = train_set[0,:]
		"""

		all_labels = train_set[1:,0]
		all_photos_without_labels = train_set[1:,1:]

		return all_photos_without_labels, all_labels
	
	def load_test(self):
		return self.load_csv( self.PATH_TO_TEST )

	def load_csv(self, filename):
		return np.asarray( list( csv.reader( open( filename, 'rt') ) ) )
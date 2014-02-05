import csv

from DataSource import *

class Benchmark():
	KAGGLE = "kaggle/heuer_kaggle_release.csv"

	KNN_BENCHMARK = "data/knn_benchmark.csv"
	RF_BENCHMARK = "data/rf_benchmark.csv"

	def __init__(self):
		self.train = DataSource.read_data( self.KAGGLE )
		
	def compare_two_sets( self, train_set, benchmark_set ):
		train_set_len = len( train_set )

		if train_set_len < len(benchmark_set):
			benchmark_set = benchmark_set[:train_set_len]

		matches = 0
		for i in range( train_set_len ):
			if int( train_set[i][1] ) == int( benchmark_set[i][1] ):
				matches = matches + 1

		return matches/float(train_set_len)

	def validate_with_knn( self ):
		_,validation_set = self.load_csv( self.KNN_BENCHMARK )
		return self.compare_two_sets( self.train, validation_set )

	def validate_with_rf( self ):
		_,validation_set = self.load_csv( self.RF_BENCHMARK )
		return self.compare_two_sets( self.train, validation_set )

	def load_csv( self, filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption,csv_as_list

import csv
from random import randint

class Validation():

	KNN_BENCHMARK = "data/knn_benchmark.csv"
	RF_BENCHMARK = "data/rf_benchmark.csv"

	def for_testing():
		caption,dataset = Validation.load_csv( Validation.RF_BENCHMARK )
		return dataset

	def validate_with_knn(train_set):
		caption,validation_set = Validation.load_csv( Validation.KNN_BENCHMARK )

		if ( len(validation_set) == len(train_set) ):
			print( Validation.compare_two_lists( validation_set, train_set, len( validation_set ) ) )
		else:
			print( "validation_set and train_set don't match" )
			return

	def validate_with_rf(train_set):
		caption,validation_set = Validation.load_csv( Validation.RF_BENCHMARK )

		if ( len(validation_set) == len(train_set) ):
			print( Validation.compare_two_lists( validation_set, train_set, len( validation_set ) ) )
		else:
			print( "validation_set and train_set don't match" )
			return

	def compare_two_lists( set_a, set_b, set_len ):
		matches = 0
		for i in range( set_len ):
			if set_a[i][1] == set_b[i][1]:
				matches = matches + 1

		return matches/float(set_len)

	def load_csv(filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption,csv_as_list

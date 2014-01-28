import csv

class Validation():

	KNN_BENCHMARK = "data/knn_benchmark.csv"
	RF_BENCHMARK = "data/rf_benchmark.csv"

	def for_testing():
		caption,dataset = Validation.load_csv( Validation.KNN_BENCHMARK )
		return dataset

	def validate_with_knn(results):
		caption,dataset = Validation.load_csv( Validation.KNN_BENCHMARK )

		if ( len(dataset) == len(results) ):
			print( len(dataset) )
		else:
			print( "Dataset and results don't match" )
			return

	def validate_with_rf(results):
		caption,dataset = Validation.load_csv( Validation.RF_BENCHMARK )

		if ( len(dataset) == len(results) ):
			print( len(dataset) )
		else:
			print( "Dataset and results don't match" )
			return

	def load_csv(filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption,csv_as_list

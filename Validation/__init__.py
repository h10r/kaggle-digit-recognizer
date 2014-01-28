import csv

class Validation():

	KNN_BENCHMARK = "../data/knn_benchmark.csv"
	RF_BENCHMARK = "../data/rf_benchmark.csv"

	def validate_with_knn(results):
		caption,dataset = load_csv( Validation.KNN_BENCHMARK )

		if ( len(dataset) != len(results) ):
			print( "Dataset and results don't match" )
			return

	def validate_with_rf(results):
		caption,dataset = load_csv( Validation.RF_BENCHMARK )

		if ( len(dataset) != len(results) ):
			print( "Dataset and results don't match" )
			return

	def load_csv(filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption,csv_as_list

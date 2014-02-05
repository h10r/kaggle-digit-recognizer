
class Benchmark():
	KAGGLE = "kaggle/heuer_kaggle_release.csv"

	KNN_BENCHMARK = "data/knn_benchmark.csv"
	RF_BENCHMARK = "data/rf_benchmark.csv"

	def __init__(self, data_source):
		self.data_source = data_source

		_,self.train = self.load_csv( self.KAGGLE )
		
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

	def validate_with_knn( self ):
		_,validation_set = self.load_csv( self.KNN_BENCHMARK )
		return compare_two_sets( self.train, validation_set )

	def validate_with_rf( self ):
		_,validation_set = self.load_csv( self.RF_BENCHMARK )
		return compare_two_sets( self.train, validation_set )

	def load_csv( self, filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption,csv_as_list

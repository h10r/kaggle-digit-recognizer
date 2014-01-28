import csv

class Features():
	PATH_TO_TRAIN = "data/train_n5.csv"
	PATH_TO_TEST = "data/test_n5.csv"

	def load_train(self):
		return self.load_csv( self.PATH_TO_TRAIN )
	
	def load_test(self):
		return self.load_csv( self.PATH_TO_TEST )

	def load_csv(self, filename):
		csv_as_list = list( csv.reader(open( filename, 'rt') ) )
		caption = csv_as_list.pop(0)
		return caption, csv_as_list

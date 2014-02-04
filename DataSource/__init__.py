import numpy as np

class DataSource():

	PATH_TO_TRAIN = "data/train20.csv"
	PATH_TO_TEST = "data/test20.csv"
	
	#PATH_TO_TRAIN = "data/train.csv"
	#PATH_TO_TEST = "data/test.csv"

	# https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience/history/969

	def read_data( self, file_name ):
		f = open(file_name)
		
		#ignore header
		f.readline()

		samples = []
		for line in f:
			line = line.strip().split(",")
			sample = [int(x) for x in line]
			samples.append(sample)
		return samples

	def write_delimited_file( self, file_path, data,header=None, delimiter="," ):
		f_out = open(file_path,"w")
		if header is not None:
			f_out.write(delimiter.join(header) + "\n")
		for line in data:
			if isinstance(line, str):
				f_out.write(line + "\n")
			else:
				f_out.write( str(line) + "\n")
		f_out.close()

	def load_train_target_and_test( self ):
		train = self.read_data( self.PATH_TO_TRAIN )

		target = [x[0] for x in train]
		train = [x[1:] for x in train]
		
		test = self.read_data( self.PATH_TO_TEST )
		
		return train,target,test
		

#!/opt/local/bin/python3.3

from DataSource import *
from Classifier import *
from Validation import *

if __name__ == "__main__":
	d = DataSource()
	
	c = Classifier( d )

	#Validation.validate_with_knn( Validation.for_testing() )


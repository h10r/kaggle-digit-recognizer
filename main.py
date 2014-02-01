#!/opt/local/bin/python3.3

from Features import *
from Classifier import *
from Validation import *

if __name__ == "__main__":
	f = Features()
	c = Classifier( f )

	#Validation.validate_with_knn( Validation.for_testing() )


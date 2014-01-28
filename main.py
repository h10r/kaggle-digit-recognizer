#!/opt/local/bin/python3.3

from Features import *
from Classifier import *
from Validation import *

if __name__ == "__main__":
	# f = Features()

	# label,values = f.load_train()
	# print(label)

	print( Validation.validate_with_knn() )

	#c = Classifier( f )

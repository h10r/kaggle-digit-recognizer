#!/opt/local/bin/python3.3

from Benchmark import *

if __name__ == "__main__":
	b = Benchmark()

	print( b.validate_with_knn() )
	print( b.validate_with_rf() )

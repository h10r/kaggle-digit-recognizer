#!/opt/local/bin/python3.3

from DataSource import *
from Classifier import *

if __name__ == "__main__":
	d = DataSource()
	
	c = Classifier( d )


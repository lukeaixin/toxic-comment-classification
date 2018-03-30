import sys
import csv

data_file = sys.argv[1]
name_file = sys.argv[2]

with open(data_file, 'r') as rfile:
	reader = csv.reader(rfile)
	names = next(reader)

with open(name_file, 'w') as wfile:
	for name in names:
		wfile.write(name + '\n')
import numpy as np
import csv

def write_csv(f):
	'''opens a csv writer object'''	
	return csv.writer(open(f,"w"))

def iter_to_csv(iterator, f):
	'''writes the contents of a generator to a csv file'''
	out = write_csv(f)
	
	out.writerow(iterator)

def confidence_interval(data, confidence=0.95):
    '''determines the confidence intervals of a given data object'''
    mean = np.mean(data)
    sample_size = len(data)
    std = np.std(data)
    standard_error = std / (np.sqrt(sample_size))
    margin_of_error = float(standard_error * 2)
    low_bound = float(mean - margin_of_error)
    high_bound = float(mean + margin_of_error)

    print(low_bound, high_bound)
    return low_bound, high_bound 
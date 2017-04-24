import csv

import numpy as np
from numpy import genfromtxt, savetxt
from sklearn.preprocessing import Imputer


# Load the predicors' header row
with open('data/predictors.csv', 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)

# Load all the predictors.
print 'Loading the data from CSVs'
predictors = genfromtxt('data/predictors.csv', delimiter=',', skip_header=1)

# Impute the predictors. This will remove all the columns which only contain
# NaNs.
print 'Imputing the predictors (%d columns)' % len(predictors[0])
median_imp = Imputer(strategy='median')
mode_imp = Imputer(strategy='most_frequent')
predictors_median = median_imp.fit_transform(predictors)
predictors_mode = mode_imp.fit_transform(predictors)

def get_valid_indexes(imp):
    """Given an imputer, return which of the original columns haven't
    been removed.
    """
    invalid_mask = np.isnan(imp.statistics_)
    valid_mask = np.logical_not(invalid_mask)
    valid_idx, = np.where(valid_mask)
    return valid_idx

def get_new_header(imp):
    """Return the new header row given all the columns removed by the
    imputer.
    """
    new_header = []
    valid_indexes = set(get_valid_indexes(imp))
    for idx, name in enumerate(header_row):
        if idx in valid_indexes:
            new_header.append(name)
    return ','.join(new_header)

# Save the imputed predictors along with a tweaker header row.
print 'Saving the imputed predictors (%d columns remaining)' % len(predictors_median[0])
savetxt('data/predictors_median.csv', predictors_median, delimiter=',', header=get_new_header(median_imp), comments='')
savetxt('data/predictors_mode.csv', predictors_mode , delimiter=',', header=get_new_header(mode_imp), comments='')

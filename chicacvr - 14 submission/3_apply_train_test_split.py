import csv

from numpy import genfromtxt, savetxt
from sklearn.model_selection import train_test_split


# Load a numpy matrix of predictors (imputed via median and mode)
print 'Loading a matrix of predictors'
predictors_median = genfromtxt('data/predictors_median.csv', delimiter=',', skip_header=1)
predictors_mode = genfromtxt('data/predictors_mode.csv', delimiter=',', skip_header=1)

# Load an array of GPAs
print 'Loading an array of GPAs'
reader = csv.reader(open('data/responses.csv', 'r'))
next(reader) # skip the header
gpas = []
for row in reader:
    gpas.append(float(row[1]))

# Apply the train test split (median)
print 'Splitting the training and testing data (median)'
predictors_train, predictors_test, responses_train, responses_test = \
    train_test_split(predictors_median, gpas, test_size=0.2, random_state=0)

print 'Saving the CSVs'
savetxt('data/predictors_median_training.csv', predictors_train, delimiter=',')
savetxt('data/predictors_median_test.csv', predictors_test, delimiter=',')
savetxt('data/responses_median_training.csv', responses_train, delimiter=',')
savetxt('data/responses_median_test.csv', responses_test, delimiter=',')

# Apply the train test split (mode)
print 'Splitting the training and testing data (mode)'
predictors_train, predictors_test, responses_train, responses_test = \
    train_test_split(predictors_mode, gpas, test_size=0.2, random_state=0)

print 'Saving the CSVs'
savetxt('data/predictors_mode_training.csv', predictors_train, delimiter=',')
savetxt('data/predictors_mode_test.csv', predictors_test, delimiter=',')
savetxt('data/responses_mode_training.csv', responses_train, delimiter=',')
savetxt('data/responses_mode_test.csv', responses_test, delimiter=',')

print 'Done!'

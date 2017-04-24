import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from a2core import *
from scipy import sparse
import sys
import pickle
#import marshal

# Cast all non-na values to float and ensure it works...
def dump_imputation(X_num,X_cat,X_header):
    cols_to_cat = set()
    for rowi in xrange(len(X_num)):
        for coli in xrange(len(X_num[rowi])):
            if X_num[rowi][coli] == 'NA':
                print "Value is NA"
                continue
            try:
                X_num[rowi][coli] = float(X_num[rowi][coli])
            except:
                print "Error with column " + str(coli) + ":" + str(X_num[rowi][coli])
                cols_to_cat.add(coli)
    X_num = np.array(X_num,dtype=np.double)
    print "X_num size:"
    print X_num.size
    print "Cols that should be categorical:"
    print cols_to_cat
    print "Idxs that should be categorical:"
    print [X_num_idxs[i] for i in list(cols_to_cat)]
    print "Ids that should be categorical:"
    print [X_header[j] for j in [X_num_idxs[i] for i in list(cols_to_cat)]]
    #sys.exit(1)
    imputed = impute_features_with_linear_regression(X_cat,X_num)

    f = open("imputed_lasso.pk",'wb')
    pickle.dump(imputed,f)
    f.close()

# f = open("imputed.pk",'rb')
# imputed = marshal.load(f)
# f.close()
# print "Imputed dims:"
# print_dims(imputed)
# print imputed[0]
#
# X_num = np.array(imputed,dtype=np.double)
# print "X_num shape: " + str(X_num.shape)


# Start
def compute_X_cat():
    X = read_csv_to_lists('background.csv')

    print "Original X has dims:"
    print_dims(X)

    X = preprocess_header(X)
    X = remove_irrelevant_features(X)

    # print "Fraction of columns with an NA: " + str(fraction_of_cols_with_na(X))
    # print "Fraction of NAs in X: " + str(average_na_fraction(X))
    # print "Fraction of NAs in X in columns containing at least one NA: " + str(average_na_fraction_where_na_present(X))
    # for frac in [0.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
    #     col_frac = fraction_of_cols_with_at_least_fraction_of_na(X,frac)
    #     print "Fraction of columns that are at least " + str(frac*100.0) + "% NA: " + str(col_frac)
    # X = drop_cols_with_na_frac(X,0.4)
    # Try to impute the rest
    # print "Dropped features with more than 40% NA values"
    # print "Fraction of columns with an NA: " + str(fraction_of_cols_with_na(X))
    # print "Fraction of NAs in X: " + str(average_na_fraction(X))

    na_mapping = find_unused(X)
    print "The category " + str(na_mapping) + " is free."
    X = replace_in_place(X,'NA',str(na_mapping))

    print "Splitting numerical and categorical features..."
    # Strip header
    X_header = X[0]
    assert (X_header[-1] == 'challengeID')
    f = open("X_header.pk",'wb')
    pickle.dump(X_header,f)
    f.close()
    X = X[1:]

    # Pull of challengeID values
    X_challengeIDs = [row[-1] for row in X]
    f = open("X_challengeIDs.pk",'wb')
    pickle.dump(X_challengeIDs,f)
    f.close()
    X = [row[:-1] for row in X]

    X_num,X_cat,X_num_idxs = categorical_numerical_split(X)
    print "There are " + str(len(X_num[0])) + " numerical and " + str(len(X_cat[0])) + " categorical variables."
    print "X_cat dims:"
    print_dims(X_cat)
    # Handle the categorical variables by simply remapping NA to a new, unused category.
    print_categorical_variable_warnings(X_cat)

    # TODO:
    # Remap the categorical values such that various missing values are just other categories.
    X_cat = reindex_categories(X_cat)
    f = open('X_cat_pre_one_hot_encoding.pk','wb')
    pickle.dump(X_cat,f)
    f.close()

    # Recode categorical values to be one-hot encodings that are numerical
    one_hot_encoder = OneHotEncoder(sparse=True)
    X_cat = one_hot_encoder.fit_transform(X_cat)
    joblib.dump(one_hot_encoder,"one-hot-encoder.pk")

    f = open("X_cat_one_hot_encoded.pk",'wb')
    pickle.dump(X_cat,f)
    f.close()

    f = open("X_num_tmp.pk","wb")
    pickle.dump(X_num,f)
    f.close()
#print "Categorical dimensions:"
#print_dims(X_cat)
#scaler = StandardScaler()
#X_cat = scaler.fit_transform(X_cat)


# Impute missing values with regression for numerical values
compute_X_cat()

f = open("X_cat_one_hot_encoded.pk",'rb')
X_cat = pickle.load(f)
f.close()
print "X_cat type:" + str(type(X_cat))


f = open("X_header.pk",'rb')
X_header = pickle.load(f)
f.close()
print "X_header type:" + str(type(X_header))

f = open("X_challengeIDs.pk",'rb')
X_challengeIDs = np.matrix(pickle.load(f),dtype=np.object)
f.close()
print "X_challengeIDs shape and type: " + str(X_challengeIDs.shape) + "," + str(type(X_challengeIDs))


# Block comment out together
# f = open("X_num_tmp.pk","rb")
# X_num = pickle.load(f)
# f.close()
# dump_imputation(X_num,X_cat,X_header)

f = open("imputed.pk",'rb')
imputed = pickle.load(f)
f.close()
print "Imputed dims:"
print_dims(imputed)
X_num = np.array(imputed,dtype=np.double)
print "X_num shape: " + str(X_num.shape)
print "X_num[0]: " + str(X_num[0,:])
print "X_num type: " + str(type(X_num))

print "X_cat: shape: " + str(X_cat.shape)
print "Casting X_num to sparse for concatenation:"
X_num = sparse.csr_matrix(X_num)
print "X_cat type and shape: " + str(type(X_num)) + "," + str(X_num.shape)

# Recombine for final output feature vectors.

X = sparse.csc_matrix(sparse.hstack((X_cat,X_num)))
print "X Shape: " + str(X.shape)
print "X type: " + str(type(X))


# Other ideas- next split into time series somehow??


#print "Dropping header row:"
#X = remove_header_row(X)
#
# print "New X dims:"
# print_dims(X)
#
# print "Removing Nonnumeric features from X:"
# X = remove_nonnumeric_columns(X)
# print "New X dims:"
# print_dims(X)

# Drop challengeID cols from X, X_train and y_train.
# Make a list for X that is the challengeID vals
# X_out = np.array(X,dtype=np.double)
#
# X_challengeIDs = [int(x[-1]) for x in X]
# X_out = X_out[:,:-1]
# joblib.dump(X_challengeIDs,'X_challengeIDs.pk')
# joblib.dump(X_out,'X.pk')
joblib.dump(X,'X.pk')

print "Loading partial training data"
y_partial = read_csv_to_lists('train.csv')[1:]

y_names = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']

X_treg,y_treg = filter_rows_to_training_and_align(X,X_challengeIDs,get_cols(y_partial,[0,1,2,3]),[1,2,3])
X_tclf,y_tclf = filter_rows_to_training_and_align(X,X_challengeIDs,get_cols(y_partial,[0,4,5,6]),[1,2,3])
X_tall,y_tall = filter_rows_to_training_and_align(X,X_challengeIDs,get_cols(y_partial,[0,1,2,3,4,5,6]),[1,2,3,4,5,6])
y_treg = np.array(y_treg,dtype=np.double)[:,1:]
y_tclf = np.array(y_tclf,dtype=np.double)[:,1:]
y_tall = np.array(y_tall,dtype=np.double)[:,1:]
joblib.dump(X_treg,'X_treg.pk')
joblib.dump(y_treg,'y_treg.pk')
joblib.dump(X_tclf,'X_tclf.pk')
joblib.dump(y_tclf,'y_tclf.pk')
joblib.dump(X_tall,'X_tall.pk')
joblib.dump(y_tall,'y_tall.pk')


for idx in xrange(1,len(y_names)+1):
    print "Dimension: " + y_names[idx-1]
    # Filter to the results for the current y value
    y_train = get_cols(y_partial,[0,idx])

    X_train,y_train = filter_rows_to_training_and_align(X,X_challengeIDs,y_train,[1])

    print "X_train shape: " + str(X_train.shape)
    print "y_train dimensions:"
    print_dims(y_train)

    # # Assert matched and sorted properly EDIT: Can't do in new format.
    # for i in xrange(len(y_train)):
    #     #print str(int(X_train[i][-1])) + "," + str(int(y_train[i][0]))
    #     assert(int(X_train[i][-1]) == int(y_train[i][0]))

    # X_train = np.array(X_train,dtype=np.double)
    # X_train = X_train[:,:-1]

    y_train = np.array(y_train,dtype=np.double)
    y_train = y_train[:,1]

    print "Final dimensions:"
    print "X_train: " + str(X_train.shape)
    print "y_train: " + str(y_train.shape)

    joblib.dump(X_train,'X_train_' + y_names[idx-1] + '.pk')
    joblib.dump(y_train,'y_train_' + y_names[idx-1] + '.pk')

import sklearn.linear_model
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
def read_csv_to_lists(fname):
    f = open(fname,'r')
    rows = [x for x in f.read().split('\n') if x != '']
    rows = [x.split(',') for x in rows]
    f.close()
    first_rowlen = len(rows[0])
    for row in rows:
        if (len(row) != first_rowlen):
            print "Warning: Non-square CSV. Row has len " + str(len(row)) + " but first row has len " + str(first_rowlen)
    print "Dimensions for csv " + fname + ": " + str(len(rows)) + " rows and " + str(len(rows[0])) + " columns."
    return rows

def write_lists_to_csv(csv,fname):
    f = open(fname,'w')
    lines = '\n'.join([','.join([str(x) for x in line]) for line in csv]) + '\n'
    f.write(lines)
    f.close()

def print_dims(arr):
    first_rowlen = len(arr[0])
    for row in arr:
        if (len(row) != first_rowlen):
            print "Warning: Non-square CSV. Row has len " + str(len(row)) + " but first row has len " + str(first_rowlen)
    print "RxC: " + str(len(arr)) + "x" + str(first_rowlen)

def get_cols(arr,colidxs):
    if type(colidxs) is int:
        return [row[colidxs] for row in arr]
    return [[row[i] for i in colidxs] for row in arr]

# Pass a list of lists. Finds and removes challengeID row wherever it is.
def remove_header_row(Xy):
    for rowidx in xrange(len(Xy)):
        for colidx in xrange(len(Xy[0])):
            if Xy[rowidx][colidx] == 'challengeID':
                return [Xy[i] for i in xrange(len(Xy)) if i != rowidx]
    print "Error: Could not find challengeID entry."
    assert False

# X should be a list of lists, rather than a numpy array
# Assumes that the LAST row contains the data names.
def remove_nonnumeric_columns(X):
    # Just drop any nonumeric features to start
    nonnumeric_cols = []
    for colidx in xrange(len(X[0])-1):
        col_nonnum = False
        for rowidx in xrange(len(X)):
            try:
                X[rowidx][colidx] = float(X[rowidx][colidx])
            except:
                col_nonnum = True
                break
        if (col_nonnum):
            nonnumeric_cols.append(colidx)
    print "There are " + str(len(nonnumeric_cols)) + " nonnumeric columns"
    return get_cols(X,[x for x in range(len(X[0])) if x not in nonnumeric_cols])


def challengeIDs_to_row_idxs(X_challengeIDs,selected_challengeIDs):
    X_challengeIDs = [int(x) for x in X_challengeIDs.tolist()[0]]
    print "There are " + str(len(X_challengeIDs)) + " challengeIDs:"
    challengeID_to_row = {X_challengeIDs[i]:i for i in xrange(len(X_challengeIDs))}
    print challengeID_to_row
    sorted_ids = sorted(selected_challengeIDs)
    print "There are " + str(len(sorted_ids)) + " sorted selected ids:"
    print sorted_ids
    return [challengeID_to_row[int(idx)] for idx in sorted(selected_challengeIDs)] # Sorted by challengeID order

# Takes in a list of lists (X) and a list of lists (y).
# Assumes that the first column of y is challengeID
# Assumes that the last column of X is challengeID
# Returns X without rows for which no training data exists in dimension dim
# Lines up rows of X_train,y_train
def filter_rows_to_training_and_align(X,X_challengeIDs,y_train,dims):
    # In case it is a boolean category, rename.
    for rowidx in xrange(len(y_train)):
        for dim in dims:
            if y_train[rowidx][dim] == 'FALSE':
                y_train[rowidx][dim] = 0
            elif y_train[rowidx][dim] == 'TRUE':
                y_train[rowidx][dim] = 1
    incomplete_training_result_idxs = []
    for rowidx in xrange(0,len(y_train)):
        for colidx in dims:
        #print "Row,col ids: " + str(rowidx) + "," + str(colidx)
            if (y_train[rowidx][colidx] == "NA"):
                incomplete_training_result_idxs.append(rowidx)
                break
    y_train = [y_train[i] for i in xrange(len(y_train)) if i not in incomplete_training_result_idxs]
    #print "Y train:"
    #print y_train
    training_ids = [int(row[0]) for row in y_train]
    training_X_row_idxs = challengeIDs_to_row_idxs(X_challengeIDs,training_ids)
    X_train = X[training_X_row_idxs,:]
    #print "training ids: " + str(training_ids)
    #print "last col of X_train: " + str([int(x[-1]) for x in X])
    #X_train = [row for row in X if int(row[-1]) in training_ids]
    #X_train = sorted(X_train,key=lambda x: x[-1])
    y_train = sorted(y_train,key=lambda x: int(x[0]))
    print y_train
    return (X_train,y_train)

# Assumes y is now a single column (i.e. list)
def encode_bool_y_as_regression(y):
    return [0.0 if x == 'FALSE' else 1.0 for x in y]

def decode_int_to_bool(y):
    return ['FALSE' if x < 0.5 else 'TRUE' for x in y]

def invert_dict(d):
    return {v:k for k,v in d.iteritems()}
    i = {}
    for k,v in d.iteritems():
        i[v] = k
    return i

# Assumes y is now a single column
def encode_float_as_cat(y):
    flts = list({e for e in y})
    print "Distinct Float Values: " + str(flts)
    cat_to_idx = {flts[i]:i for i in xrange(len(flts))}
    y_cat = map(lambda e: cat_to_idx[e], y)
    return (y_cat,cat_to_idx)

def decode_float_as_cat(y_cat,cat_to_idx):
    idx_to_flt = invert_dict(cat_to_idx)
    y = map(lambda e: idx_to_flt[e], y_cat)
    return y

def transpose(arr):
    return [[arr[i][j] for i in xrange(len(arr))] for j in xrange(len(arr[0]))]

def col_to_list(arr,colidx):
    return [row[colidx] for row in arr]

def list_has_na(l):
    return len(filter(lambda x: x == 'NA',l)) > 0

def fraction_missing_in_list(l):
    num_nas = len(filter(lambda x: x == 'NA',l))
    return (float(num_nas)) / float(len(l))

def fraction_of_cols_with_na(arr):
    counter = 0
    for colidx in xrange(len(arr[0])):
        if (list_has_na(col_to_list(arr,colidx))):
            counter += 1
    return float(counter) / float(len(arr[0]))

def average_na_fraction(arr):
    fracs = [fraction_missing_in_list(col_to_list(arr,colidx)) for colidx in xrange(len(arr[0]))]
    return sum(fracs) / float(len(fracs))

def average_na_fraction_where_na_present(arr):
    arr_new = []
    cols = transpose(arr)
    for col in cols:
        if list_has_na(col):
            arr_new.append(col)
    return average_na_fraction(transpose(arr_new))

def fraction_of_cols_with_at_least_fraction_of_na(arr,min_na_frac):
    fracs = [fraction_missing_in_list(col_to_list(arr,colidx)) for colidx in xrange(len(arr[0]))]
    min_fracs = filter(lambda x: x > min_na_frac,fracs)
    return float(len(min_fracs)) / float(len(arr[0]))

def drop_cols_with_na_frac(arr,frac):
    cols = transpose(arr)
    fracs = [fraction_missing_in_list(cols[colidx]) for colidx in xrange(len(arr[0]))]
    out = []
    for colidx in xrange(len(arr[0])):
        if (fracs[colidx] <= frac):
            out.append(cols[colidx])
    return transpose(out)
#def clfs_to_prediction(X, X_challengeIDs, clfs):

def get_colidxs_with_na_present(arr):
    fracs = [fraction_missing_in_list(col_to_list(arr,colidx)) for colidx in xrange(len(arr[0]))]
    return [i for i in xrange(len(fracs)) if fracs[i] > 0.0]

def remove_irrelevant_features(arr):
    header = arr[0]
    ids = [elt for elt in header if 'id' in elt]
    ids = [elt for elt in ids if 'grid' not in elt and 'kid' not in elt and elt != 'pcg5idstat']
    explicitly_dropped = ids + ['cf4fint']
    print explicitly_dropped
    features = transpose(arr)
    cols = [feature for feature in features if feature[0] not in explicitly_dropped]
    out = transpose(cols)
    print "There were " + str(len(header)) + " features and now there are " + str(len(out[0])) + " features."
    return out

# Get rid of quotes
def preprocess_header(arr):
    for idx in xrange(len(arr[0])):
        arr[0][idx] = arr[0][idx].replace('"','')
    return arr

def is_present(arr,e):
    for row in arr:
        for elt in row:
            if elt == e:
                return True
    return False

def find_unused(arr):
    e = -10
    while is_present(arr,str(e)):
        e -= 1
    return e

def replace_in_place(arr,o,n):
    for ri in xrange(len(arr)):
        for ci in xrange(len(arr[ri])):
            if (arr[ri][ci] == o):
                arr[ri][ci] = n
    return arr

def categorical_numerical_split(arr):
    features = transpose(arr)
    num_feature_idxs = []
    manual_in_cat = [11467, 10012, 10015, 10016, 10020, 10021, 10023, 10024]
    #'hv4j5a_ot', 'hv5_dspr', 'hv5_ppvtae', 'hv5_ppvtpr', 'hv5_wj9pr', 'hv5_wj9ae', 'hv5_wj10pr', 'hv5_wj10ae'
    for feature_idx in xrange(len(features)):
        if feature_idx in manual_in_cat:
            continue
        if len({elt for elt in features[feature_idx]}) > 18:
            num_feature_idxs.append(feature_idx)
            continue
        for instance in features[feature_idx]:
            if '.' in instance:
                num_feature_idxs.append(feature_idx)
                break

    numerical_features = transpose([features[idx] for idx in num_feature_idxs])
    categorical_features_idxs = [idx for idx in xrange(len(features)) if idx not in num_feature_idxs]
    categorical_features = transpose([features[idx] for idx in categorical_features_idxs])
    return (numerical_features,categorical_features,num_feature_idxs)

def print_categorical_variable_warnings(arr):
    # If a presumed categorical value has too many categories, print a warning
    features = transpose(arr)
    for feature in features:
        elts = {x for x in feature}
        if len(elts) > 18:
            print "Warning: Category " + feature[0] + " has " + str(len(elts)) + " categories."

def reindex_categories(arr):
    features = transpose(arr)
    out = []
    for feature in features:
        elts = list({x for x in feature})
        recoder = {elts[i]:i for i in xrange(len(elts))}
        out.append([recoder[elt] for elt in feature])
    return transpose(out)

def impute_features_with_linear_regression(X, to_impute):
    var_sel = VarianceThreshold(threshold=(0.249))
    print "X dims: " + str(X.shape)
    X = var_sel.fit_transform(X)
    print "New X dims: " + str(X.shape)
    ys = transpose(to_impute)
    imputed = []
    for yi in xrange(len(ys)):
        print "Imputing variable " + str(yi) + "..."
        y = list(ys[yi])
        print "Length y:" + str(len(y))
        # Split into train and test by getting the values that exist vs those that don't
        predict_idxs = [ei for ei in xrange(len(y)) if y[ei] == 'NA' or y[ei] < 0.0]
        if (len(predict_idxs) == 0):
            print "Variable " + str(yi) + " does not need imputation"
            continue
        train_idxs = [ei for ei in xrange(len(y)) if ei not in predict_idxs]
        print "There are " + str(len(train_idxs)) + " known values and " + str(len(predict_idxs)) + " values to impute."
        X_train = X[train_idxs,:]#[X[i] for i in train_idxs]
        X_predict = X[predict_idxs,:]#[X[i] for i in predict_idxs]
        y_train = [y[i] for i in train_idxs]
        #print "Computing f_regression..."
        #selector = SelectKBest(mutual_info_regression, k=500)
        #X_train = selector.fit_transform(X_train,y_train)
        #X_predict = selector.transform(X_predict)
        print "Imputing..."
        #imputer = sklearn.linear_model.LassoLarsCV(n_jobs=-1,normalize=False)
        imputer = sklearn.linear_model.LinearRegression(normalize=True,n_jobs=-1)
        imputer.fit(X_train,y_train)
        y_predict = imputer.predict(X_predict)
        y_out = []
        predidx = 0
        knownidx = 0
        for i in xrange(len(y)):
            if i in train_idxs:
                y_out.append(y_train[knownidx])
                knownidx += 1
            else:
                assert(i in predict_idxs)
                y_out.append(y_predict[predidx])
                predidx += 1
        print "Appending: " + str(y_out)
        imputed.append(y_out)
    return transpose(imputed)

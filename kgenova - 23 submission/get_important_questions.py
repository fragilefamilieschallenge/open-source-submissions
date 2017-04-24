
import pickle
from a2core import *
from sklearn.externals import joblib



def feature_to_sum_of_weights(encoder,i,coefs):
    if (i >= encoder.feature_indices_.shape[0]-1):
        return coefs[i]
    weights_start = encoder.feature_indices_[i]
    weights_end = encoder.feature_indices_[i+1]
    weight_idxs = range(weights_start,weights_end)
    weights = [coefs[i] for i in weight_idxs]
    maxi = 0
    maxv = 0
    for i in xrange(len(weights)):
        if (abs(weights[i]) >= abs(weights[maxi])):
            maxi = i
            maxv = weights[i]
    return maxv

def get_top_10(weights,X_header):
    assert(len(weights) == len(X_header))
    l = [(weights[i],X_header[i]) for i in xrange(len(X_header))]
    l = sorted(l,key= lambda x: -abs(x[0]))
    print "Top 10 in descending order:"
    print l[:10]
    print '\n'.join([str(round(e[0],5)) + '&' + e[1]+'\\\\\\hline' for e in l[:10]])

encoder = joblib.load('one-hot-encoder.pk')

X_num = joblib.load('X_num-jl.pk')
print "X_num dims:"
print_dims(X_num)

X_cat_pre_one_hot_encoding = joblib.load("X_cat_pre_one_hot_encoding-jl.pk")
print "X_cat type:" + str(type(X_cat_pre_one_hot_encoding))
print_dims(X_cat_pre_one_hot_encoding)

X_cat_post_one_hot_encoding = joblib.load("X_cat_one_hot_encoded-jl.pk")
print "X_cat type:" + str(type(X_cat_post_one_hot_encoding))
print X_cat_post_one_hot_encoding.shape
#print_dims(X_cat_post_one_hot_encoding)

X_header = joblib.load('X_header-jl.pk')[:-1]
print "X_header type:" + str(type(X_header)) + ", " + str(len(X_header))



y_names = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']
for i in xrange(6):
    fitter = joblib.load('elasticnet-regressor-X-trainval-' + y_names[i] + '.pk')
    print y_names[i]
    model = fitter.clf.best_estimator_
    coefs = model.coef_
    print "Coefs shape: " + str(coefs.shape)

    num_out = len(X_num[0]) + len(X_cat_pre_one_hot_encoding[0])
    weights = [feature_to_sum_of_weights(encoder,i,coefs) for i in xrange(num_out)]
    #print weights
    #print len(weights)
    get_top_10(weights,X_header)

#Author: William Hinthorn
#Class: COS 424

# from ast import literal_eval as make_tuple
import math
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
# from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LassoLarsCV
# from sklearn.metrics import roc_curve, auc
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import  GridSearchCV, cross_val_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, ExtraTreesRegressor
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold,  RFECV, RFE
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from sklearn import linear_model, decomposition, preprocessing
from minepy import MINE
import sys
import re

global debug
debug = False


def select(background, train, train_label):
    ''' Selects the appropriate subests of features and targets s.t. there exit no N/A
        train_label = e.g. materialHardship, eviction, all, etc.'''
    if train_label == 'all':
        subTrain = train
        # ind = pd.isnull(subTrain).all(1).nonzero()[0]
        # subTest = subTrain.loc[ind, :]
        subTrain.dropna(axis=0, subset=['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining'], inplace=True, how='all')
    else:
        #subTrain = train[['challengeID', train_label]]
        subTrain = train.loc[:, ('challengeID', train_label)]
        # ind = pd.isnull(subTrain).any(1).nonzero()[0]
        # subTest = subTrain.loc[ind, :]
        subTrain.dropna(axis=0, subset=[train_label], inplace=True, how='all')

    #subBackground = select x in backgroun s.t  background['challengeID'] in subTrain['challengeID']
    #select challengeID in features that have a match in training labels
    subBackground = background.loc[background['challengeID'].isin(subTrain['challengeID'])]
    backTest = background.loc[~background['challengeID'].isin(subTrain['challengeID'])]
    #reverse check
    subTrain = subTrain.loc[subTrain['challengeID'].isin(subBackground['challengeID'])]
    subBackground = subBackground.sort_values(['challengeID'], ascending=True)
    subTrain = subTrain.sort_values(['challengeID'], ascending=True)
    subTest = pd.concat([background['challengeID'],background['challengeID'], pd.Series(np.nan)], axis=1, keys=['challengeID1', 'challengeID', train_label])
    subTrainBroad = subTrain.set_index('challengeID')
    subTest = subTest.set_index('challengeID1')
    subTest.update(subTrainBroad, join='left', overwrite=True)
    # if debug:
    #     print "original shape"
    #     print sorted(subTest['challengeID'], reverse=True)
    print "ba drop"
    print subTest.shape
    subTest = subTest[pd.isnull(subTest[train_label])]
    subTest = subTest.sort_values(['challengeID'], ascending=True)
    print subTest.shape
    return subBackground, subTrain, backTest, subTest

def testModel(X_scaled, Y):
    '''Test the current dataset using a number of regressors'''
    # Y = Y / np.max(Y)
    meanSquared = (Y[:, 1] - np.full(len(Y[:, 1]), np.mean(Y[:, 1]))) ** 2
    std = np.std(meanSquared)
    print "Baseline Error:\t %0.4f (+/- %0.4f)" % (-np.sum(meanSquared) / Y.shape[1], std * 2)
    print "Testing on %d features" % (X_scaled.shape[1])

    regressors = {
        ('linear', linear_model.LinearRegression()),
        ('ElasticNet', linear_model.ElasticNet(alpha=0.3, l1_ratio=0.1))#, #from previous one
        # ('RandomForest', RandomForestRegressor(n_jobs=-1, n_estimators=40, verbose=True))
    }

    for (name, reg) in regressors:
        cross_val = cross_val_score(reg, X_scaled, Y[:, 1], cv=5, scoring="neg_mean_squared_error")
        print '%s:\tcvs Error: %0.4f (+/- %0.4f) ' % (
            name, cross_val.mean(), cross_val.std() * 2)  # mean & 95% conf interval for k-folds

def predictScores(X_scaled, Y, X_test, Y_test, test_label):
    reg = linear_model.ElasticNet(alpha=0.3, l1_ratio=0.1)
    print "Predicting Scores"
    reg.fit(X_scaled, Y[:, 1])
    predictions = reg.predict(X_test)
    if debug:
        print Y
        print Y_test
        print Y.shape
        print Y_test.shape
        print "Num Predictions"
        print len(predictions)
        print np.count_nonzero(~np.isnan(Y[:, 1]))
        print np.count_nonzero(~np.isnan(predictions))
        print "Sum = %d" % (np.count_nonzero(~np.isnan(Y[:, 1])) + np.count_nonzero(~np.isnan(predictions)))
    Y_test[:, 1] = predictions
    # temp = pd.DataFrame(Y_test[:, 0], columns=['temp'])
    # temp = temp.rename(columns={"challengeID": "temp"})
    size= Y_test.shape[0] + Y.shape[0]
    screen = pd.concat([pd.DataFrame(np.arange(size)+1, columns=['temp']),
                        pd.DataFrame(np.full(size, np.nan), columns=[test_label])], axis=1)

    Y_test = pd.DataFrame(Y_test, columns=['challengeID', test_label]).set_index('challengeID')

    # Y_test = pd.concat([temp, Y_test], axis=1)
    # Y_test = Y_test.set_index('temp')
    screen = screen.set_index('temp')
    Y = pd.DataFrame(Y, columns=['challengeID', test_label]).set_index('challengeID')
    print 'uniques'
    # print np.sum(~Y_test['challengeID'].isin(Y['challengeID']))
    # print np.sum(~Y_['challengeID'].isin(Y_test['challengeID']))
    # print np.count_nonzero(np.isnan(screen[test_label]))
    screen.update(Y, join='left', overwrite=False)
    # print np.count_nonzero(np.isnan(screen[test_label]))
    screen.update(Y_test, join='left', overwrite=False)
    # print np.count_nonzero(np.isnan(screen[test_label]))

    if debug:
        print screen
    outputf = open('./ffc/' + test_label + '_guessed.csv', 'w')
    screen.to_csv(outputf, index=False, float_format='%10.5f')
    # np.savetxt('./ffc/' + test_label + '_guessed.csv', Y_test, delimiter=",", fmt='%10.5f')


def printSizes(str, X_scaled, Y, X_test):
    print "Sizes in %s" % str
    print X_scaled.shape
    print Y.shape
    print X_test.shape

def main(argv):
    #global train_background, train_outcomes
    # Process arguments
    path = ''
    usage_message = 'Usage: \n python classifySentiment.py -p <path> -i <inputfile> -s <pca> -l <randomLasso> -f <randomForest>' \
                    ' -d <debug> -c <column> -v <varThresh> -j <randomProjections>, -e <recFeatureElim> -o <oneHotEnc>' \
                    '-m <minimize>, -Q <squash>'
    inputf = "output.csv"
    train_label = 'gpa'
    varThresh = False
    univar = False
    pcaSelect = False
    rProjectSelect = False
    rForestSelect = False
    lassoSelect = False
    expandOhe = False
    rec_Feature_Elimination = False
    scaleY = False
    num_feat = 500
    parr = False
    global debug
    try:
        opts, args = getopt.getopt(argv, "p:i:d:c:v:u:f:s:l:j:e:o:m:Q",
                                   ["path=", "inputf=",  "column=", "varThresh=", "univar=", "rfe=", "oneHot="])
    except getopt.GetoptError:
        print usage_message
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print usage_message
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-i", "--path"):
            inputf = arg
        elif opt in ("-s", "--pca"):
            pcaSelect = True
        elif opt in ('-l', "--lasso"):
            lassoSelect = True
        elif opt in ('-f', '--forest'):
            rForestSelect = True
        elif opt in ("-d", "--debug"):
            debug = True
        elif opt in ("-c", "--column"):
            train_label = arg
        elif opt in ("-v", "--varThresh"):
            varThresh = True
            p = float(arg)
        elif opt in ("-u", "--univar"):
            univar = True
        elif opt in ("-j", "--rProject"):
            rProjectSelect = True
        elif opt in ("-e", "--rfe"):
            rec_Feature_Elimination = True
            num_feat = int(arg)
        elif opt in ("-o", "--onehot"):
            expandOhe = True
        elif opt in ("-m", "--minimize"): #standardize the output vector
            scaleY = True
        elif opt in ("-Q", "--squash"):
            parr = True

    # Get *preprocessed* data
    bg = open(path + "/" + "imputed_" + inputf, 'r')
    X = pd.read_csv(bg, low_memory=False)
    oc =  open(path + "/train.csv", 'r')
    Y = pd.read_csv(oc, low_memory=False)


    # Remove redundant ID's (mother and father)
    regex = re.compile('.*id[0-9].*')
    mf_ids = []
    for col in X.columns:
        if regex.match(col):
            mf_ids.append(col)
    X.drop(mf_ids, axis=1, inplace=True)
    X.drop(labels=['idnum'], axis=1, inplace=True)
    # print sorted(X['challengeID'])


    #Select only challengeid's in Y
    # drop all rows in background.csv that are not in train.csv
    X, Y, X_test, Y_test = select(X, Y, train_label)

    if debug:
        print "Original sizes"
        print X.shape
        print X_test.shape
        print Y.shape
        print Y_test.shape
    if scaleY:
        Y = preprocessing.scale(Y)
        print "SCALING Y"
        # Y = preprocessing.scale(Y_test)

    #Get the labels of the coefficients
    if not expandOhe:
        labels = X.axes[1]
    else:
        #Separate based on type of data:
        #First row bind the items
        if debug:
            print "Train and test shapes pre OHE"
            print X.shape
            print X_test.shape
        length = X.shape[0]
        X_floats = X.select_dtypes(include=['float64'])
        X_test_floats = X_test.select_dtypes(include=['float64'])
        X_ints = X[X.columns.difference(['challengeID'])].select_dtypes(include=['int64']) #leave out the 'challengeID'
        X_test_ints = X_test[X_test.columns.difference(['challengeID'])].select_dtypes(include=['int64'])

        #Assume integer data is categorical, apply one-hot encoding
        ohe = preprocessing.OneHotEncoder()
        X_full = np.concatenate((X, X_test), axis=0)
        X_full_ints = np.concatenate((X_ints, X_test_ints), axis=0)
        X_full_floats = np.concatenate((X_floats, X_test_floats), axis=0)
        mins = np.min(X_full_ints)
        X_full_ints -= mins # OHE needs only nonnegative integers.
        ohe.fit(X_full_ints)
        print "Transforming OHE"
        X_full_ints = ohe.transform(X_full_ints)
        X_full = np.concatenate((X_full_floats, X_full_ints.todense()), axis=1)
        X, X_test = np.split(X_full, [length], axis=0)
        #Split into the appropriate sets
        if debug:
            print "Train and test shapes post OHE"
            printSizes('ohe', X, Y, X_test)
        labels = np.arange(X.shape[1])


    #first 2 inputs are id's...
    X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    preVarSize =  X.shape[1]

    # Optionally eliminate columns of low variance
    if varThresh:
        thresh = p * (1 - p)
        X_scaled = X - np.min(X, axis=0)
        X_scaled = X_scaled / (np.max(X_scaled, axis=0) + 0.001)
        sel = VarianceThreshold(threshold=thresh)
        sel = sel.fit(X_scaled)
        X_scaled = sel.transform(X)
        X_scaled = preprocessing.scale(X_scaled)  # rescale'
        X_test = sel.transform(X_test)
        X_test = preprocessing.scale(X_test)
        if debug:
            print "X's Size pre fit"
            print X_scaled.shape
            print X_test.shape
            print "Y's Size prefit"
            print Y.shape
            print Y_test.shape

        labels = labels[np.where(sel.get_support())]
        if debug:
            if labels.shape[0] != X_scaled.shape[1]:
                print "labels wrong shape"
                print labels.shape
        postVarSize = X_scaled.shape[1]
        if X_scaled.shape[1] != X_test.shape[1]:
            print "Error Scaling Variances"
        print "Removed %d columns of Var < %f" % (preVarSize - postVarSize, thresh)
        print "New size is (%d, %d)" % (X_scaled.shape[0], X_scaled.shape[1])
    else:
        X_scaled = preprocessing.scale(X)
        X_test = preprocessing.scale(X_test)





    # Try a randomized lasso to pick most stable coefficients.
    def rLasso(X_scaled, Y, labels, X_test):
        print "Features sorted by their score for Randomized Lasso:"
        scores = np.zeros(X_scaled.shape[1])
        alphas = [0.003, 0.002]#, 0.001]
        for i in alphas:
            a = i
            print "Trying alpha %f" % (a)
            randomized_lasso = linear_model.RandomizedLasso(n_jobs=1, alpha=a, sample_fraction=0.25, verbose=True)
            printSizes('rlasso', X_scaled, Y, X_test)
            randomized_lasso.fit(X_scaled, Y[:, 1])
            scores = scores + randomized_lasso.scores_
            if debug:
                for score, label in sorted(zip(map(lambda x: round(x, 6), randomized_lasso.scores_),
                                 labels), reverse=True):
                    if score > 0.015:
                        print "%s: %f" % (label, score)


        scores = scores / len(alphas) # get mean values
        meanImportance = np.mean(scores)
        print "Average score for variable = %f" % (meanImportance)
        if meanImportance > 0.00001:
            if X_scaled.shape[1] > 100:
                thresh = 1.0
            else:
                thresh = 1.0
            keptIndices = np.where(scores > thresh * meanImportance)
            print "Top Scores for Random Lasso"
            if debug:
                for (score, label) in sorted(zip(scores,labels),key=lambda(score, label): score,  reverse=True):
                    if score > meanImportance:
                        print "%s: %f" % (label, score)

            printSizes('rlassoBeforeCut', X_scaled, Y, X_test)
            labels = labels[keptIndices]
            X_scaled = np.squeeze(X_scaled[:, keptIndices])
            X_test = np.squeeze(X_test[:, keptIndices])
            printSizes('rlassoAfterCut', X_scaled, Y, X_test)
        else:
            print "Not useful, aborting"
        print "New size of X"
        print X_scaled.shape
        return (X_scaled, Y, labels, X_test)



        # Try a randomized lasso to pick most stable coefficients.

    def lasso_stability(X_scaled, Y, labels, X_test):
        print "Features sorted by their stability score using lasso stability paths:"
        if debug:
            print X_scaled.shape
            alpha_grid, scores_path = linear_model.lasso_stability_path(X_scaled, Y[:, 1], n_jobs = -1, random_state=42,
                                                       eps=0.05, sample_fraction=0.50, verbose=debug)
            plt.figure(num=1)
            #plot as a function of the alpha/alpha_max
            variables = plt.plot(alpha_grid[1:] ** 0.333, scores_path.T[1:], 'k')
            ymin, ymax = plt.ylim()
            plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
            plt.ylabel('Stability score: proportion of times selected')
            plt.title('Stability Scores Path')
            plt.axis('tight')
            plt.figure(num=2)
            auc = (scores_path.dot(alpha_grid))
            auc_plot = plt.plot((scores_path.dot(alpha_grid)))
            plt.xlabel(r'Features')
            plt.ylabel(r'Area under stability curve')
            plt.title('Overall stability of features')
            plt.show()
            if X_scaled.shape[1] > 500:
                k = X_scaled.shape[1] / 3
            else:
                k = X_scaled.shape[1] / 2
            print "Top %d performing features" % (k)
            ind = np.argpartition(auc, -k)[-k:]
            for (arg, value) in sorted(zip(labels[ind], auc[ind]), key=lambda (x, y): y, reverse=True):
                print arg, value
            print ind
            print np.where(ind)
            labels = labels[np.where(ind)]
            X_scaled = np.squeeze(X_scaled[:, np.where(ind)])
            X_test = np.squeeze(X_test[:, np.where(ind)])
            printSizes('lasso_stability end', X_scaled, Y, X_test)

        else:
            print 'Debug option not set, supress plotting'
        return (X_scaled, Y, labels, X_test)

    #simple PCA reduction (not finished) Distorts the labels
    def pcaReduce(X_scaled, Y, labels, X_test):
        print "Reduction via Principle Component"
        pca = decomposition.PCA(svd_solver='randomized', n_components=X_scaled.shape[1]/2)
        pca.fit(X_scaled)
        if debug:
            print "Old Size of X"
            print X_scaled.shape
            print X_test.shape
        X_scaled = pca.transform(X_scaled)
        X_test = pca.transform(X_test)
        if debug:
            print "New size of X & X_test"
            print X_scaled.shape
            print X_test.shape
        # i = np.identity(X_scaled.shape[1])
        # coef = pca.transform(i)
        # labels = pd.DataFrame(coef, index=labels)
        if debug:
            if not expandOhe:
                print labels[:10]
            print X_scaled[:10, :10]
        labels = np.arange(X_scaled.shape[1]) # just serve as indices now
        return (X_scaled, Y, labels, X_test)

    def randomProject(X_scaled, Y, labels):
        '''Conduct a Gaussian random projection using Johnson Lindnenstrauss min dimension'''
        print "Reduction via Random Projections"
        transformer = GaussianRandomProjection(eps = 0.1)
        X_scaled = transformer.fit_transform(X_scaled)
        #minDim = transformer.n_component_
        print "Components" #% (minDim)
        print X_scaled.shape
        return (X_scaled, Y, labels)


    def extraTreesReduce(X_scaled, Y, labels, X_test):
        print "Reducing dimensionality through Extra Trees Regression"
        clf = ExtraTreesRegressor(n_jobs=-1, n_estimators=50, verbose=True)
        clf = clf.fit(X_scaled, Y[:, 1])
        meanImportance = np.mean(clf.feature_importances_)
        if X_scaled.shape[1] > 100:
            thresh = 1.15
        else:
            thresh = 1.0
        keptIndices = np.where(np.array(clf.feature_importances_) > thresh * meanImportance)
        print "Top Scores for Extra Trees"
        if debug:
            for thing in clf.feature_importances_:
                if thing > 1.50 * meanImportance:
                    print thing

        if not expandOhe:
            labels = labels[keptIndices]
        X_scaled = np.squeeze(X_scaled[:, keptIndices])
        X_test = np.squeeze(X_test[:, keptIndices])
        if debug:
            printSizes('AFter Extra tress reductin', X_scaled, Y, X_test)
        return (X_scaled, Y, labels, X_test)


        # Calculate the Maximal Information Coefficient

    def univarSelect(X_scaled, Y, labels):
        m = MINE()

        def MIC(x):
            m.compute_score(x, Y);
            return m.mic()
        newColumns = np.array(map(lambda x: MIC(x), X_scaled.T))
        print "Conducting Univariate MIC Trimming"
        toKeep = np.where(newColumns > 0.1)
        X_scaled = X_scaled[:, toKeep]
        if not expandOhe:
            labels = labels[toKeep]
        newColumns = newColumns[toKeep]
        scores = zip(labels, newColumns)
        print "Sorted Scores"
        print sorted(scores, key=lambda t: t[1], reverse=True)
        X_scaled = np.squeeze(X_scaled)
        print "New Shape"
        print X_scaled.shape
        return (X_scaled, Y, labels)


    def elasticCVParamTuning(X_scaled, Y, labels):
        '''Use to get Elastic Net params for final predictor'''
        print "Accuracy with Elastic Net"
        elastic = linear_model.ElasticNetCV(random_state=42, cv=6, l1_ratio=[0.01, .1, .5, .7, .9, .95, .99, 1], n_jobs=-1)
        elastic.fit(X_scaled, Y)
        # print elastic.mse_path_
        coef = np.array(elastic.coef_)
        scores = zip(labels, coef)
        print "CV Params"
        print elastic.alpha_
        print elastic.l1_ratio_
        if debug:
            for (key, val) in sorted(scores, key=lambda t: abs(t[1]), reverse=True)[:20]:
                print "%s: %f" % (key, val)

    def recFeatElim(X_scaled, Y, labels, model, num_feat=500):
        n_features = X_scaled.shape[1]
        rfe = RFE(estimator=model, n_features_to_select=num_feat, step=300, verbose=debug)
        rfe = rfe.fit(X_scaled, Y)
        if debug:
            print "Recursive Feature Elimination Values"
            print rfe.support_
            print rfe.ranking_
            print labels[np.argmin(rfe.ranking_)]
        X_scaled = X_scaled[:, rfe.support_]
        labels = labels[np.where(rfe.support_)]
        return (X_scaled, Y, labels)


    if univar:
        (X_scaled, Y, labels) = univarSelect(X_scaled, Y, labels)
    if pcaSelect:
        (X_scaled, Y, labels, X_test) = pcaReduce(X_scaled, Y, labels, X_test)
    if rProjectSelect:
        (X_scaled, Y, labels) = randomProject(X_scaled, Y, labels)

    # testModel(X_scaled, Y)
    X_full = np.arange(1)
    labels_parr = np.arange(1)
    if lassoSelect:
        if parr and pcaSelect:
            X_full = np.copy(X_scaled)
            (X_parr, Y, labels_parr, X_test) = lasso_stability(X_scaled, Y, labels, X_test)
            (X_parr, Y, labels_parr, X_test) = rLasso(X_parr, Y, labels_parr, X_test)
        else:
            (X_scaled, Y, labels, X_test) = lasso_stability(X_scaled, Y, labels, X_test)
            (X_scaled, Y, labels, X_test) = rLasso(X_scaled, Y, labels, X_test)

    testModel(X_scaled, Y)
    if rForestSelect:
        (X_scaled, Y, labels, X_test) = extraTreesReduce(X_scaled, Y, labels, X_test)
    if rec_Feature_Elimination:
        (X_scaled, Y, labels) = recFeatElim(X_scaled, Y, labels,
                                            RandomForestRegressor(), num_feat)
    if parr and pcaSelect:
        #Add variables selected by recFeatElim to lasso select
        X_scaled =X_full[:, np.unique(np.concatenate([labels, labels_parr]))]


    #elasticCVParamTuning(X_scaled, Y, labels)
    testModel(X_scaled, Y)
    # print labels
    # if expandOhe or pcaSelect:
    #     print X_test.shape
    #     X_test = X_test[:, labels]
    #     print X_test.shape
    #     print X_test
    # else:
    #     X_test = X_test[:, labels] #need to figure how to use textual labels

    print "Dimensions"
    print X_scaled.shape
    print Y.shape
    print X_test.shape
    print Y_test.shape
    predictScores(X_scaled, Y, X_test, Y_test, train_label)
    print "Exitting"
    exit()


if __name__ == "__main__":
    main(sys.argv[1:])
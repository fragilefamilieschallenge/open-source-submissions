#!/usr/bin/env python

from os.path import join
import time

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


base_dir = "data"
background_file = join(base_dir, "background.out.csv")
label_file = join(base_dir, "train.out.csv")
orig_label_file = join(base_dir, "train.csv")
out_file = "andrew_prediction.csv"

name_to_model = {
  #'bnb': lambda: BernoulliNB(),
  #'mnb': lambda: MultinomialNB(),
  #'svm': lambda: make_svm_model(),
  #'dt': lambda: make_dt_model(),
  'rf': lambda: make_rf_model(),
  #'gbc': lambda: make_gbc_model(),
  #'knn': lambda: KNeighborsClassifier()
}

n_folds = 5
chi2_fraction = 1
max_features = 10000
just_training = False
label_for_training = "jobTraining"
model_for_writing = name_to_model.iteritems().next()[0]

def main():
  print "Reading in %s..." % background_file
  background_df = pd.read_csv(background_file, low_memory=False)
  print "Reading in %s..." % label_file
  label_df = pd.read_csv(label_file, low_memory=False)
  print "Reading in %s..." % orig_label_file
  orig_label_df = pd.read_csv(orig_label_file, low_memory=False)

  if just_training:
    do_the_training(label_df, background_df, label_for_training)
  else:
    write_to_csv(label_df, orig_label_df, background_df, model_for_writing)

def make_dt_model():
  param_grid = [{'min_samples_leaf':[1, 10, 100], 'max_features':[0.5, 0.75, 0.9]}]
  return GridSearchCV(DecisionTreeClassifier(), param_grid)

def make_rf_model():
  param_grid = [{'n_estimators':[50, 100], 'min_samples_leaf':[10, 25], 'max_features':[0.75, 'auto']}]
  return GridSearchCV(RandomForestClassifier(), param_grid)

def make_gbc_model():
  param_grid = [{'n_estimators':[50, 100], 'learning_rate':[1.0], 'min_samples_leaf':[10, 25], 'max_features':[0.75, 'auto']}]
  return GridSearchCV(GradientBoostingClassifier(), param_grid)

def make_svm_model():
  param_grid = [{'C':[0.1, 1, 10, 100]}]
  return BaggingClassifier(GridSearchCV(LinearSVC(), param_grid), n_jobs=4)

def get_label(df, label_name):
  '''
  Extract the specified label from the given dataframe.
  :return a numpy array with a single column for the label.
  '''
  labels = df[label_name].as_matrix().tolist()
  labels = [1 if label else 0 for label in labels]
  return np.array(labels)

def get_matching_rows(background_df, challenge_ids):
  '''
  :return a smaller background_df with rows matching the specified challenge IDs.
  '''
  return background_df[background_df["challengeID"].isin(challenge_ids)]

def get_keep_indices(background_df, train_labels, challenge_ids, fraction = 1, max_features = 10000):
  '''
  :return indices corresponding to features to keep
  '''
  k = min(int(background_df.shape[1] * fraction), max_features)
  background_df_for_training = get_matching_rows(background_df, challenge_ids)
  select = SelectKBest(lambda X, y: np.array([-1 * t for t in chi2(X, y)[0].tolist()]), k)
  select.fit(background_df_for_training.values, train_labels)
  keep_indices = np.where(select.get_support())[0].tolist()
  drop_indices = list(set(range(background_df.shape[1])) - set(keep_indices))
  if len(drop_indices) > 0 and len(drop_indices) < 100:
    print "  - Dropping these columns: %s" % background_df.columns[drop_indices].values.tolist()
  return keep_indices

def print_result(label_pairs, elapsed):
  '''
  Convenience method to print the test result of binary classification.
  :param a list of (predicted_labels, expected_labels) pairs, one for each fold
  '''
  accuracy, roc_auc, f1, precision, recall = [], [], [], [], []
  num_labels = len(label_pairs[0][1])
  for (predicted_labels, expected_labels) in label_pairs:
    assert len(predicted_labels) == len(expected_labels)
    num_correct = 0 
    rounded_predicted_labels = [round(l) for l in predicted_labels]
    for i in range(len(predicted_labels)):
      if rounded_predicted_labels[i] == expected_labels[i]:
        num_correct += 1
    accuracy += [float(num_correct) * 100 / len(expected_labels)]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(expected_labels, predicted_labels)
    roc_auc += [auc(false_positive_rate, true_positive_rate)]
    f1 += [f1_score(expected_labels, rounded_predicted_labels)]
    precision += [precision_score(expected_labels, rounded_predicted_labels)]
    recall += [recall_score(expected_labels, rounded_predicted_labels)]
  # Take the average of the things
  accuracy = float(sum(accuracy)) / len(accuracy)
  roc_auc = float(sum(roc_auc)) / len(roc_auc)
  f1 = float(sum(f1)) / len(f1)
  precision = float(sum(precision)) / len(precision)
  recall = float(sum(recall)) / len(recall)
  num_correct = int(num_labels * accuracy / 100)
  print "\n================================"
  print "You guessed %s/%s = %.3f%% correct." % (num_correct, num_labels, accuracy)
  #print "  - False positive rate: %s" % false_positive_rate.tolist()
  #print "  - True positive rate: %s" % true_positive_rate.tolist()
  #print "  - Thresholds: %s" % thresholds.tolist()
  print "  - AUC: %s" % roc_auc
  print "  - Precision: %s" % precision
  print "  - Recall: %s" % recall
  print "  - F1: %s" % f1
  print "  - Time (s): %.3f" % elapsed
  print "================================\n"

def do_the_training(label_df, background_df, label_name):
  for model_name in name_to_model:
    print "Running model '%s'..." % model_name
    start_time = time.time()
    kf = KFold(n_splits = n_folds)
    result = []
    for i, (train_indices, test_indices) in enumerate(kf.split(label_df)):
      # NOTE: let's just forget CV for now
      #if i > 0:
      #  break
      fold_start_time = time.time()
      train_df = label_df.transpose()[train_indices].transpose()
      train_df = train_df[~train_df[label_name].isnull()]
      test_df = label_df.transpose()[test_indices].transpose()
      test_df = test_df[~test_df[label_name].isnull()]
      train_labels = get_label(train_df, label_name)
      test_labels = get_label(test_df, label_name)
      train_challenge_ids = set(train_df["challengeID"].values.tolist())
      test_challenge_ids = set(test_df["challengeID"].values.tolist())
      # Do the chi2 reduction
      feature_keep_indices = get_keep_indices(\
        background_df, train_labels, train_challenge_ids, chi2_fraction, max_features)
      train_features = get_matching_rows(background_df, train_challenge_ids)[feature_keep_indices]
      test_features = get_matching_rows(background_df, test_challenge_ids)[feature_keep_indices]
      model = name_to_model[model_name]()
      model.fit(train_features.values, train_labels)
      if type(model) is GridSearchCV:
        print "  - Best hyperparameters were: %s" % model.best_params_
      #predicted_labels = [float(model.predict_proba([tf])[0][1]) for tf in test_features.values]
      predicted_labels = [model.predict([tf])[0] for tf in test_features.values]
      expected_labels = test_labels.tolist()
      result += [(predicted_labels, expected_labels)]
      (num_samples, num_features) = train_features.shape
      print "  - Finished running on fold %i, used %s features, %s training samples, took %s s" %\
        (i, num_features, num_samples, time.time() - fold_start_time)
    elapsed = time.time() - start_time
    print_result(result, elapsed)

def write_to_csv(label_df, orig_label_df, background_df, model_to_use):
  max_challenge_id = max(background_df["challengeID"].as_matrix().tolist())
  result = {}
  continuous_labels = ["gpa", "grit", "materialHardship"]
  boolean_labels = ["eviction", "layoff", "jobTraining"]
  start_time = time.time()
  # Just fill these with all zeros for now
  for label_name in continuous_labels:
    result[label_name] = [0.0] * max_challenge_id
  # Predict boolean labels
  for label_name in boolean_labels:
    result[label_name] = [None] * max_challenge_id
    train_df = label_df[~label_df[label_name].isnull()]
    train_labels = get_label(train_df, label_name)
    train_challenge_ids = train_df["challengeID"].values.tolist()
    train_challenge_ids_set = set(train_challenge_ids)
    # Do the chi2 reduction
    feature_keep_indices = get_keep_indices(\
      background_df, train_labels, train_challenge_ids_set, chi2_fraction, max_features)
    train_features = get_matching_rows(\
      background_df, train_challenge_ids_set)[feature_keep_indices]
    model = name_to_model[model_to_use]()
    print "Training model %s for label '%s' using %s features and %s samples..." %\
      (model_to_use, label_name, train_features.shape[1], train_features.shape[0])
    model.fit(train_features.values, train_labels)
    if type(model) is GridSearchCV:
      print "  - Best hyperparameters were: %s" % model.best_params_
    print "Predicting missing values for '%s'..." % label_name
    for i, row in background_df.iterrows():
      challenge_id = int(row["challengeID"])
      result_index = challenge_id - 1
      features = row[feature_keep_indices].values
      predicted_label = round(model.predict_proba([features])[0][1]) == 1
      result[label_name][result_index] = predicted_label
    # What is the training error?
    expected_labels = []
    predicted_labels = []
    for i, row in train_df.iterrows():
      challenge_id = int(row["challengeID"])
      result_index = challenge_id - 1
      expected_labels += [1 if row[label_name] else 0]
      predicted_labels += [result[label_name][result_index]]
    elapsed = time.time() - start_time
    print_result([(predicted_labels, expected_labels)], elapsed)

  # Write our prediction CSV
  print "Writing out predictions to %s..." % out_file
  with open(out_file, "w") as writer:
    header = ["challengeID"] + continuous_labels + boolean_labels
    header = ",".join(["\"%s\"" % h for h in header])
    writer.write(header + "\n")
    for i in range(max_challenge_id):
      challenge_id = i + 1
      line = [challenge_id]
      line += [result[label_name][i] for label_name in continuous_labels]
      line += [result[label_name][i] for label_name in boolean_labels]
      line = ",".join([str(x) for x in line])
      writer.write(line + "\n")


if __name__ == "__main__":
  main()


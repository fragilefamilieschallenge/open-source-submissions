# -------------------------------------------------------------------------------------------------
# COS 424 
# Fragile Families 
#
# knnImputation.R
# ----------------------------------------------------
# Authors: Viola Mocz (vmocz) & Sonia Hashim (shashim)
#
# Description: Performs simple imputation on data set - removes dates and idnum column and
# replaces missing values with the weighted average of K-Nearest Neighbors (k = 10) 
# in each col. Makes minor adjustments - if column has only one value other than NA, will
# turn feature into a binary feature. 
# -------------------------------------------------------------------------------------------------

# JUST IN CASE:  sudo ln -f -s $(/usr/libexec/java_home)/jre/lib/server/libjvm.dylib /usr/local/lib
library(FSelector)
library(VIM)

# function: f_fewVals(df)
# ----------------------------------------------------
# description: Eedit entries with 1 val. or NA to become binary cols. where 1 represents the value 
# and 0 NA. Also removes cols. with only one value
f_fewVals <- function(data) {
  ncols = ncol(data)
  to_remove <- integer(ncols)
  
  # Remove cols that have all missing vals 
  all_na <- sapply(data, function(x)all(is.na(x)))
  to_remove[all_na == TRUE] <- 1
  
  # For each col...
  for (i in 1:ncols) {
    vals = unique(data[i])
    n_distinct_vals = nrow(vals)
    # Edit entries with 1 val. or NA to binary data
    if (n_distinct_vals == 2) {
      # Find value that is not NA 
      if (is.na(vals[1,1])) {
        v = vals[2,1]
      } else {
        v = vals[1,1]
      }
      # Set all NA to 0 and all values to 1 
      data[i][is.na(data[i])] <- 0
      data[i][data[i] == v] <- 1
    } 
    # Remove cols. with only one value 
    if (n_distinct_vals == 1) {
      to_remove[i] = 1 
    }
  }
  data[to_remove == 1] <- NULL
  return(data)
}

# function: f_nacols(df)
# ----------------------------------------------------
# description: prints list of colnames where cols contain NA. Used for debugging! 
# attribution: http://stackoverflow.com/questions/10574061/show-columns-with-nas-in-a-data-frame
f_nacols <- function(df) {
  # df[df < 0] <- NA 
  colnames(df)[unlist(lapply(df, function(x) any(is.na(x))))]
}

# function: f_median(df)
# ----------------------------------------------------
# description: simple median imputation 
# attribution: adapted from script provided by COS 424 TA's - thanks! 
f_median <- function(data) {
  ncols = ncol(data)
  to_remove = integer(ncols)
  for (i in 1:ncols){
    x = data[,i]
    pos_x <- x[x>0]
    median <- median(pos_x, na.rm = TRUE)
    if (is.na(median) || median < 0) {
      to_remove[i] = 1
    } else {
      x[x < 0 | is.na(x)] <- median 
      data[,i] <- x
    }
  }
  data[to_remove == 1] <- NULL
  return(data)
}

# -------------------------------------------------------------------------------------------------
# Feature Selection Functions 
# -------------------------------------------------------------------------------------------------

# function: f_lowInfo(df)
# ----------------------------------------------------
# description: remove cols with > 60% missing data 
f_lowInfo <- function(data) {
  ncols = ncol(data)
  nrows = nrow(data)
  to_remove = integer(ncols)
  for (i in 1:ncols) {
    if (sum(is.na(data[i])) > .4*nrows) {
      to_remove[i] = 1
    }
  }
  data[to_remove == 1] <- NULL
  return(data)
}

# function: f_lowVar(df)
# ----------------------------------------------------
# description: remove features with variance < first quantile of variances 
f_lowVar <- function(data) {
  ncols = ncol(data)
  vars = integer(ncols)
  for (i in 1:ncols) {
    vars[i] = var(data[i], na.rm = TRUE)
  }
  q = unname(quantile(vars, 0.25))
  print(summary(vars))
  data[vars < q] <- NULL
  return(data)
}

# function: f_RFimportance(df)
# ----------------------------------------------------
# description: use Random Forest importance (1 - Mean Decrease in Accuracy) to 
#   perform feature selection
# parameters: outcome_var is the label of the indicator being used and 
#   nfeat is the number of features to threshold the data set at
f_RFimportance <- function(data, outcome_var, nfeat) {
  combined_data = cbind(data, label=outcome_var)
  impt = random.forest.importance(outcome_var~., combined_data)
  feat = cutoff.k(impt, nfeat)
  ncols = ncol(data)
  to_keep = integer(ncols)
  col_names = colnames(data)
  for (i in 1:nfeat) {
    col = feat[i]
    j = which(colnames(data) == col)
    to_keep[j] = 1
  }
  
  data[to_keep == 0] <- NULL
  return(data)
}

# -------------------------------------------------------------------------------------------------
setwd("~/Desktop/Princeton/COS424/hw2")

# Imputation method: use knn 
fname = "knn"

# OPT 1 using feature engineering 
usingFeatEngineering = TRUE
# OPT 2 using feature selection - removing col with > 60% missing data and cutting off 
# first quartile of columns with low variance
usingFS_Var = TRUE 
# OPT 3 using feature selection - random forest importance, importance = mean decrease
# in accuracy of RF when the feature is removed 
usingFS_RFImportance = FALSE

# Read in Fragile Families data 
train <- read.csv("fragilefamilieschallenge/train.csv",  na.strings = c("NA", "Other"))
if (!usingFeatEngineering) {
  data <- read.csv("fragilefamilieschallenge/background.csv",  na.strings = c("NA", "Other"))
  # Remove dates 
  data <- data[, sapply(data, class) != "factor"]
  # Remove cols. with no variance or all missing vals
  data <- f_fewVals(data)
} else {
  data <- read.csv("data/featEngData.csv",  na.strings = c("NA", "Other"))
  fname = paste0(fname,"_eng")
}

# Remove idnum
data[which(colnames(data) == "idnum")] <- NULL

## FEATURE SELECTION
if (usingFS_Var) {
  # Remove cols with > 60% missing data
  data <- f_lowInfo(data)
  # OPT FS_1  - Remove features with variance < first quantile of variances 
  data <- f_lowVar(data)
  fname = paste0(fname,"_fsvar")
}
if (usingFS_RFImportance) {
  fname = paste0(fname, "_fsRF")
}

## IMPUTATION: replace missing values with weighted average of K-Nearest Neighbors
# on account of computational efficiency, piece
data <- kNN(data, k=3)


data[data < 0] <- NA
data <- f_lowInfo(data)
data <- f_lowVar(data)
data <- kNN(data, k=3, trace=TRUE)



# Partition into labeled and unlabeled data
# Note: we only need to keep track of the labeled data 
has_label <- logical(nrow(data))
nrows_train = nrow(train)
for (i in 1:nrows_train) {
  id = train$challengeID[i]
  j = which(data$challengeID == id)
  has_label[j] = TRUE
}
labeled <- data[has_label,]

# outcome_var = c("gpa", "grit", "materialHardship", "eviction", "layoff", "jobTraining")
outcome_var = c("gpa", "grit", "materialHardship")
nvar = length(outcome_var)

# For every outcome variable... 
for (v in 1:nvar) {
  # Remove rows with missing data for indicator we're examining
  indicator = outcome_var[v]
  train_v <- train
  labeled_v <- labeled
  index = which(colnames(train) == indicator)
  c = train[,index]
  is_missing_train <- logical(nrow(train))
  is_missing_label <- logical(nrow(labeled))
  for (i in 1:nrows_train) {
    if(is.na(c[i])) {
      is_missing_train[i] = TRUE 
      id = train$challengeID[i]
      j = which(labeled$challengeID == id)
      is_missing_label[j] = TRUE 
    }
  }
  train_v = train_v[!is_missing_train,]
  labeled_v = labeled_v[!is_missing_label,]
  
  ## Feature Selection - Random Forest Importance 
  if (usingFS_RFImportance) {
    if (indicator == "gpa") {
      labeled_v <- f_RFimportance(labeled_v, train_v$gpa, 2000)
    }
    if (indicator == "grit") {
      labeled_v <- f_RFimportance(labeled_v, train_v$grit, 2000)
    }
    if (indicator == "grit") {
      labeled_v <- f_RFimportance(labeled_v, train_v$materialHardship, 2000)
    }
    
    v_feat <- colnames(labeled_v)
    data_v <- data[v_feat]
    
    v_data_file = paste0("data/",fname,"_",indicator,"_data.csv")
    write.csv(data_v, file = v_data_file, row.names = FALSE)
  }
  
  label_file = paste0("data/", fname,'_', indicator, "_labeled.csv")
  train_file = paste0("data/", fname,'_', indicator, "_train.csv")
  
  write.csv(labeled_v, file = label_file, row.names = FALSE)
  write.csv(train_v, file = train_file, row.names = FALSE)
}

# Write to data/median_data.csv 
data_file = paste0("data/",fname,"_data.csv")
write.csv(data, file = data_file, row.names = FALSE)

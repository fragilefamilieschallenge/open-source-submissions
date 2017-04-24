#  -----------------------------------------------------------------  LIBRARIES ---------------------------------------------------------------

library(Amelia)
# JUST IN CASE:  sudo ln -f -s $(/usr/libexec/java_home)/jre/lib/server/libjvm.dylib /usr/local/lib
library(FSelector)

# -----------------------------------------------------------------  CLEANING DATA  -----------------------------------------------------------

data <- read.csv("background.csv",  na.strings = c("NA", "Other"))
# Removes dates 
data <- data[, sapply(data, class) != "factor"]


# Save challengeID and remove idnum, other ids
challengeID = data[,ncol(data)]
data[which(colnames(data) == "idnum")] <- NULL
data[which(colnames(data) == "challengeID")] <- NULL
data[which(colnames(data) == "mothid1")] <- NULL
data[which(colnames(data) == "mothid2")] <- NULL
data[which(colnames(data) == "mothid3")] <- NULL
data[which(colnames(data) == "mothid4")] <- NULL
data[which(colnames(data) == "hv3mothid3")] <- NULL
data[which(colnames(data) == "hv4mothid4")] <- NULL
data[which(colnames(data) == "fathid1")] <- NULL
data[which(colnames(data) == "fathid2")] <- NULL
data[which(colnames(data) == "fathid3")] <- NULL
data[which(colnames(data) == "fathid4")] <- NULL
data[which(colnames(data) == "hv3fathid3")] <- NULL
data[which(colnames(data) == "hv4fathid4")] <- NULL

#  ------------------------------------------------------- PRELIMINARY FEATURE SELECTION  ----------------------------------------------------

# function: f_fewVals(df)
# ----------------------------------------------------
# description: Edit entries with 1 val. or NA to become binary cols. where 1 represents the value 
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
data <- f_fewVals(data)

# Remove cols with > 50% missing data 
f_lowInfo <- function(data) {
  ncols = ncol(data)
  nrows = nrow(data)
  to_remove = integer(ncols)
  for (i in 1:ncols) {
    if (sum(is.na(data[i])) > .5*nrows) {
      to_remove[i] = 1
    }
  }
  data[to_remove == 1] <- NULL
  return(data)
}
data <- f_lowInfo(data)

# Remove features with variance < first quantile of variances 
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
data <- f_lowVar(data)

# Remove features that are highly correlated to each other, greater than 0.95
# Code taken from: http://stackoverflow.com/questions/18275639/remove-highly-correlated-variables

f_highCor <- function(data) {
  tmp <- cor(data)
  tmp[upper.tri(tmp)] <- 0
  diag(tmp) <- 0
  keep_col <- apply(tmp,2,function(x) all(abs(x)<=0.95, na.rm=TRUE))
  data <- data[,keep_col]
  return(data)
}
data <- f_highCor(data)
data_save <- data

#  ------------------------------------------------------- FEATURE ENGINEERING USING REMAINING FEATURES  -------------------------------------------------------


#  ----------------------------------------------------------------- CUTOFF FEATURE SELECTION  ----------------------------------------------------------------- 

# Import train
labels <- read.csv("train.csv")

# Use correlation to elimate features with either Spearman or Pearson correlation within epsilon of 0 
# NOTE: Using eviction as desired outcome variable 
f_RFimportance <- function(data, outcome_var) {
  combined_data = cbind(data, label = outcome_var)
  
  impt = random.forest.importance(label~., combined_data)
  nfeat = 100
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
data <- f_RFimportance(data, labels$eviction)

# Add back challenge ID 
data[,"challengeID"] <- challengeID

# Imputation! 
data.out <- amelia(data, m=10, idvars = c("challengeID"))
write.amelia(obj=data.out, file.stem = "amelia_data")

train <- read.csv("train.csv",  na.strings = c("NA", "Other"))
amelia1 <- read.csv("amelia_data1.csv")
amelia2 <- read.csv("amelia_data2.csv")
amelia3 <- read.csv("amelia_data3.csv")
amelia4 <- read.csv("amelia_data4.csv")
amelia5 <- read.csv("amelia_data5.csv")
amelia6 <- read.csv("amelia_data6.csv")
amelia7 <- read.csv("amelia_data7.csv")
amelia8 <- read.csv("amelia_data8.csv")
amelia9 <- read.csv("amelia_data9.csv")
amelia10 <- read.csv("amelia_data10.csv")

f_dataset <- function(data, train, name) {
  # Partition into labelled and unlabelled data
  has_label <- logical(nrow(data))
  nrows_train = nrow(train)
  for (i in 1:nrows_train) {
    id = train$challengeID[i]
    j = which(data$challengeID == id)
    has_label[j] = TRUE
  }
  labeled <- data[has_label,]
  unlabeled <- data[!has_label,]
  
  
  # Remove rows with missing data for indicator we're examining
  indicator = "eviction"
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
  train = train[!is_missing_train,]
  labeled = labeled[!is_missing_label,]
  
  #Remove first and last columns (more ids)
  data <- data[,-1]
  data <- data[,-101]
  labeled <- labeled[,-1]
  labeled <- labeled[,-101]
  unlabeled <- unlabeled[,-1]
  unlabeled <- unlabeled[,-101]
  
  # Write to data/median.csv and data/median_train.csv 
  write.csv(data, file = paste("data/", name, "_data.csv", sep=""), row.names = FALSE)
  write.csv(labeled, file = paste("data/", name, "_labeled.csv", sep=""), row.names = FALSE)
  write.csv(unlabeled, file = paste("data/", name, "_unlabeled.csv", sep=""), row.names = FALSE)
  write.csv(train, file = paste("data/", name, "_train.csv", sep=""), row.names = FALSE)
}

f_dataset(amelia1,train, "amelia1")
f_dataset(amelia2,train, "amelia2")
f_dataset(amelia3,train, "amelia3")
f_dataset(amelia4,train, "amelia4")
f_dataset(amelia5,train, "amelia5")
f_dataset(amelia6,train, "amelia6")
f_dataset(amelia7,train, "amelia7")
f_dataset(amelia8,train, "amelia8")
f_dataset(amelia9,train, "amelia9")
f_dataset(amelia10,train, "amelia10")

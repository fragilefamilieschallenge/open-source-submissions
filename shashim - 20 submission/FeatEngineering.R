# -------------------------------------------------------------------------------------------------
# COS 424 
# Fragile Families 
#
# Playground.R
# ----------------------------------------------------
# Authors: Viola Mocz (vmocz) & Sonia Hashim (shashim)
#
# Description: Space to develop in R. 
# -------------------------------------------------------------------------------------------------

setwd("~/Desktop/Princeton/COS424/hw2")

# Read in Fragile Families data 
data <- read.csv("fragilefamilieschallenge/background.csv",  na.strings = c("NA", "Other"))

# -------------------------------------------------------------------------------------------------
# Data Preprocessing 
# -------------------------------------------------------------------------------------------------
# Remove dates 
data <- data[, sapply(data, class) != "factor"]

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
data <- f_fewVals(data)


# -------------------------------------------------------------------------------------------------
# Feature Engineering Methods 
# -------------------------------------------------------------------------------------------------
f_engineerMean <- function(data) {
  ncols = ncol(data)
  cnames = colnames(data)
  seen = integer(ncols)
  # For every feature... 
  for (i in 1:ncols) {
    # If the feature has not been combined into an aggregate feature... 
    if (seen[i] == 0) {
      # Find all adjacent features (same name with the exception of the year)
      name = cnames[i]
      search_str = substr(name, 1, nchar(name)-1)
      matches = startsWith(cnames, search_str)
      seen[matches == TRUE] <- 1 
      # Don't engineer an id feature or a feature with no additional matches
      if (!grepl("id", name) && sum(matches[matches == TRUE]) > 1) {
        # Create and add the engineered feature 
        feat <- rowMeans(data[matches ==TRUE], na.rm = TRUE)
        data <- cbind(data, feat)
        # Rename the engineered feature 
        feat_name = paste0(search_str, "_mean")
        colnames(data)[ncol(data)] = feat_name
      }
      seen[i] = 1
    }
  }
  return(data)
}
data <- f_engineerMean(data)

f_engineerMaxPool <- function(data) {
  ncols = ncol(data)
  cnames = colnames(data)
  seen = integer(ncols)
  # For every feature... 
  for (i in 1:ncols) {
    # If the feature has not been combined into an aggregate feature... 
    if (seen[i] == 0) {
      # Find all adjacent features (same name with the exception of the year)
      name = cnames[i]
      search_str = substr(name, 1, nchar(name)-1)
      matches = startsWith(cnames, search_str)
      seen[matches == TRUE] <- 1 
      # Don't engineer an id feature or a feature with no additional matches
      if (!grepl("id", name) && sum(matches[matches == TRUE]) > 1) {
        # Create and add the engineered feature 
        feat <- apply(data[matches == TRUE], 1, max, na.rm = TRUE)
        feat[!is.finite(feat)] <- NA 
        data <- cbind(data, feat)
        # Rename the engineered feature 
        feat_name = paste0(search_str, "_maxPool")
        colnames(data)[ncol(data)] = feat_name
      }
      seen[i] = 1
    }
  }
  return(data)
}
data <- f_engineerMaxPool(data)

# Write to data/featengdata.csv 
write.csv(data, file = "data/featengdata.csv", row.names = FALSE)
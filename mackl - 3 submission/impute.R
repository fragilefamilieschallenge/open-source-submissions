

#########################
## MISSING DATA SCRIPT ##
## (to be used as a    ##
##  temporary fix)     ##
#########################

## Missing values in FF are coded
## -9 Not in wave – Did not participate in survey/data collection component
## -8 Out of range – Response not possible; rarely used
## -7 Not applicable (also -10/-14) – Rarely used for survey questions
## -6 Skipped “Valid skip” – Intentionally not asked question; question does not apply to respondent or response known based on prior information.
## -5 Not asked “Invalid skip” – Respondent not asked question in the version of the survey they received (present in two “pilot” cities).
## -3 Missing – Data is missing due to some other reason; rarely used
## -2 Don’t know – Respondent asked question; Responded “Don’t Know”.
## -1 Refuse – Respondent asked question; Refused to answer question.

## Each of these has a meaning. In particular, for -6,
## you should look in the questionnaire and try not to figure
## out why the respondent was skipped out of a question

## In general, it's best to deal with missing data in a
## problem-specific way, by thinking about what each vaue
## of missing means and using multiple imputation to fill
## in values that are invalidly missing.
## If you just need a fast workaround, the code below
## replaces any missing values with the most common
## observed value.

## WORKAROUND FOR MISSING DATA

## Pass a data frame to this function, and it will
## return the same data frame with missing values singly
## imputed with the mode, and with a new set of variables
## coded miss_`var' indicating whether each variable was missing.

clean.data <- function(data) {
  # Fixes date bug
  tmp <- as.numeric(data$cf4fint) - min(as.numeric(data$cf4fint))
  data$cf4fint <- tmp
  ## Create a data frame of indicators for missing
  missing <- data < 0
  ## Rename the columns of the missing indicators to start with miss_
  colnames(missing) <- paste("miss_",colnames(data),sep = "")
  ## Keep only the columns where some are missing
  missing <- missing[,apply(missing,2,var,na.rm = T) > 0]
  ## Replace missing values in the data frame with the mode
  ## (note: mode function adapted from http://stackoverflow.com/questions/2547402/is-there-a-built-in-function-for-finding-the-mode)
  filled.data <- lapply(
    data, function(x) {
      ## Replace negative values with NA for missing
      x[x<0] <- NA
      # ## Identify the unique values of that variable
      # ux <- unique(na.omit(x[x > 0]))
      # ## Find the mode
      # mode <- ux[which.max(tabulate(match(na.omit(x[x > 0]), ux)))]
      # ## Replace with the mode if missing
      # ## (note: using the mode since, unlike the median or mean,
      # ## it is defined even for factor variables)
      # if (is.na(mode)) mode <- 1
      # x[x < 0 | is.na(x)] <- mode
      return(x)
    }
  )
  ## Combine that with the indicators
  #filled.data <- cbind(as.data.frame(filled.data),missing)
  filled.data <- as.data.frame(filled.data)
  return(filled.data)
}

#setwd("/Users/macklee/Google Drive/Spring 2017/COS424/A2")
setwd("C:/Users/Mack/Google Drive/Spring 2017/COS424/A2")

raw <- read.csv(file = "background.csv", header = TRUE, sep = ",")


newraw <-clean.data(raw)


nonintegercol <- which(sapply(raw, typeof) == "logical")


numraw <-clean.data(raw)
#numraw <- data.frame(apply(newraw, 2, function(x) as.numeric(as.character(x))))
pMiss <- function(x){sum(is.na(x))/length(x)*100}
percentageMissing <- apply(numraw,2,pMiss)
sapply(numraw, function(x) sum(is.na(x)))
numraw <- numraw[,which(percentageMissing < 10)]
write.csv(numraw, file = "newbackground.csv")

library(mice)
completed<- mice(data = newraw, m=5, method = 'pmm')
completed <- mice(data = numraw, m=1, method = "pmm")


numbackground <- read.csv(file = "numbackground.csv", header = TRUE, sep =",")
start.time <- Sys.time()
completed <- mice(data = numbackground, m=5, method = "norm")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 
completedData <- complete(completed, 1)
write.csv(completedData, file = "completedBackground.csv")

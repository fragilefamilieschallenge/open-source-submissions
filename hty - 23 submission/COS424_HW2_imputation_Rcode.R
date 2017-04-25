rm(list = ls())
install.packages("missForest")
library(VIM)
library(mice)
library(Amelia)
library(missForest)
library(foreach)



##### read the data ########
Raw.Data = read.csv("/Users/leoniu/Google Drive/active files/Princeton University/2. Courses/COS 424/2. Homework/Assignment2/fragilefamilieschallenge/background.csv")
sum(is.na(Raw.Data))/prod(dim(Raw.Data))
Data = Raw.Data
z = complete.cases(Data)
sum(z)

##### Eliminate Features with all are missing data ##########
## threshold = 50%
MS.th = 0.5

missing.obs = apply(is.na(Data), 2, mean)
missing.fts = apply(is.na(Data), 1, mean)
par(mfrow = c(2,1))
hist(missing.obs, main = "Histogram of missing observations proportion for each feature", 
    xlab = "proportion of non-missing observations" )
hist(missing.fts, main = "Histogram of missing features proportion for each observation", 
     xlab = "proportion of non-missing features" )
sum(missing.obs > MS.th)
sum(missing.fts == 1)
#colnames(Data)[missing.obs==1]

Data = Data[, missing.obs < MS.th]

####  Eliminate Features with constant values for all observations  ###########

num.var = dim(Data)[2]
unique.var = list()
unique.na <- function(Data){
  unique(na.omit(Data))
}
unique.var = apply(Data, 2, unique.na)
unique.length = lapply(unique.var, length)
unique.length = unlist(unique.length)
unique.test = unique.length == 1
sum(unique.test)
Data = Data[, !unique.test]

##### Factorize Non-numerical Features ######

num.var = dim(Data)[2]
factor.index = NULL
for(i in 1:num.var){
  if(!is.numeric(Data[, i])){
    Data[, i] = factor(Data[, i])
    factor.index = c(factor.index, i)
  }
}
factor.names = names(Data)[factor.index]


#### Explore Perfect Collinearity ####
Data.copy = Data
num.var = dim(Data.copy)[2]
for(i in 1:num.var){
  if(!is.numeric(Data.copy[, i])){
    Data.copy[, i] = as.numeric(Data.copy[, i])
  }
}
Data.copy[is.na(Data.copy)] = 0
data.cor = cor(Data.copy)
diag(data.cor) = 0
data.cor[is.na(data.cor)] = 0
data.cor[lower.tri(data.cor)] = 0
sum(data.cor==1)
colin = rep(FALSE, dim(data.cor)[1])

for(i in 2: dim(data.cor)[1]){
  colin[data.cor[,i] == 1] = TRUE
}
sum(colin)

Data = Data[,!colin]


## Plot the Data ##

par(mfrow = c(1,1))
matrixplot(Data)


##### Data Imputation #########

######  Impute 1  : Single Imputation: median for numerical variable, mode for nominal variable.
######  Results Data.SI

Data.SI = Data
# Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

n.var = dim(Data.SI)[2]
for(i in 1:n.var){
  col.data = Data.SI[,i]
  if(sum(is.na(col.data))){
    if(is.numeric(col.data)){
      col.data[is.na(col.data)] = median(col.data, na.rm = TRUE)
    }else{
      col.data[is.na(col.data)] = getmode(na.omit(col.data))
    }
   
    Data.SI[,i] = col.data
  }
}

write.csv(Data.SI, file=  "/Users/leoniu/Google Drive/active files/Princeton University/2. Courses/COS 424/2. Homework/Assignment2/fragilefamilieschallenge/Data-SI.csv")



### We use threshold to filter out highly correlated data
CL.th = 0.9
Data.copy = Data.SI
num.var = dim(Data.copy)[2]
for(i in 1:num.var){
  if(!is.numeric(Data.copy[, i])){
    Data.copy[, i] = as.numeric(Data.copy[, i])
  }
}
data.cor = cor(Data.copy)
diag(data.cor) = 0
data.cor[is.na(data.cor)] = 0
data.cor[lower.tri(data.cor)] = 0
sum(data.cor==1)
colin = rep(FALSE, dim(data.cor)[1])

for(i in 2: dim(data.cor)[1]){
  colin[data.cor[,i] >= CL.th] = TRUE
}
sum(colin)

Data.SI = Data.SI[,!colin]


### Delete those nominal variables with more than 50 levels
LV.th = 30
num.var = dim(Data.SI)[2]
nominal.test = rep(TRUE, num.var)
for(i in 1:num.var){
  if(is.factor(Data.SI[, i])){
    if(length(levels(Data.SI[, i])) > LV.th){
      nominal.test[i] = FALSE
    }
  }
}
sum(nominal.test)
Data.SI = Data.SI[, nominal.test]


### Delete those nominal variables with low variance
VR.th = 0.1
num.var = dim(Data.SI)[2]
var.test = rep(TRUE, num.var)
for(i in 1:num.var){
  if(is.numeric(Data.SI[, i])){
    if(sd(Data.SI[, i]) / mean(abs(Data.SI[, i])) < VR.th){
      var.test[i] = FALSE
    }
  }
}
sum(var.test)
Data.SI = Data.SI[, var.test]



write.csv(Data.SI, file=  "/Users/leoniu/Google Drive/active files/Princeton University/2. Courses/COS 424/2. Homework/Assignment2/fragilefamilieschallenge/Data-SI-small.csv")
#Data.clean = read.csv("/Users/leoniu/Google Drive/active files/Princeton University/2. Courses/COS 424/2. Homework/Assignment2/fragilefamilieschallenge/background_single_impute.csv")


##### Impute 2: Using Mice Package

Data.MI = mice(Data, m=5, maxit = 5, method = "fastpmm")


##### Impute 3:  Using missForest
install.packages("doParallel")
library(doParallel)
cl <- makeCluster(10)
registerDoParallel(cl)

start.time <- Sys.time()
Data.MF = missForest(Data, parallelize = "variables")
end.time <- Sys.time()
end.time - start.time


##### Impute 4: Using Amelia Package
start.time <- Sys.time()
Data.AM = amelia(Data, m = 5, parallel = "multicore"
                     #, noms = factor.names[1]
)
end.time <- Sys.time()
end.time - start.time

##### Impute 5: Using mi Package
Data.mmii = mi(Data)

##### Impute 6: Using Hmisc Package
library(Hmisc)
Data.H = aregImpute(~., data = Data, n.impute  = 3)

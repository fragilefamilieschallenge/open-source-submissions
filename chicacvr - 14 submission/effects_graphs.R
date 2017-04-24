# Claudia V. Roberts
# COS 424
# Assignment 2

# Load in all the cross validation results from the 108 model combinations 
all_model_data = read.csv("all_results.csv", header = TRUE, stringsAsFactors = FALSE)

#Change column names
colnames(all_model_data) = c("Imputation", "Scaling", "FeatureSelection", "Model", "MeanMSE", "TrainingTime")

# Graph: Mean vs. Median
Linear_x_med = jitter(rep(1, 8))
Linear_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Linear Regression", 5]

Lars_x_med = jitter(rep(1, 8))
Lars_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Lars CV", 5]

Ridge_x_med = jitter(rep(1, 8))
Ridge_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_med = jitter(rep(1, 8))
Elastic_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_med = jitter(rep(1, 8))
Orthog_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_med = jitter(rep(1, 8))
Lasso_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Lasso CV", 5]

Tree_x_med = jitter(rep(1, 6))
Tree_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_med = jitter(rep(1, 8))
SVR_y_med = all_model_data[all_model_data$Imputation == "Imputer (median)" & all_model_data$Model == " Linear SVR", 5]

Linear_x_mode = jitter(rep(1.5, 8))
Linear_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Linear Regression", 5]

Lars_x_mode = jitter(rep(1.5, 8))
Lars_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Lars CV", 5]

Ridge_x_mode = jitter(rep(1.5, 8))
Ridge_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_mode = jitter(rep(1.5, 8))
Elastic_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_mode = jitter(rep(1.5, 8))
Orthog_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_mode = jitter(rep(1.5, 8))
Lasso_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Lasso CV", 5]

Tree_x_mode = jitter(rep(1.5, 6))
Tree_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_mode = jitter(rep(1.5, 8))
SVR_y_mode = all_model_data[all_model_data$Imputation == "Imputer (mode)" & all_model_data$Model == " Linear SVR", 5]

max(all_model_data[,5]) # -0.3628858
min(all_model_data[,5]) # -1.622771e+29

# Mean and median of the MSEs for mean and median
mean(all_model_data[all_model_data$Imputation == "Imputer (median)", 5]) # -1.190167e+24
median(all_model_data[all_model_data$Imputation == "Imputer (median)", 5]) # -0.4065199
mean(all_model_data[all_model_data$Imputation == "Imputer (mode)", 5]) # -2.617379e+27
median(all_model_data[all_model_data$Imputation == "Imputer (mode)", 5]) # -0.4008769

plot(Linear_x_med, Linear_y_med * -1, col="red", log = "y", xlim = c(.9, 1.6), ylim = c(.3, 4.2)
     , main = "Effect of Imputation Strategies on Mean MSE", xlab = "Imputation Method", ylab = "Mean MSE"
     , xaxt = "n", cex.main = .8, cex.lab = .8)
axis(1, labels = c("Median \n\n mean:1.19e+24 \n median:0.40", "Mode \n\n mean:2.61e+27 \n median:0.40"), 
     at = c(1, 1.5), cex.axis = .7)
legend("top", inset=0, title="Models",
       c("LR","LARSCV","RCV","ENCV", "OMPCV", "LassoCV", "DTR", "SVR"), 
       fill=c("red", "green", "blue", "orange", "purple", "gray", "chocolate4", "cyan4"), 
       horiz=TRUE, cex = .48)
points(Lars_x_med, Lars_y_med * -1, col="green")
points(Ridge_x_med, Ridge_y_med * -1, col="blue")
points(Elastic_x_med, Elastic_y_med * -1, col="orange")
points(Orthog_x_med, Orthog_y_med * -1, col="purple")
points(Lasso_x_med, Lasso_y_med * -1, col="gray")
points(Tree_x_med, Tree_y_med * -1, col="chocolate4")
points(SVR_x_med, SVR_y_med * -1, col="cyan4")
points(Linear_x_mode, Linear_y_mode * -1, col="red")
points(Lars_x_mode, Lars_y_mode * -1, col="green")
points(Ridge_x_mode, Ridge_y_mode * -1, col="blue")
points(Elastic_x_mode, Elastic_y_mode * -1, col="orange")
points(Orthog_x_mode, Orthog_y_mode * -1, col="purple")
points(Lasso_x_mode, Lasso_y_mode * -1, col="gray")
points(Tree_x_mode, Tree_y_mode * -1, col="chocolate4")
points(SVR_x_mode, SVR_y_mode * -1, col="cyan4")

# Graph: Standardization vs. No Scaling
Linear_x_noscale = jitter(rep(1, 8))
Linear_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Linear Regression", 5]

Lars_x_noscale = jitter(rep(1, 8))
Lars_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Lars CV", 5]

Ridge_x_noscale = jitter(rep(1, 8))
Ridge_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_noscale = jitter(rep(1, 8))
Elastic_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_noscale = jitter(rep(1, 8))
Orthog_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_noscale = jitter(rep(1, 8))
Lasso_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Lasso CV", 5]

Tree_x_noscale = jitter(rep(1, 6))
Tree_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_noscale = jitter(rep(1, 8))
SVR_y_noscale = all_model_data[all_model_data$Scaling == " No Scaling" & all_model_data$Model == " Linear SVR", 5]

Linear_x_stand = jitter(rep(1.5, 8))
Linear_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Linear Regression", 5]

Lars_x_stand = jitter(rep(1.5, 8))
Lars_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Lars CV", 5]

Ridge_x_stand = jitter(rep(1.5, 8))
Ridge_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_stand = jitter(rep(1.5, 8))
Elastic_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_stand = jitter(rep(1.5, 8))
Orthog_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_stand = jitter(rep(1.5, 8))
Lasso_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Lasso CV", 5]

Tree_x_stand = jitter(rep(1.5, 6))
Tree_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_stand = jitter(rep(1.5, 8))
SVR_y_stand = all_model_data[all_model_data$Scaling == " Standardize" & all_model_data$Model == " Linear SVR", 5]

# Mean and median of the MSEs for scaling and standardize 
mean(all_model_data[all_model_data$Scaling == " No Scaling", 5]) # -1.06119
median(all_model_data[all_model_data$Scaling == " No Scaling", 5]) # -0.4007757
mean(all_model_data[all_model_data$Scaling == " Standardize", 5]) # -2.618569e+27
median(all_model_data[all_model_data$Scaling == " Standardize", 5]) # -0.4097345

plot(Linear_x_noscale, Linear_y_noscale*-1, col="red", log = "y", xlim = c(.9, 1.6), ylim = c(.3, 4.2)
     , main = "Effect of Data Scaling on Mean MSE", xlab = "Scaling Method", ylab = "Mean MSE"
     , xaxt = "n", cex.main = .8, cex.lab = .8)
axis(1, labels = c("No Scaling \n\n mean:1.06 \n median:0.40", "Standardize \n\n mean:2.61e+27 \n median:0.40"), 
     at = c(1, 1.5), cex.axis = .7)
legend("top", inset=0, title="Models",
       c("LR","LARSCV","RCV","ENCV", "OMPCV", "LassoCV", "DTR", "SVR"), 
       fill=c("red", "green", "blue", "orange", "purple", "gray", "chocolate4", "cyan4"), 
       horiz=TRUE, cex = .48)
points(Lars_x_noscale, Lars_y_noscale*-1, col="green")
points(Ridge_x_noscale, Ridge_y_noscale*-1, col="blue")
points(Elastic_x_noscale, Elastic_y_noscale*-1, col="orange")
points(Orthog_x_noscale, Orthog_y_noscale*-1, col="purple")
points(Lasso_x_noscale, Lasso_y_noscale*-1, col="gray")
points(Tree_x_noscale, Tree_y_noscale*-1, col="chocolate4")
points(SVR_x_noscale, SVR_y_noscale*-1, col="cyan4")
points(Linear_x_stand, Linear_y_stand*-1, col="red")
points(Lars_x_stand, Lars_y_stand*-1, col="green")
points(Ridge_x_stand, Ridge_y_stand*-1, col="blue")
points(Elastic_x_stand, Elastic_y_stand*-1, col="orange")
points(Orthog_x_stand, Orthog_y_stand*-1, col="purple")
points(Lasso_x_stand, Lasso_y_stand*-1, col="gray")
points(Tree_x_stand, Tree_y_stand*-1, col="chocolate4")
points(SVR_x_stand, SVR_y_stand*-1, col="cyan4")

# Graph: Feature Selection 
Linear_x_nofeat = jitter(rep(1, 4))
Linear_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Linear Regression", 5]

Lars_x_nofeat = jitter(rep(1, 4))
Lars_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Lars CV", 5]

Ridge_x_nofeat = jitter(rep(1, 4))
Ridge_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_nofeat = jitter(rep(1, 4))
Elastic_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_nofeat = jitter(rep(1, 4))
Orthog_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_nofeat = jitter(rep(1, 4))
Lasso_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Lasso CV", 5]

SVR_x_nofeat = jitter(rep(1, 4))
SVR_y_nofeat = all_model_data[all_model_data$FeatureSelection == " No Feature Selection" & all_model_data$Model == " Linear SVR", 5]

Linear_x_mi = jitter(rep(1.5, 4))
Linear_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Linear Regression", 5]

Lars_x_mi = jitter(rep(1.5, 4))
Lars_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Lars CV", 5]

Ridge_x_mi = jitter(rep(1.5, 4))
Ridge_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_mi = jitter(rep(1.5, 4))
Elastic_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_mi = jitter(rep(1.5, 4))
Orthog_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_mi = jitter(rep(1.5, 4))
Lasso_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Lasso CV", 5]

Tree_x_mi = jitter(rep(1.5, 4))
Tree_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_mi = jitter(rep(1.5, 4))
SVR_y_mi = all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%" & all_model_data$Model == " Linear SVR", 5]

Linear_x_ften = jitter(rep(2, 4))
Linear_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Linear Regression", 5]

Lars_x_ften = jitter(rep(2, 4))
Lars_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Lars CV", 5]

Ridge_x_ften = jitter(rep(2, 4))
Ridge_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_ften = jitter(rep(2, 4))
Elastic_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_ften = jitter(rep(2, 4))
Orthog_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_ften = jitter(rep(2, 4))
Lasso_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Lasso CV", 5]

Tree_x_ften = jitter(rep(2, 4))
Tree_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_ften = jitter(rep(2, 4))
SVR_y_ften = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%" & all_model_data$Model == " Linear SVR", 5]

Linear_x_ftw = jitter(rep(2.5, 4))
Linear_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Linear Regression", 5]

Lars_x_ftw = jitter(rep(2.5, 4))
Lars_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Lars CV", 5]

Ridge_x_ftw = jitter(rep(2.5, 4))
Ridge_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Ridge CV", 5]

Elastic_x_ftw = jitter(rep(2.5, 4))
Elastic_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Elastic Net CV", 5]

Orthog_x_ftw = jitter(rep(2.5, 4))
Orthog_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Orthogonal Matching Pursuit CV", 5]

Lasso_x_ftw = jitter(rep(2.5, 4))
Lasso_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Lasso CV", 5]

Tree_x_ftw = jitter(rep(2.5, 4))
Tree_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Decision Tree Regressor", 5]

SVR_x_ftw = jitter(rep(2.5, 4))
SVR_y_ftw = all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%" & all_model_data$Model == " Linear SVR", 5]

# Mean and median of the MSEs for feature selection 
mean(all_model_data[all_model_data$FeatureSelection == " No Feature Selection", 5]) # -5.797973e+27
median(all_model_data[all_model_data$FeatureSelection == " No Feature Selection", 5]) # -0.4090777
mean(all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%", 5]) # -2.507303e+23
median(all_model_data[all_model_data$FeatureSelection == " Feature Selection Mutual Info Regression 20%", 5]) # -0.4008054
mean(all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%", 5]) # -1.134988
median(all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 10%", 5]) # -0.4023369
mean(all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%", 5]) # -0.8761519
median(all_model_data[all_model_data$FeatureSelection == " Feature Selection F Regression 20%", 5]) # -0.464641

plot(Linear_x_nofeat, Linear_y_nofeat*-1, col="red", log = "y", xlim = c(.9, 2.6), ylim = c(.3, 4.2)
     , main = "Effect of Feature Selection on Mean MSE", xlab = "Feature Selection Method", ylab = "Mean MSE"
     , xaxt = "n", cex.main = .8, cex.lab = .8)
axis(1, labels = c("None \n\n mean:5.79e+27 \n median:0.40", "MI 20% \n\n mean:2.5e+23 \n median:0.40", "F-Reg 10% \n\n mean:1.13 \n median:0.40", "F-Reg 20% \n\n mean:0.87 \n median:0.46"), 
     at = c(1, 1.5, 2, 2.5), cex.axis = .7)
legend("top", inset=0, title="Models",
       c("LR","LARSCV","RCV","ENCV", "OMPCV", "LassoCV", "DTR", "SVR"), 
       fill=c("red", "green", "blue", "orange", "purple", "gray", "chocolate4", "cyan4"), 
       horiz=TRUE, cex = .48)
points(Lars_x_nofeat, Lars_y_nofeat*-1, col="green")
points(Ridge_x_nofeat, Ridge_y_nofeat*-1, col="blue")
points(Elastic_x_nofeat, Elastic_y_nofeat*-1, col="orange")
points(Orthog_x_nofeat, Orthog_y_nofeat*-1, col="purple")
points(Lasso_x_nofeat, Lasso_y_nofeat*-1, col="gray")
points(SVR_x_nofeat, SVR_y_nofeat*-1, col="cyan4")
points(Linear_x_mi, Linear_y_mi*-1, col="red")
points(Lars_x_mi, Lars_y_mi*-1, col="green")
points(Ridge_x_mi, Ridge_y_mi*-1, col="blue")
points(Elastic_x_mi, Elastic_y_mi*-1, col="orange")
points(Orthog_x_mi, Orthog_y_mi*-1, col="purple")
points(Lasso_x_mi, Lasso_y_mi*-1, col="gray")
points(Tree_x_mi, Tree_y_mi*-1, col="chocolate4")
points(SVR_x_mi, SVR_y_mi*-1, col="cyan4")
points(Linear_x_ften, Linear_y_ften*-1, col="red")
points(Lars_x_ften, Lars_y_ften*-1, col="green")
points(Ridge_x_ften, Ridge_y_ften*-1, col="blue")
points(Elastic_x_ften, Elastic_y_ften*-1, col="orange")
points(Orthog_x_ften, Orthog_y_ften*-1, col="purple")
points(Lasso_x_ften, Lasso_y_ften*-1, col="gray")
points(Tree_x_ften, Tree_y_ften*-1, col="chocolate4")
points(SVR_x_ften, SVR_y_ften*-1, col="cyan4")
points(Linear_x_ftw, Linear_y_ftw*-1, col="red")
points(Lars_x_ftw, Lars_y_ftw*-1, col="green")
points(Ridge_x_ftw, Ridge_y_ftw*-1, col="blue")
points(Elastic_x_ftw, Elastic_y_ftw*-1, col="orange")
points(Orthog_x_ftw, Orthog_y_ftw*-1, col="purple")
points(Lasso_x_ftw, Lasso_y_ftw*-1, col="gray")
points(Tree_x_ftw, Tree_y_ftw*-1, col="chocolate4")
points(SVR_x_ftw, SVR_y_ftw*-1, col="cyan4")


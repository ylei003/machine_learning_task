


# Load libraries ---------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(randomForest)
library(caTools)
library(pROC)
library(MASS)
library(rpart)
library(rpart.plot)
library(caret)
library(class)
library(e1071)

# Data preparation -------------------------------------------------------------
# Load and check the data
df <- read.csv("Datasets/heart.csv", sep = ",", header = TRUE)
str(df)
summary(df)
sum(is.na(df))

# Numerical columns
numerical <- df %>% 
  dplyr::select("age","trtbps","chol","thalachh","oldpeak")

# Categorical columns
categorical <- df %>%
  dplyr::select("sex","cp","fbs","restecg","exng","slp","caa","thall","output") %>%
  mutate_if(is.numeric, as.factor)

df1 <- cbind(numerical, categorical)

# Correlation plot 
correlation_matrix <- cor(df)
corrplot(correlation_matrix, method = "number", type = "lower", diag = FALSE)

# Variable importance
rf <- randomForest(output ~ ., data = df1, importance = TRUE)
importance_measures <- importance(rf)
varImpPlot(rf) # caa, cp, thall are the top 3 important variables 

# Perform stepwise selection using AIC as the selection criterion
selected_model_aic <- step(glm(output ~ ., data = df, family = binomial), direction = "both", trace = FALSE)

# Perform stepwise selection using BIC as the selection criterion
selected_model_bic <- step(glm(output ~ ., data = df, family = binomial), direction = "both", trace = FALSE, k = log(nrow(df)))

# Get the selected variables for AIC and BIC models
selected_variables_aic <- names(coef(selected_model_aic))
selected_variables_bic <- names(coef(selected_model_bic))

selected_variables_aic
selected_variables_bic

# Subset with variables selected using BIC
df2 <- df1 %>%
  dplyr::select( "sex","cp","thalachh","exng","oldpeak","caa","thall","output")


# Linear models ----------------------------------------------------------------
# Train test split 
set.seed(123)
split = sample.split(df2$output, SplitRatio = 0.7)
train_linear = subset(df2, split == TRUE)
test_linear = subset(df2, split == FALSE)

# Feature Scaling
set.seed(123)
train_linear[c(3,5)] = scale(train_linear[c(3,5)])
test_linear[c(3,5)] = scale(test_linear[c(3,5)])

# Logistic regression ----------------------------------------------------------
# log_model <- glm(output ~ ., data = train_linear, family = binomial)
# predictions_log <- predict(log_model, newdata = test_linear, type = "response")
# predicted_classes <- ifelse(predictions_log > 0.5, 1, 0)
# 
# confusion_matrix <- table(predicted_classes, test_linear$output)
# confusion_matrix
# correct_count <- sum(predicted_classes == test_linear$output)
# total_count <- nrow(test_linear)
# accuracy <- (correct_count / total_count)
# accuracy # 0.8242

train_dummy <- model.matrix(~ sex + cp + exng + caa + thall + thalachh + oldpeak, data = train_linear)[, -1]
train_dummy_df <- as.data.frame(train_dummy)
train_dummy_df$output <- train_linear$output

test_dummy <- model.matrix(~ sex + cp + exng + caa + thall + thalachh + oldpeak, data = test_linear)[, -1]
test_dummy_df <- as.data.frame(test_dummy)
test_dummy_df$output <- test_linear$output

log_model <- glm(output ~ ., data = train_dummy_df, family = binomial)
predictions_log <- predict(log_model, newdata = test_dummy_df, type = "response")
predicted_classes <- ifelse(predictions_log > 0.5, 1, 0)

confusion_matrix <- table(predicted_classes, test_dummy_df$output)
confusion_matrix
correct_count <- sum(predicted_classes == test_dummy_df$output)
total_count <- nrow(test_dummy_df)
accuracy <- (correct_count / total_count)
accuracy # 0.8242

# View the summary of the logistic regression model
summary(log_model)



# Linear Discriminant Analysis (LDA) -------------------------------------------
lda_model <- lda(output ~ ., data = train_linear)
predictions_lda <- predict(lda_model, newdata = test_linear)$class

confusion_matrix <- table(predictions_lda, test_linear$output)
confusion_matrix
correct_count <- sum(predictions_lda == test_linear$output)
total_count <- nrow(test_linear)
accuracy <- (correct_count / total_count)
accuracy # 0.8132


# Tree-based models ------------------------------------------------------------
# Classification tree (ct) -----------------------------------------------------
set.seed(123)
split <- createDataPartition(df2$output, p = 0.7, list = FALSE)
train_ct <- df2[split, ]
test_ct <- df2[-split, ]

tree_model <- rpart(output ~ ., data = train_ct, method = "class")
rpart.plot(tree_model, type = 0, extra = 101, under = TRUE, fallen.leaves = TRUE)
predictions_ct <- predict(tree_model, newdata = test_ct, type = "class")

confusion_matrix <- confusionMatrix(predictions_ct, test_ct$output)
confusion_matrix
accuracy <- confusion_matrix$overall["Accuracy"]
accuracy # 0.7889

# Random forest ----------------------------------------------------------------
set.seed(123)
split <- sample.split(df2$output, SplitRatio = 0.7)
train_rf <- subset(df2, split == TRUE)
test_rf <- subset(df2, split == FALSE)

rf_model <- randomForest(output ~ ., data = train_rf, ntree = 100)
predictions_rf <- predict(rf_model, newdata = test_rf)

confusion_matrix <- confusionMatrix(predictions_rf, test_rf$output)
confusion_matrix
accuracy <- confusion_matrix$overall["Accuracy"]
accuracy # 0.8022


# Distance-based model ---------------------------------------------------------
# Train test split 
set.seed(123)
split = sample.split(df2$output, SplitRatio = 0.7)
train_scaled = subset(df2, split == TRUE)
test_scaled = subset(df2, split == FALSE)

# Feature Scaling
set.seed(123)
train_scaled[c(3,5)] = scale(train_scaled[c(3,5)])
test_scaled[c(3,5)] = scale(test_scaled[c(3,5)])

# KNN --------------------------------------------------------------------------
k <- 5 
predictions_knn <- knn(train_scaled[, -8], test_scaled[, -8], train_scaled$output, k = k)

confusion_matrix <- confusionMatrix(predictions_knn, test_scaled$output)
confusion_matrix
accuracy <- confusion_matrix$overall["Accuracy"]
accuracy # 0.7802

# SVM --------------------------------------------------------------------------
svm_model <- svm(output ~ ., data = train_scaled, kernel = "radial", gamma = 1, cost = 1)
predictions_svm <- predict(svm_model, newdata = test_scaled)

confusion_matrix <- confusionMatrix(predictions_svm, test_scaled$output)
accuracy <- confusion_matrix$overall["Accuracy"]
accuracy # 0.7802

ggplot(data = train_scaled, aes(x = thalachh, y = oldpeak, color = output)) +
  geom_point(size = 2) +
  labs(title = "Data Points with SVM Decision Boundary (RBF Kernel)",
       x = "thalachh", y = "oldpeak") +
  scale_color_manual(values = c("blue", "red")) +  # Customizing colors for classes
  theme_minimal()

plot(svm_model, train_scaled, oldpeak~thalachh)


# ROC curves -------------------------------------------------------------------
roc_logistic <- roc(test_linear$output, predictions_log)
roc_lda <- roc(test_linear$output, as.numeric(predictions_lda))
roc_ct <- roc(ifelse(predictions_ct == "1", 1, 0), ifelse(test_ct$output == "1", 1, 0))
roc_rf <- roc(test_rf$output, as.numeric(predictions_rf))
roc_knn <- roc(ifelse(predictions_knn == "1", 1, 0), ifelse(test_scaled$output == "1", 1, 0))
roc_svm <- roc(ifelse(predictions_svm == "1", 1, 0), ifelse(test_scaled$output == "1", 1, 0))

plot(roc_logistic, col = "blue", main = "ROC Curves for Multiple Models")
lines(roc_lda, col = "red", lwd = 2)
lines(roc_ct, col = "green", lwd = 2)
lines(roc_rf, col = "purple", lwd = 2)
lines(roc_knn, col = "orange", lwd = 2)
lines(roc_svm, col = "cyan", lwd = 2)

legend("bottomright", legend = c("Logistic Regression", "LDA", "Classification Tree", "Random Forest", "KNN", "SVM"),
       col = c("blue", "red", "green", "purple", "orange", "cyan"), lwd = 2)








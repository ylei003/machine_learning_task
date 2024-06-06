


# Load libraries ---------------------------------------------------------------
library(dplyr)
library(caret)
library(car)
# library(leaps)
library(boot)

library(rpart)
# install.packages("rpart.plot")
library(rpart.plot)
# install.packages("xgboost")
library(xgboost)

# Data Preparation -------------------------------------------------------------
df <- read.csv("Datasets/insurance.csv", sep = ",", header = TRUE)
df <- df[, -which(names(df) == "index")]
summary(df) # no missing values
sum(is.na(df)) # no null values
str(df) 

# Categorical variable (factor) Continuous variable (numerical)
df <- df %>%
  mutate(sex = as.factor(sex),
         smoker = as.factor(smoker),
         region = as.factor(region),
         age = as.numeric(age),
         children = as.numeric(children))

# Create dummy variables using one-hot encoding 
dummy_vars <- dummyVars(" ~ sex + smoker + region", data = df)
dummies <- as.data.frame(predict(dummy_vars, newdata = df))
# k-1 columns for dummy with k levels
dummies <- dummies[, !(grepl("sex.female", names(dummies)))]
dummies <- dummies[, !(grepl("smoker.no", names(dummies)))]
dummies <- dummies[, !(grepl("region.northwest", names(dummies)))]

df_dummy <- cbind(df, dummies)
df_dummy <- subset(df_dummy, select = -c(sex, smoker, region))

# Make dummies categorical variable
df_dummy$sex.male <- as.factor(df_dummy$sex.male)
df_dummy$smoker.yes <- as.factor(df_dummy$smoker.yes)
df_dummy$region.northeast <- as.factor(df_dummy$region.northeast)
df_dummy$region.southeast <- as.factor(df_dummy$region.southeast)
df_dummy$region.southwest <- as.factor(df_dummy$region.southwest)


# Linear Regression Model ------------------------------------------------------

# Standardize continuous variables (excluding the target variable)
df_dummy1 <- df_dummy
numerical <- sapply(df_dummy1, is.numeric) & 
  !grepl("charges", names(df_dummy1))
df_dummy1[, numerical] <- scale(df_dummy1[, numerical])

# Train-test split
set.seed(123) 
split_index <- createDataPartition(df_dummy1$charges, p = 0.7, list = FALSE)
train1 <- df_dummy1[split_index, ]
test1 <- df_dummy1[-split_index, ]

# Train the model and predict 
lm <- lm(charges ~ ., data = train1)
pred <- predict(lm, newdata = test1)

# Check multicollinearity and evaluate model fit
vif <- vif(lm) # not necessary to use shrinkage methods as VIF is lower than 2
rmse <- sqrt(mean((test1$charges - pred)^2)) # rmse is 6196.277

# Diagnostic plot to assess linear regression assumptions 
par(mfrow = c(2, 2))
plot(lm, which = 1)
plot(lm, which = 2)
plot(lm, which = 3)
plot(lm, which = 5)
par(mfrow = c(1, 1))

# 10-fold cross validation for polynomial regression of degree 1, 2, 3 ---------
# Scale all continuous variables (including the target variable)
df_dummy2 <- df_dummy
numerical2 <- sapply(df_dummy2, is.numeric)
df_dummy2[, numerical2] <- scale(df_dummy2[, numerical2])

set.seed(123)
cv.10.error = rep(0, 3)
for (i in 1:3){
  glm.fit <- glm(charges ~ poly(age, i) + poly(bmi, i) + poly(children, i), data = df_dummy2)
  cv.10.error[i] <- cv.glm(df_dummy2, glm.fit, K = 10)$delta[1]
}
cv.10.error # 0.8835, 0.8824, 0.8858


# Tree-based Model -------------------------------------------------------------
# Regression Tree --------------------------------------------------------------
# Train-test split
set.seed(123) 
split_index <- createDataPartition(df_dummy$charges, p = 0.7, list = FALSE)
train <- df_dummy[split_index, ]
test <- df_dummy[-split_index, ]

m1 <- rpart(
  formula = charges ~ .,
  data    = train,
  method  = "anova"
)
rpart.plot(m1) # Tree with 4 internal nodes and 5 terminal nodes
plotcp(m1) # cross validation error, cost complexity (α) value, number of terminal nodes

m2 <- rpart(
  formula = charges ~ .,
  data    = train,
  method  = "anova", 
  control = list(cp = 0, xval = 10)
)
plotcp(m2)
abline(v = 5, lty = "dashed") # 5 terminal nodes minimizes the cross validation error
m1$cptable 

pred <- predict(m1, newdata = test)
rmse <- sqrt(mean((test$charges - pred)^2)) # rmse is 5241.7

# Gradient Boosting using xgBoost  ---------------------------------------------
X <- subset(df_dummy, select = -c(charges)) # Features
y <- df_dummy$charges                       # Target variable

set.seed(123) 
train_index <- sample(1:nrow(df_dummy), 0.8 * nrow(df_dummy))  # 80/20 split
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Convert data to matrix format
X_train_matrix <- as.matrix(apply(X_train, 2, as.numeric))
X_test_matrix <- as.matrix(apply(X_test, 2, as.numeric))
# Create DMatrix objects
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train)
dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test)

# Set parameters for xgboost
params <- list(
  objective = "reg:squarederror",  # Regression task
  eval_metric = "rmse"              # Root Mean Squared Error as evaluation metric
)

model <- xgboost(data = dtrain, params = params, nrounds = 100)
predictions <- predict(model, dtest)
rmse <- sqrt(mean((predictions - y_test)^2)) # rmse is 4870.048


























# Load libraries ---------------------------------------------------------------
library(corrplot)
library(factoextra)
library(cluster)
library(dplyr)

# Data preparation -------------------------------------------------------------
df <- read.csv("Datasets/Country-data.csv", sep=",", header=TRUE)
summary(df) # no missing values 

str(df) # check data types
df$income <- as.numeric(df$income)
df$gdpp <- as.numeric(df$gdpp)
df.numeric <- df[, sapply(df, is.numeric)] # Create numeric data frame 

# Principal Component Analysis -------------------------------------------------
corr <- cor(df.numeric)
corrplot(corr, method = "number", type = "lower", diag = FALSE)

# Standardization to obtain mean zero and standard deviation one 
pca <- prcomp(df.numeric, scale=TRUE)

summary(pca)
pca.loadings <- pca$rotation
print(pca.loadings, digits = 3)
print(pca$sdev)

# Proportional variance explained (pve) 
pve <- pca$sdev^2/sum(pca$sdev^2)
plot(1:length(pve), pve, type = "b", 
     main = "Scree Plot - Proportional Variance Explained",
     xlab = "Principal Component", ylab = "Prop. Variance Explained")
abline(h = 0.1, col = "red", lty = 2)


# K-means clustering -----------------------------------------------------------
df.scaled <- scale(df.numeric) # Scaling the numeric data

# Elbow methods to determine k (optimal k is 4 or 5)
set.seed(123)
wss <- function(k) {
  kmeans(df.scaled, k, nstart = 25 )$tot.withinss
}
k.values <- 1:10
wss.values <- sapply(k.values, wss) # Compute wss for k = 1 to k = 10
plot(k.values, wss.values,
     type="b", pch = 19, frame = FALSE, 
     main="Elbow Method for K-means",
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# Silhouette method to determine k (optimal k is 5)
sil.score <- function(k){
  km.result <- kmeans(df.scaled, centers = k, nstart=25)
  ss <- silhouette(km.result$cluster, dist(df.scaled))
  mean(ss[, 3])
}
k <- 2:10
avg.sil <- sapply(k, sil.score) # Compute wss for k = 2 to k = 10
plot(k, type='b', avg.sil, 
     main="Silhouette Method",
     xlab='Number of clusters', 
     ylab='Average Silhouette Scores', frame=FALSE)  

# alternative using the factoextra package 
fviz_nbclust(df.scaled, kmeans, method='silhouette') 

# Gap Statistic Method (optimal k is 3)
set.seed(123)
gap_stat <- clusGap(df.scaled, FUN = kmeans, nstart = 25, K.max = 10, B = 20) 
print(gap_stat, method = "firstmax")
fviz_gap_stat(gap_stat)

# Clustering results with 5 clusters
set.seed(321)
final.result <- kmeans(df.scaled, 5, nstart = 25)
print(final.result)
fviz_cluster(final.result, data = df.scaled)

# Add the clusters to data before being scaled 
df.numeric %>%
  mutate(Cluster = final.result$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean") 

# Hierarchical Clustering ------------------------------------------------------
dist.matrix <- dist(df.scaled, method = "euclidean")

hc.complete <- hclust(dist.matrix, method = "complete")
clusters <- cutree(hc.complete, k = 5)
plot(hc.complete)
rect.hclust(hc.complete, k = 5, border = 4:6)

hc.average <- hclust(dist.matrix, method = "average")
clusters <- cutree(hc.average, k = 5)
plot(hc.average)
rect.hclust(hc.average, k = 5, border = 4:6)

hc.single <- hclust(dist.matrix, method = "single")
clusters <- cutree(hc.single, k = 5)
plot(hc.single)
rect.hclust(hc.single, k = 5, border = 4:6)

hc.ward <- hclust(dist.matrix, method = "ward.D2")
clusters <- cutree(hc.ward, k = 5)
plot(hc.ward)
rect.hclust(hc.ward, k = 5, border = 4:6)



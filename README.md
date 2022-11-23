# Comparing-Clustering-Methods-on-Military-Power-by-Countries

A. Baseline Model K-means++
Since our goal is to cluster countries by their military power, we first tried the k-means++ clustering algorithm as our baseline model. K-means++ is a centroid-based partition- ing clustering method. It is simple to implement and identify unknown groups of data from complex datasets. Compared to other clustering methods, k-means clustering technique is more efficient and fast, its computational cost is only O(K*n*d).

First, k-means++ needs to determine a specific number of clusters in order to do clustering. We used the elbow method to decide the number of clusters. In order to do that, we imported the k-means package from scikit-learn and constructed a graph. The x-axis of the graph is the number of clusters, and the y-axis is the sum of squared distance between each point and the centroid in a cluster (WCSS). From the graph, as the number of clusters increases, the WCSS value will start to decrease. It is obvious that since the k starts from 4, the slope of the line becomes relatively flat, which means that the WCSS does not have a big change after 4. We can conclude that 4 is the elbow point. Therefore, we chose 4 as our number of clusters.

Next, we used k-means++ to initialize the centroids. K- means++ algorithm works as the following:
1) Chooseonecenteruniformlyatrandomamongthedata points.
2) For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
3) Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
4) Repeat Steps 2 and 3 until k centers have been chosen.
5) Now that the initial centers have been chosen, proceed using standard k-means clustering.

The reason why we do not use k-means to select centroids is that it will randomly choose the centroid, which means that these centroids may fall into the same cluster. However, with k-means ++, only the first centroid is selected randomly, the remaining centroids are chosen by a probability proportional to its squared distance. This selection process for calculating the distance is using the Euclidean distance method. K- means++ removes the drawback of k-means that is dependent on initialization of centroid. This step helps us to select 4 distinct centroids.
Advance Techniques for Visualization: We assigned each data point to its nearest centroid using Euclidean distance to form different clusters. After that, we updated the centroids by taking the mean of the data points in each cluster. And then, we repeated the same process until convergence (no further changes) and the maximum number of iterations reached. The final result of k-means++ clustering in a two dimensional space is shown in the coding file.

B. Model Enhancement (GMM)
Our advanced model is the Gaussian Mixture Model (GMM), which is an extension of k-means++, but it is more powerful to estimate clustering. It is a distribution-based method that assumes our dataset follows Gaussian or normal distribution. This method does not assume clusters are any geometry. Also it works well with non-linear geometric distributions while k-means++ does not work efficiently with complex geometrical shaped data. Opposed to k-means++, GMM does not bias the cluster size to have specific struc- tures. What’s more, GMM is a soft clustering algorithm compared to k-means. Soft clustering means that each data point can belong to more than one cluster.
GMM also specifies the number of clusters (k) in advance, so we used 4 clusters because of the elbow method.
Then, for our dataset with 46 features, we would have a mixture of 4 Gaussian distributions, each having a certain mean vector and variance matrix. These values are deter- mined using a technique called Expectation-Maximization (EM). The steps of EM is shown as the following:
1) Initialize the parameters.
2) Expectation Step - E step: Given the current cluster
centers, each object is assigned to the cluster with a center that is closest to the object. Here, an object is expected to belong to the closest cluster.
3) Maximization Step - M step: Given the cluster as- signment, for each cluster, the algorithm adjusts the center so that the sum of the distances from the objects assigned to this cluster and the new center is minimized.
This expectation-maximization (EM) algorithm provides an iterative solution to maximum likelihood estimation. The program continues to run this algorithm until convergence which means no further changes.
We built the model and fit the data into the model by using the scikit-learn package. In this model, each point is assigned a probability of belonging to a cluster, and each cluster is modeled based on a different Gaussian distribution.
Advance Techniques for Visualization: Similar to the il- lustration of the steps in the PCA analysis for k-means++ above, we used 2 components here, so the data will transform into two dimensions. The final visualization of the GMM algorithm by applying T-SNE is shown in the coding file.
We also implemented another dimensional reduction for this section called T-SNE, the T-Stochastic Neighbor Embed- ding, is a non-linear dimensionality reduction algorithm used for transforming high-dimensional data into low-dimension. It preserves local and global structure. T-SNE can handle outliers better than PCA. However, the outliers have been removed in the data preprocessing step, so there may be no big difference between them. T-SNE involves Hyperparam- eters such as learning rate and numbers of steps. The steps are shown as the following:
1) Compute pairwise similarity Pij for every element i and j (j is the neighbor of i).
2) Make Pij symmetric.
3) Choose a random solution Y0
4) While not done:
• compute pairwise similarities for Yi
• compute the gradient
• update the solution
• check if reach max iteration
We implemented T-SNE by using the pre-written scikit- learn packages. The final visualization of the GMM algo- rithm by applying T-SNE is also shown in the coding file. However, by comparing the PCA and T-SNE graph, we can see that the t-SNE did a slightly worse job than PCA. T-SNE seems to have more clusters or data points that overlap with each other than PCA. The data distribution in PCA also gives a better expression of the countries’ status in general. Therefore, PCA may be a better choice here.

C. Future Work
In the future, we are planning to try some additional clustering methods on this model, such as density based clustering method DBSCAN. Also, as the war is taking place between Ukraine and Russia, we are planning to analyze the impact of wars on the military power of countries as well by comparing modeling results before and after the war.

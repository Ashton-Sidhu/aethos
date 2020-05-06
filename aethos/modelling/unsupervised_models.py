import pandas as pd

from aethos.modelling.model import ModelBase
from aethos.config import shell
from aethos.model_analysis.unsupervised_model_analysis import UnsupervisedModelAnalysis
from aethos.analysis import Analysis
from aethos.cleaning.clean import Clean
from aethos.preprocessing.preprocess import Preprocess
from aethos.feature_engineering.feature import Feature
from aethos.visualizations.visualizations import Visualizations
from aethos.stats.stats import Stats
from aethos.modelling.util import add_to_queue


class Unsupervised(
    ModelBase, Analysis, Clean, Preprocess, Feature, Visualizations, Stats
):
    def __init__(
        self, x_train, exp_name="my-experiment",
    ):
        """
        Class to run analysis, transform your data and run Unsupervised algorithms.

        Parameters
        -----------
        x_train: pd.DataFrame
            Training data or aethos data object

        exp_name : str
            Experiment name to be tracked in MLFlow.
        """

        super().__init__(
            x_train, "", x_test=None, test_split_percentage=0.2, exp_name=exp_name,
        )

    @add_to_queue
    def KMeans(
        self, model_name="km", run=True, verbose=1, **kwargs,
    ):
        # region
        """
        NOTE: If 'n_clusters' is not provided, k will automatically be determined using an elbow plot using distortion as the mteric to find the optimal number of clusters.

        K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

        The objective of K-means is simple: group similar data points together and discover underlying patterns.
        To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.

        In other words, the K-means algorithm identifies k number of centroids,
        and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

        For a list of all possible options for K Means clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        Parameters
        ----------

        model_name : str, optional
            Name for this model, by default "kmeans"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.

        init : {‘k-means++’, ‘random’ or an ndarray}
            Method for initialization, defaults to ‘k-means++’:
                ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.

                ‘random’: choose k observations (rows) at random from data for the initial centroids.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

        n_init : int, default: 10
            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a single run.

        random_state : int, RandomState instance or None (default)
            Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See Glossary.

        algorithm : “auto”, “full” or “elkan”, default=”auto”
            K-means algorithm to use.
            The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. 
            “auto” chooses “elkan” for dense data and “full” for sparse data.
                    
        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and further analysis

        Examples
        --------
        >>> model.KMeans()
        >>> model.KMeans(model_name='kmean_1, n_cluster=5)
        >>> model.KMeans(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import KMeans

        def find_optk():

            from yellowbrick.cluster import KElbowVisualizer

            model = KMeans(**kwargs)

            visualizer = KElbowVisualizer(model, k=(4, 12))
            visualizer.fit(self.train_data)
            visualizer.show()

            print(f"Optimal number of clusters is {visualizer.elbow_value_}.")

            return visualizer.elbow_value_

        n_clusters = kwargs.pop("n_clusters", None)

        if not n_clusters:
            n_clusters = find_optk()

        model = KMeans

        model = self._run_unsupervised_model(
            model,
            model_name,
            run=run,
            verbose=verbose,
            n_clusters=n_clusters,
            **kwargs,
        )

        return model

    @add_to_queue
    def DBScan(
        self, model_name="dbs", run=True, verbose=1, **kwargs,
    ):
        # region
        """
        Based on a set of points (let’s think in a bidimensional space as exemplified in the figure), 
        DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points.
        It also marks as outliers the points that are in low-density regions.

        The DBSCAN algorithm should be used to find associations and structures in data that are hard to find manually but that can be relevant and useful to find patterns and predict trends.
        
        For a list of all possible options for DBSCAN please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        Parameters
        ----------

        model_name : str, optional
            Name for this model, by default "dbscan"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.

        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

        metric : string, or callable
            The metric to use when calculating distance between instances in a feature array.
            If metric is a string or callable, it must be one of the options allowed by sklearn.
            If metric is “precomputed”, X is assumed to be a distance matrix and must be square.
            X may be a sparse matrix, in which case only “nonzero” elements may be considered neighbors for DBSCAN.

        p : float, optional
            The power of the Minkowski metric to be used to calculate distance between points.

        n_jobs : int or None, optional (default=None)
            The number of parallel jobs to run.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            See Glossary for more details.
        
        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and further analysis

        Examples
        --------
        >>> model.DBScan()
        >>> model.DBScan(model_name='dbs_1, min_samples=5)
        >>> model.DBScan(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import DBSCAN

        model = DBSCAN

        model = self._run_unsupervised_model(model, model_name, run=run, **kwargs,)

        return model

    @add_to_queue
    def IsolationForest(
        self, model_name="iso_forest", run=True, verbose=1, **kwargs,
    ):
        # region
        """
        Isolation Forest Algorithm

        Return the anomaly score of each sample using the IsolationForest algorithm

        The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

        For more Isolation Forest info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
        
        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "iso_forest"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

        n_estimators : int, optional (default=100)
            The number of base estimators in the ensemble.

        max_samples : int or float, optional (default=”auto”)
            The number of samples to draw from X to train each base estimator.

                    If int, then draw max_samples samples.
                    If float, then draw max_samples * X.shape[0] samples.
                    If “auto”, then max_samples=min(256, n_samples).

            If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).

        contamination : float in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
            Used when fitting to define the threshold on the decision function.
            If ‘auto’, the decision function threshold is determined as in the original paper.

        max_features : int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.

                    If int, then draw max_features features.
                    If float, then draw max_features * X.shape[1] features.

        bootstrap : boolean, optional (default=False)
            If True, individual trees are fit on random subsets of the training data sampled with replacement.
            If False, sampling without replacement is performed.

        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.IsolationForest()
        >>> model.IsolationForest(model_name='iso_1, max_features=5)
        >>> model.IsolationForest(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.ensemble import IsolationForest

        model = IsolationForest

        model = self._run_unsupervised_model(
            model, model_name, run=run, verbose=verbose, **kwargs,
        )

        return model

    @add_to_queue
    def OneClassSVM(
        self, model_name="ocsvm", run=True, verbose=1, **kwargs,
    ):
        # region
        """
        Trains a One Class SVM model.

        Unsupervised Outlier Detection.

        For more Support Vector info, you can view it here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
        
        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "ocsvm"     

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1    	

        kernel : string, optional (default=’rbf’)
            Specifies the kernel type to be used in the algorithm.
            It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
            If none is given, ‘rbf’ will be used.
            If a callable is given it is used to precompute the kernel matrix.

        degree : int, optional (default=3)
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

        gamma : float, optional (default=’auto’)
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.

        coef0 : float, optional (default=0.0)
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

        tol : float, optional
            Tolerance for stopping criterion.

        nu : float, optional
            An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
            Should be in the interval (0, 1]. By default 0.5 will be taken.

        shrinking : boolean, optional
            Whether to use the shrinking heuristic.

        cache_size : float, optional
            Specify the size of the kernel cache (in MB).
        
        max_iter : int, optional (default=-1)
            Hard limit on iterations within solver, or -1 for no limit.

        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and analyze results

        Examples
        --------
        >>> model.OneClassSVM()
        >>> model.OneClassSVM(model_name='ocs_1, max_iter=100)
        >>> model.OneClassSVM(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.svm import OneClassSVM

        model = OneClassSVM

        model = self._run_unsupervised_model(
            model, model_name, run=run, verbose=verbose, **kwargs,
        )

        return model

    @add_to_queue
    def AgglomerativeClustering(
        self, model_name="agglom", run=True, **kwargs,
    ):
        # region
        """
        Trains a Agglomerative Clustering Model

        Each data point as a single cluster at the outset and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all data points
        
        Hierarchical clustering does not require us to specify the number of clusters and we can even select which number of clusters looks best since we are building a tree.
        
        Additionally, the algorithm is not sensitive to the choice of distance metric; all of them tend to work equally well whereas with other clustering algorithms, 
        the choice of distance metric is critical. 
        
        A particularly good use case of hierarchical clustering methods is when the underlying data has a hierarchical structure and you want to recover the hierarchy;
        other clustering algorithms can’t do this.
        
        These advantages of hierarchical clustering come at the cost of lower efficiency, as it has a time complexity of O(n³), unlike the linear complexity of K-Means and GMM.

        For a list of all possible options for Agglomerative clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "agglom" 

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        n_clusters : int or None, optional (default=2)
            The number of clusters to find.
            It must be None if distance_threshold is not None.

        affinity : string or callable, default: “euclidean”
            Metric used to compute the linkage.
            Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
            If linkage is “ward”, only “euclidean” is accepted.
            If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.

        compute_full_tree : bool or ‘auto’ (optional)
            Stop early the construction of the tree at n_clusters.
            This is useful to decrease computation time if the number of clusters is not small compared to the number of samples.
            This option is useful only when specifying a connectivity matrix.
            Note also that when varying the number of clusters and using caching, it may be advantageous to compute the full tree.
            It must be True if distance_threshold is not None.

        linkage : {“ward”, “complete”, “average”, “single”}, optional (default=”ward”)
            Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.

                'ward' minimizes the variance of the clusters being merged.
                'average' uses the average of the distances of each observation of the two sets.
                'complete' or maximum linkage uses the maximum distances between all observations of the two sets.
                'single' uses the minimum of the distances between all observations of the two sets.

        distance_threshold : float, optional (default=None)
            The linkage distance threshold above which, clusters will not be merged.
            If not None, n_clusters must be None and compute_full_tree must be True.
                    
        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and further analysis

        Examples
        --------
        >>> model.AgglomerativeClustering()
        >>> model.AgglomerativeClustering(model_name='ag_1, n_clusters=5)
        >>> model.AgglomerativeClustering(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering

        model = self._run_unsupervised_model(model, model_name, run=run, **kwargs,)

        return model

    @add_to_queue
    def MeanShift(
        self, model_name="mshift", run=True, **kwargs,
    ):
        # region
        """
        Trains a Mean Shift clustering algorithm.

        Mean shift clustering aims to discover “blobs” in a smooth density of samples.

        It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region.

        These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

        For more info on Mean Shift clustering please visit: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift

        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "mshift"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        bandwidth : float, optional
            Bandwidth used in the RBF kernel.

            If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth; see the documentation for that function for hints on scalability (see also the Notes, below).

        seeds : array, shape=[n_samples, n_features], optional
            Seeds used to initialize kernels.
            If not set, the seeds are calculated by clustering.get_bin_seeds with bandwidth as the grid size and default values for other parameters.

        bin_seeding : boolean, optional
            If true, initial kernel locations are not locations of all points, but rather the location of the discretized version of points, where points are binned onto a grid whose coarseness corresponds to the bandwidth.
            Setting this option to True will speed up the algorithm because fewer seeds will be initialized.
            default value: False Ignored if seeds argument is not None.        
            
        min_bin_freq : int, optional
            To speed up the algorithm, accept only those bins with at least min_bin_freq points as seeds.
            If not defined, set to 1.

        cluster_all : boolean, default True
            If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel.
            If false, then orphans are given cluster label -1.
                    
        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and further analysis

        Examples
        --------
        >>> model.MeanShift()
        >>> model.MeanShift(model_name='ms_1', cluster_all=False)
        >>> model.MeanShift(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.cluster import MeanShift

        model = MeanShift

        model = self._run_unsupervised_model(model, model_name, run=run, **kwargs,)

        return model

    @add_to_queue
    def GaussianMixtureClustering(
        self, model_name="gm_cluster", run=True, verbose=1, **kwargs,
    ):
        # region
        """
        Trains a GaussianMixture algorithm that implements the expectation-maximization algorithm for fitting mixture
        of Gaussian models.

        A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

        There are 2 key advantages to using GMMs.
        
        Firstly GMMs are a lot more flexible in terms of cluster covariance than K-Means; due to the standard deviation parameter, the clusters can take on any ellipse shape, rather than being restricted to circles.
        
        K-Means is actually a special case of GMM in which each cluster’s covariance along all dimensions approaches 0.
        Secondly, since GMMs use probabilities, they can have multiple clusters per data point.
        
        So if a data point is in the middle of two overlapping clusters, we can simply define its class by saying it belongs X-percent to class 1 and Y-percent to class 2. I.e GMMs support mixed membership.

        For more information on Gaussian Mixture algorithms please visit: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

        Parameters
        ----------
        model_name : str, optional
            Name for this model, by default "gm_cluster"

        run : bool, optional
            Whether to train the model or just initialize it with parameters (useful when wanting to test multiple models at once) , by default False

        verbose : int, optional
            Verbosity level of model output, the higher the number - the more verbose. By default, 1

        n_components : int, defaults to 1.
            The number of mixture components/ number of unique y_train values.

        covariance_type : {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}
            String describing the type of covariance parameters to use. Must be one of:

            ‘full’
                each component has its own general covariance matrix

            ‘tied’
                all components share the same general covariance matrix

            ‘diag’
                each component has its own diagonal covariance matrix

            ‘spherical’
                each component has its own single variance

        tol : float, defaults to 1e-3.
            The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

        reg_covar : float, defaults to 1e-6.
            Non-negative regularization added to the diagonal of covariance.
            Allows to assure that the covariance matrices are all positive.

        max_iter : int, defaults to 100.
            The number of EM iterations to perform.

        n_init : int, defaults to 1.
            The number of initializations to perform. The best results are kept.

        init_params : {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
            The method used to initialize the weights, the means and the precisions. Must be one of:

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

        weights_init : array-like, shape (n_components, ), optional
            The user-provided initial weights
            If it None, weights are initialized using the init_params method.
            Defaults to None. 

        means_init : array-like, shape (n_components, n_features), optional
            The user-provided initial means
            If it None, means are initialized using the init_params method.
            Defaults to None

        precisions_init : array-like, optional.
            The user-provided initial precisions (inverse of the covariance matrices), defaults to None. If it None, precisions are initialized using the ‘init_params’ method. The shape depends on ‘covariance_type’:

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
            
        Returns
        -------
        UnsupervisedModelAnalysis
            UnsupervisedModelAnalysis object to view results and further analysis

        Examples
        --------
        >>> model.GuassianMixtureClustering()
        >>> model.GuassianMixtureClustering(model_name='gm_1, max_iter=1000)
        >>> model.GuassianMixtureClustering(run=False) # Add model to the queue
        """
        # endregion

        from sklearn.mixture import GaussianMixture

        model = GaussianMixture

        model = self._run_unsupervised_model(
            model, model_name, run=run, verbose=verbose, **kwargs,
        )

        return model

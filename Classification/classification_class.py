import time
import math

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression

class DataProcessing:

    def __init__(self, X, y, test_size=0.2, val_size=0.2, stratified=True, shuffle=False,
                 scaling_method='standard', dim_reduction = None, threshold_dim_reduction=0.97):
        """
        Initializes the DataProcessing class.
        
        :param X: input features.
        :type X: pandas.DataFrame
        
        :param y: target variable.
        :type y: pandas.DataFrame
        
        :param test_size: dataset proportion for testing.
        :type test_size: float
        
        :param val_size: dataset proportion for validation.
        :type val_size: float
        
        :param stratified: stratified or not
        :type stratified: bool, default=True

        :param shuffle: shuffle
        :type shuffle: bool, default=True
        
        :param scaling_method: method for scaling ('standard' for StandardScaler or 'minmax' for MinMaxScaler).
        :type scaling_method: str

        :param dim_reduction: dimension reduction method to use
        :type dim_reduction: str, default=None
        
        :param threshold_dim_reduction: threshold to use for dimension reduction (PCA)
        :type threshold_dim_reduction: float, default=0.97
        
        """
        self.X_df = X
        self.X = np.array(X)
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.y = y.squeeze()
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.test_size = test_size
        self.val_size = val_size
        self.stratified=stratified
        self.scaling_method = scaling_method
        self.dim_reduction = dim_reduction
        self.threshold_dim_reduction = threshold_dim_reduction

        _ = self.split_data(stratified=stratified, shuffle=shuffle)

        self.X_train_val_untouched = np.concatenate((self.X_train, self.X_val), axis=0)
        self.y_train_val_untouched = np.concatenate((self.y_train, self.y_val), axis=0)
        
        self.X_train, self.X_val, self.X_test = self.scaling(self.X_train, self.X_val, self.X_test)
        
        if dim_reduction == "PCA":
            self.X_train, self.X_val, self.X_test, _ = self.apply_PCA(threshold_dim_reduction)
    
    def split_data(self, stratified=True, shuffle=True):
        '''
        Splits the data into training, validation and tests sets

        :param stratified: indicates if the separation must be stratified according to the target variable
        :type stratified: bool, default=True
        :param shuffle: shuffle
        :type shuffle: bool, default=True
        :return: X_train, X_val, X_test, y_train, y_val, y_test: arrays for training, validation and tests feature and target sets
        :rtype: tuple
        '''

        if stratified:
            s1 = self.y
        else:
            s1 = None
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.test_size, 
                                                            random_state=42, shuffle=True, stratify=s1)
        
        validation_size = self.val_size / (1 - self.test_size) 

        if stratified:
            s2 = y_train
        else:
            s2 = None
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_size,
                                                          random_state=42, shuffle=True, stratify=s2)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scaling(self, X_train, X_val, X_test):
        """
        Performs scaling
       
        :param X_train: training array of features
        :type X_train: numpy.ndarray
        :param X_val: training array of features
        :type X_val: numpy.ndarray 
        :param X_test: training array of features
        :type X_test: numpy.ndarray 
        :return: scaled datasets
        :rtype: tuple
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def apply_PCA(self, threshold=0.97):
        """Chooses the best number of components for PCA

        :param threshold: percentage of explained variance kept, defaults to 0.97
        :type threshold: float, optional
        :return: X_train, X_val, X_test, cumulative_variance
        :rtype: tuple
        """
        pca=PCA(n_components=min(self.n, self.p)) # min(n_observations, n_features) is the number max of components
        _ = pca.fit_transform(self.X_train)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance >= threshold) + 1
        print("Number of components for the chosen cumulative variance:", num_components)
        
        pca = PCA(n_components=num_components)
        X_train = pca.fit_transform(self.X_train)
        X_val = pca.transform(self.X_val)
        X_test = pca.transform(self.X_test)

        self.pca_n_components = num_components
        
        return X_train, X_val, X_test, cumulative_variance

    def histogram (self, bins=10):
        """
        Plots a histogram of y values
        """
        plt.hist(self.y, bins=bins, edgecolor='black')
        plt.title("Histogram")
        plt.xlabel("Values")
        plt.ylabel("Frequencies")
        plt.show()

    def plot_boxplot(self, xlabels, title="Features boxplot"):
        """
        Generates a boxplot for each column in the provided DataFrame with column names on the x-axis.
    
        This function uses Seaborn for creating aesthetically pleasing boxplots. Each box represents the distribution 
        of data for a column in the DataFrame.
    
        :param xlabels: x labels
        :type xlabels: list
        :param title: Graphic title
        :type title: str, default="Features boxplot"
        :return: Displays the boxplot.
        :rtype: None
        """
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.X)
    
        plt.title("Features boxplot", fontsize=16)
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Values", fontsize=14)
    
        plt.xticks(ticks=range(len(xlabels)), labels=xlabels, rotation=45, ha='right')
    
        plt.tight_layout()
        plt.show()

class BinaryClassification:
    """This class allows to perform diverse binary classification techniques.
        For each classification method, a function is implemented to get the optimal hyperparameters.
        Cross Validation and plotting results can also be done.
    """
    def __init__(self, data, stratified=True, metric="accuracy", average="binary"):
        """Initialization of the class

        :param data: object of class DataProcessing
        :type data: DataProcessing
        
        :param X: array of features 
        :type X: pandas.core.frame.DataFrame
        
        :param metric: metric to use in the class
        :type metric: str, default="accuracy"

        :param average: parameter average for f1-score, precision and recall
        :type metric str, default="binary"
        """
        self.data = data
        self.nb_labels = np.unique(self.data.y).shape[0]
        self.methods = ["LogReg", "ClassificationDecisionTree", "RandomForest"]
        self.models = {}

        self.metrics_dict = {"accuracy": accuracy_score,
                             "f1-score": lambda y_true, y_pred: f1_score(y_true, y_pred, average=self.average),
                             "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average=self.average),
                             "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average=self.average),
                             "roc_auc": roc_auc_score}

        if metric not in self.metrics_dict.keys():
            raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(self.metrics_dict.keys())}.")
        if average not in list(['micro', 'macro', 'samples', 'weighted', 'binary']):
            raise ValueError(f"Unsupported average type '{average}'. Choose from 'micro', 'macro', 'samples', 'weighted', 'binary'.")

        chosen_metric = self.metrics_dict[metric]
        self.metric_name = metric
        self.metric = chosen_metric
        self.average=average

        if metric == 'accuracy':
            self.scoring_function = 'accuracy'
        elif metric == 'f1-score':
            self.scoring_function = make_scorer(f1_score, average=self.average)
        elif metric == 'precision':
            self.scoring_function = make_scorer(precision_score, average=self.average)
        elif metric == 'recall':
            self.scoring_function = make_scorer(recall_score, average=self.average)
        elif metric == 'roc_auc':
            self.scoring_function = 'roc_auc'
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(self.metrics_dict.keys())}.")

    def logisticRegression(self):
        """Performs Logistic Regression and returns the best hyperparameters
        
        :return: best hyperparameters and performance
        :rtype: tuple
        """
        # print("Optimization Logistic Regression")
        logreg = LogisticRegression(max_iter=10000)
        
        # Hyperparameters to test
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # GridSearchCV with 5 folds
        grid_search = GridSearchCV(
            estimator=logreg,
            param_grid=param_grid,
            cv=5,
            scoring=self.scoring_function,
            n_jobs=-1
        )
        grid_search.fit(self.data.X_train, self.data.y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.best_estimator_.predict(self.data.X_val)
        performance = self.metric(self.data.y_val, y_pred)
        
        return best_params, performance

    def ClassificationDecisionTree(self):
        """Performs Classification Decision Tree and returns the optimal hyperparameters with best performance associated on validation set
        The hyperparameters are the maximum depth, the minimum samples split, the minimum samples leaf and the criterion.

        :return: best hyperparameters and performance
        :rtype: tuple
        """
        # print("Optimization Classification Decision Tree")
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        tree = DecisionTreeClassifier(random_state=0)
        grid_search = GridSearchCV(
            estimator=tree,
            param_grid=param_grid,
            cv=5,
            scoring=self.scoring_function,
            n_jobs=-1
        )
        grid_search.fit(self.data.X_train, self.data.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        y_pred = grid_search.best_estimator_.predict(self.data.X_val)
        performance = self.metric(self.data.y_val, y_pred)

        return best_params, performance
        
    def RandomForest(self, depth_max=10):
        """Performs Random Forest and returns the optimal hyperparameters with best performance associated on validation set
        The hyperparameters are the maximum depth, the minimum samples split, the minimum samples leaf and the criterion.
        Note: testing the combinations of all those parameters can take some time.

        :param depth_max: maximum depth
        :type depth_max: int, default=10
        
        :return: best hyperparameters and performance
        :rtype: tuple
        """
        # print("Optimization Random Forest")
        param_grid = {
            'max_depth': list(range(1, depth_max + 1)) + [None],
            # 'min_samples_split': [2, 5, 10],
            # 'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        rf = RandomForestClassifier(random_state=0)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring=self.scoring_function,
            n_jobs=-1
        )
        grid_search.fit(self.data.X_train, self.data.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        y_pred = grid_search.best_estimator_.predict(self.data.X_val)
        performance = self.metric(self.data.y_val, y_pred)

        return best_params, performance

    def optimal_hyperparameters(self, depth_max=10):

        """
        Optimizes hyperparameters for all classification methods, updates `self.models` with the best models, 
        and returns a dictionary containing the best hyperparameters and their corresponding performance metrics.
    
        :param depth_max: Maximum depth for tree-based models such as Random Forest and Extra Trees.
        :type depth_max: int, default=10
    
        :return: A dictionary where each key is the name of a classification method and each value is another dictionary 
                containing the best hyperparameters (`best_params`) and the corresponding performance metrics (`performance`).
        :rtype: dict
        """
        
        results = {}

        # Optimization Logistic Regression
        best_params_logreg, perf_logreg = self.logisticRegression()
        results['LogReg'] = {'best_params': best_params_logreg, 'performance': perf_logreg}
        
        # Optimization Classification Decision Tree
        best_params_tree, perf_tree = self.ClassificationDecisionTree()
        results['DecisionTree'] = {'best_params': best_params_tree, 'performance': perf_tree}
        
        # Optimization Random Forest
        best_params_rf, perf_rf = self.RandomForest(depth_max=depth_max)
        results['RandomForest'] = {'best_params': best_params_rf, 'performance': perf_rf}

        best_params=results

        self.models = {
            "LogReg": LogisticRegression(**best_params['LogReg']['best_params'], max_iter=10000),
            "ClassificationDecisionTree": DecisionTreeClassifier(**best_params['DecisionTree']['best_params'], random_state=0),
            "RandomForest" : RandomForestClassifier(**best_params['RandomForest']['best_params'], random_state=0)
        }
        
        return results

    def TrainTest(self, find_optimal_hyperparameters=False, depth_max=10):

        """
        Optimizes hyperparameters for all classification methods, trains each model, and evaluates their performance 
        on both training and testing datasets.
    
        This function performs the following steps:
        1. Optimizes hyperparameters for various classification methods by invoking `optimal_hyperparameters`.
        2. Trains each optimized model on the training data.
        3. Makes predictions on both training and testing datasets.
        4. Evaluates and stores performance metrics for each model on both datasets.
        5. Prepares data for visualization or further analysis.

        :param find_optimal_hyperparameters: Whether to perform hyperparameters optimization before training.
        :type find_optimal_hyperparameters: bool, default=False
    
        :param depth_max: Maximum depth for tree-based models such as Random Forest and Extra Trees.
        :type depth_max: int, default=10
    
        :return: 
            A tuple containing:
            - metrics_results: Dictionary containing the metrics computed on training and testing datasets
            - predictions: Dictionary containing predictions for each method on training and testing datasets
            - models_dict: Dictionary containing the trained models.
        
        :rtype: tuple (dict, dict)
        """

        if find_optimal_hyperparameters:
            _ = self.optimal_hyperparameters(
                depth_max=depth_max
            )

        treemod = DecisionTreeClassifier()

        metrics_results = {

        "accuracy" : { 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "f1-score" : { 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "recall" :{ 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "precision" :{ 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "roc_auc" : { 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            }
        }

        models_dict = {}
        predictions = {}
        
        for i, method in enumerate(self.methods):
        
            model = self.models[method]
            model.fit(self.data.X_train, self.data.y_train)

            y_train_pred = model.predict(self.data.X_train)
            y_test_pred = model.predict(self.data.X_test)

            predictions[method + "Train"] = y_train_pred
            predictions[method + "Test"] = y_test_pred
            
            for metric in list(self.metrics_dict.keys()):
                metrics_results[metric][method + " Train"].append(self.metrics_dict[metric](self.data.y_train, y_train_pred))
                metrics_results[metric][method + " Test"].append(self.metrics_dict[metric](self.data.y_test, y_test_pred))

            models_dict[method] = model

        return metrics_results, predictions, models_dict
    
    def TrainTestLogisticRegression(self, find_optimal_hyperparameters=False, depth_max=10):

        """
        Optimizes hyperparameters for all classification methods, trains each model, and evaluates their performance 
        on both training and testing datasets.
    
        This function performs the following steps:
        1. Optimizes hyperparameters for various classification methods by invoking `optimal_hyperparameters`.
        2. Trains each optimized model on the training data.
        3. Makes predictions on both training and testing datasets.
        4. Evaluates and stores performance metrics for each model on both datasets.
        5. Prepares data for visualization or further analysis.

        :param find_optimal_hyperparameters: Whether to perform hyperparameters optimization before training.
        :type find_optimal_hyperparameters: bool, default=False
    
        :param depth_max: Maximum depth for tree-based models such as Random Forest and Extra Trees.
        :type depth_max: int, default=10
    
        :return: 
            A tuple containing:
            - metrics_results: Dictionary containing the metrics computed on training and testing datasets
            - predictions: Dictionary containing predictions for each method on training and testing datasets
            - models_dict: Dictionary containing the trained models.
        
        :rtype: tuple (dict, dict)
        """

        # Optimization Logistic Regression
        best_params_logreg, _= self.logisticRegression()

        metrics_results = {

        "accuracy" : { 
            "LogReg Train":[],
            "LogReg Test" :[],
            },

        "f1-score" : { 
            "LogReg Train":[],
            "LogReg Test" :[],
            },

        "recall" :{ 
            "LogReg Train":[],
            "LogReg Test" :[],
            },

        "precision" :{ 
            "LogReg Train":[],
            "LogReg Test" :[],
            },

        "roc_auc" : { 
            "LogReg Train":[],
            "LogReg Test" :[],
            }
        }

        models_dict = {}
        predictions = {}
        
        model = LogisticRegression(**best_params_logreg, max_iter=10000)
        model.fit(self.data.X_train, self.data.y_train)

        y_train_pred = model.predict(self.data.X_train)
        y_test_pred = model.predict(self.data.X_test)
        
        predictions["LogReg Train"] = y_train_pred
        predictions["LogReg Test"] = y_test_pred
            
        for metric in list(self.metrics_dict.keys()):
            metrics_results[metric]["LogReg Train"].append(self.metrics_dict[metric](self.data.y_train, y_train_pred))
            metrics_results[metric]["LogReg Test"].append(self.metrics_dict[metric](self.data.y_test, y_test_pred))

        models_dict["LogReg"] = model

        return metrics_results, predictions, models_dict
    

    def CrossValidationKFold(self, n_splits=5, shuffle=True, depth_max=10):

        """
        Optimizes hyperparameters for all classification methods, trains each model and evaluates their performance using KFold,
        stratified or not depending on the class initialization
    
        This function performs the following steps:
        1. Optimizes hyperparameters for various classification methods by invoking `optimal_hyperparameters`.
        2. Trains each optimized model on the training data.
        3. Makes predictions on both training and testing datasets.
        4. Evaluates and stores performance metrics for each model on both datasets.
        5. Prepares data for visualization or further analysis.
    
        :param n_splits: Number of splits for cross-validation.
        :type n_splits: int, default=5
    
        :param shuffle: Whether to shuffle the data before splitting into batches.
        :type shuffle: bool, default=False
    
        :param depth_max: Maximum depth for tree-based models such as Random Forest.
        :type depth_max: int, default=10
    
        :return: metrics_results: Dictionary containing the metrics computed on each fold
        :rtype: dict
        """

        results = self.optimal_hyperparameters(
            depth_max=depth_max
        )
        
        if self.data.stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle)

        # kf = TimeSeriesSplit(n_splits=5)

        treemod = DecisionTreeClassifier()

        metrics_results = {

        "accuracy" : { 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "f1-score" : { 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "recall" :{ 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "precision" :{ 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            },

        "roc_auc" : { 
            "LogReg Train":[], "ClassificationDecisionTree Train" : [], "RandomForest Train":[],
            "LogReg Test" :[], "ClassificationDecisionTree Test" : [], "RandomForest Test":[],
            }
        }
        
        for i, method in enumerate(self.methods):

            # for train_index, test_index in kf.split(self.data.X_train_val_untouched): # for timeSeriesSplit
            for train_index, test_index in kf.split(self.data.X_train_val_untouched, self.data.y_train_val_untouched):
                X_train, X_test = self.data.X_train_val_untouched[train_index], self.data.X_train_val_untouched[test_index]
                y_train, y_test = self.data.y_train_val_untouched[train_index], self.data.y_train_val_untouched[test_index]

                scaler = StandardScaler() if self.data.scaling_method == 'standard' else MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                if self.data.dim_reduction == "PCA":
                    pca = PCA(n_components=self.data.pca_n_components)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)

                model = self.models[method]
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                for metric in list(self.metrics_dict.keys()):
                    metrics_results[metric][method + " Train"].append(self.metrics_dict[metric](y_train, y_train_pred))
                    metrics_results[metric][method + " Test"].append(self.metrics_dict[metric](y_test, y_test_pred))
        
        return metrics_results

    def check_variable_in_list(self, var, list):
        """
        Checks if a variable is present in a given list and raises an error if not.
        
        :param var: the variable to check.
        :type var: Any
        
        :param list: the list to search within.
        :type list: list
        
        :raises ValueError: if `var` is not found in `list`.
        """
        if var not in list:
            raise ValueError(f"The variable '{var}' is not present. Choose a name among {list}")
        pass

    def metricBoxplot(self, data_to_plot, labels, metric_name="Accuracy"):
        """
        Plots a Boxplot comparison of a chosen metric for all classification methods

        :param data_to_plot: contains the metric values for all methods for each Fold
        :type data_to_plot: dict
        :param labels: contains the methods names
        :type labels: list
        :param metric_name: Name of the metric being plotted.
        :type metric_name: str, default="Accuracy"
        """
        plt.figure(figsize=(11, 6))
        plt.boxplot(data_to_plot, patch_artist=True, boxprops=dict(facecolor="lightblue"),showmeans=True)
        plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
        plt.title(f'Comparison Train and Test set. Kfold {metric_name}')
        position_ligne = 12.5
        plt.axvline(x=position_ligne, color='red', linestyle='--', linewidth=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def BoxplotsAllMetrics(self, metrics_results):
        """
        Plots Boxplots comparisons of all metrics for all classification methods

        :param metrics_results: contains the metrics values for all methods for each Fold
        :type metrics_results: dict
        """
        for metric in list(self.metrics_dict.keys()):
            data_to_plot = list(metrics_results[metric].values())
            labels = list(metrics_results[metric].keys())
            self.metricBoxplot(data_to_plot, labels, metric)

    def ROC_curve (self, method_name = "RandomForest"):
        """
        Plots the ROC curve of the desired method

        :param method_name: the method name
        :type method_name: string, default="RandomForest"
        """
        self.check_variable_in_list(method_name, self.methods)
        y_proba = self.models[method_name].predict_proba(self.data.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.data.y_test, y_proba);
        figure = plt.figure(figsize=(4,2))
        plt.plot(fpr,tpr, linewidth = 2)
        plt.title(f'Roc curve {method_name}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.plot(tpr,tpr,"k--", linewidth = 2)
        plt.grid(linestyle = 'dashed')
        plt.show()

    def ROC_curve_all_methods (self):
        """
        Plots ROC curves of all methods
        """
        num_methods = len(self.methods)
        ncols = 4
        nrows = math.ceil(num_methods / ncols) 
        
        # Subplot creation
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten()
        
        for idx, method in enumerate(self.methods):
            y_proba = self.models[method].predict_proba(self.data.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.data.y_test, y_proba)
            
            ax = axes[idx]
            ax.plot(fpr, tpr, linewidth=2, label='ROC curve')
            ax.set_title(f'ROC Curve: {method}')
            ax.set_xlabel('False Positive Rate (FPR)')
            ax.set_ylabel('True Positive Rate (TPR)')
            ax.plot([0, 1], [0, 1], "k--", linewidth=2, label='Random Guess')
            ax.grid(linestyle='dashed')
            ax.legend()
        
        # If not multiple of 4, hide subplots not used
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def createMeansDataframe(self, results_metrics, method_names):
        """
        Creates two datasets (one on training sets, one on validation sets)
        of all metrics for all classification methods

        :param metrics_results: contains the metrics values for all methods for each Fold
        :type metrics_results: dict
        :param method_names: contains the methods names
        :type method_names: list
        :return: two DataFrames with mean metrics for training and testing sets.
        :rtype: tuple
        """
        metrics = self.metrics_dict.keys()

        means_train_dict = {}
        means_test_dict = {}
        
        for metric in metrics:
            if metric not in results_metrics:
                raise ValueError(f"Metric '{metric}' is not present in results_metrics.")
            
            mean_values = []
            for i in range  (len(self.methods)):
                # Check if the method exists in the current metric
                if method_names[i] in results_metrics[metric]:
                    scores = results_metrics[metric][method_names[i]]
                    if scores:
                        mean = np.mean(scores)
                    else:
                        mean = np.nan  # Assign NaN if the score list is empty
                else:
                    mean = np.nan  # Assign NaN if the method is missing for the metric
                mean_values.append(mean)
            
            column_name = f'Mean {metric.capitalize()}'
            means_train_dict[column_name] = mean_values

        for metric in metrics:
            if metric not in results_metrics:
                raise ValueError(f"Metric '{metric}' is not present in results_metrics.")
            
            mean_values = []
            for i in range  (len(self.methods), len(method_names)):
                # Check if the method exists in the current metric
                if method_names[i] in results_metrics[metric]:
                    scores = results_metrics[metric][method_names[i]]
                    if scores:
                        mean = np.mean(scores)
                    else:
                        mean = np.nan  # Assign NaN if the score list is empty
                else:
                    mean = np.nan  # Assign NaN if the method is missing for the metric
                mean_values.append(mean)
            
            column_name = f'Mean {metric.capitalize()}'
            means_test_dict[column_name] = mean_values

        df_train = pd.DataFrame(means_train_dict, index=method_names[:3])
        df_test = pd.DataFrame(means_test_dict, index=method_names[3:])
        
        return df_train, df_test

    def get_best_method(self, df, metric, ens="Test"):
        """
        Returns the name of the method with the best performance for a given metric,
        excluding 'Train' from the names.
        
        :param df: DataFrame containing the metrics for different methods
        :type df: pandas.DataFrame
        :param metric: The metric column to consider for selecting the best method
        :type metric: str
        :param ens: Specifies which ensemble we are working on (Train or Test)
        :type ens: str, default="Test"
        :return: The name of the best method without 'Train'
        :rtype: str
        """
        if ("Mean " + metric) not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame.")
        
        best_method = df.loc[df["Mean " + metric].idxmax()].name
        return best_method.replace(' ' + ens, '')

    def evaluate_model(self, model):
        """
        Evaluates the specified model by calculating and displaying the classification report.
        
        :param model: model to evaluate
        :type model: sklearn.base.BaseEstimator
        """
        y_pred = model.predict(self.data.X_test)
        report = classification_report(self.data.y_test, y_pred)
        print("Classification Report:")
        print(report)

    def get_metrics(self, model, method_name, metrics_results):
        """
        Calculates and updates evaluation metrics for a given model and method.
    
        This method uses the provided model to make predictions on the test dataset,
        computes various evaluation metrics as defined in the metrics dictionary,
        and updates the `metrics_results` dictionary with the computed values for the specified method.
    
        :param model: The trained model to evaluate.
        :type model: sklearn.base.BaseEstimator
        :param method_name: The name identifier for the evaluation method or model variant.
        :type method_name: str
        :param metrics_results: A dictionary containing metrics to be updated with the new results.
        :type metrics_results: dict
        :return: The updated `metrics_results` dictionary with the new metric values for the given method.
        :rtype: dict
        """
        y_pred = model.predict(self.data.X_test)

        for metric in list(self.metrics_dict.keys()):
             metrics_results[metric][method_name] = self.metrics_dict[metric](self.data.y_test, y_pred)
        
        return metrics_results
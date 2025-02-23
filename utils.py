from Classification import classification_class as classification
from Features_extraction import feature_extraction_class as feature_extraction
from Data_processing.images_processing_class import ImagesProcessing
import pandas as pd

def perform_classification(method_name, df_Y, 
                           name_best_models, metrics_results_best_methods, best_params_method=None,
                            feature_extraction_method=None, df_features=None, dim_reduction = None):
    """
    Perform feature extraction, classification, and evaluation.

    :param method_name: Name of the feature extraction method.
    :type method_name: str

    :param best_params_method: parameters to use for the Feature Extraction method
    :type best_params_method: dict

    :param df_Y: Dataframe containing labels for classification.
    :type df_Y: pd.DataFrame

    :param name_best_models: Dictionary storing the best models per method.
    :type name_best_models: dict

    :param metrics_results_best_methods: Dictionary storing classification results.
    :type metrics_results_best_methods: dict

    :param feature_extraction_method: Function to extract features from images.
    :type feature_extraction_method: function

    :param df_features: DataFrame containing features
    :type df_features: pd.DataFrame

    :param dim_reduction: dimension reduction method to use on the features
    :type dim_reduction: str, default=None

    :return: Updated dictionary with classification metrics for the best model.
    :rtype: dict

    """

    # Extract features
    if df_features is None: # Possible to give directly features or to extract features
        if best_params_method is None:
            df_features = pd.DataFrame(feature_extraction_method())
        else:
            df_features = pd.DataFrame(feature_extraction_method(**best_params_method))

    print(f"Performing Classification for {method_name}")

    # Data Processing & Classification
    data = classification.DataProcessing(df_features, df_Y, stratified=False, dim_reduction=dim_reduction)
    env = classification.BinaryClassification(data, average="macro")

    # Cross Validation
    metrics_results = env.CrossValidationKFold()
    labels = list(metrics_results['f1-score'].keys())

    # Create Dataframes for Results
    results_train_KFold, results_test_KFold = env.createMeansDataframe(metrics_results, labels)
    results_train_KFold.style.highlight_max(axis=0)

    # Identify Best Method
    best_method_name = env.get_best_method(results_test_KFold, "F1-score", ens="Test")
    print(f"Best method name for {method_name}: {best_method_name}")

    # Store Best Model Name
    name_best_models[method_name] = best_method_name

    # Train and Evaluate the Best Model
    metrics_results, predictions, models = env.TrainTest()
    env.evaluate_model(models[best_method_name])

    # If PCA we return also the number of components used
    if dim_reduction == "PCA":
        return env.get_metrics(models[best_method_name], method_name, metrics_results_best_methods), df_features, data.pca_n_components
    else:
        return env.get_metrics(models[best_method_name], method_name, metrics_results_best_methods), df_features
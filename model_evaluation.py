from typing import Callable, Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def calculate_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Args:
        targets: True target labels.
        preds: Predicted class labels.

    Returns:
        Accuracy score.
    """
    return accuracy_score(targets, preds)


def plot_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    name: str = ""
) -> None:
    """
    Plot a normalized confusion matrix.

    Args:
        targets: True target labels.
        preds: Predicted class labels.
        name: Name of the dataset or model used in the plot title.
    """
    cf = confusion_matrix(targets, preds, normalize="true")

    plt.figure()
    sns.heatmap(cf, annot=True, fmt=".2f")
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.title(f"{name} Confusion Matrix")
    plt.show()


def calculate_precision(
    targets: np.ndarray,
    preds: np.ndarray
) -> float:
    """
    Calculate precision.

    Args:
        targets: True target labels.
        preds: Predicted class labels.

    Returns:
        Precision score.
    """
    return precision_score(targets, preds, zero_division=0)


def calculate_recall(
    targets: np.ndarray,
    preds: np.ndarray
) -> float:
    """
    Calculate recall.

    Args:
        targets: True target labels.
        preds: Predicted class labels.

    Returns:
        Recall score.
    """
    return recall_score(targets, preds, zero_division=0)


def calculate_f1(
    targets: np.ndarray,
    preds: np.ndarray
) -> float:
    """
    Calculate the F1 score.

    Args:
        targets: True target labels.
        preds: Predicted class labels.

    Returns:
        F1 score.
    """
    return f1_score(targets, preds, zero_division=0)


def calculate_auroc(
    targets: np.ndarray,
    pred_proba: np.ndarray
) -> float:
    """
    Calculate the Area Under the ROC Curve (AUROC).

    Args:
        targets: True target labels.
        pred_proba: Predicted probabilities for the positive class.

    Returns:
        AUROC score.
    """
    fpr, tpr, _ = roc_curve(targets, pred_proba)

    return auc(fpr, tpr)


def plot_roc_curve(
    targets: np.ndarray,
    pred_proba: np.ndarray,
    name: str = ""
) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
        targets: True target labels.
        pred_proba: Predicted probabilities for the positive class.
        name: Name of the dataset or model used in the plot title.
    """
    fpr, tpr, _ = roc_curve(targets, pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        lw=2,
        linestyle="--"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curve for {name}")
    plt.legend(loc="lower right")
    plt.show()


def calculate_average_precision(
    targets: np.ndarray,
    pred_proba: np.ndarray
) -> float:
    """
    Calculate Average Precision (AP).

    Args:
        targets: True target labels.
        pred_proba: Predicted probabilities for the positive class.

    Returns:
        Average Precision score.
    """
    return average_precision_score(targets, pred_proba)


def plot_precision_recall_curve(
    targets: np.ndarray,
    pred_proba: np.ndarray,
    name: str = ""
) -> None:
    """
    Plot the Precision-Recall curve.

    Args:
        targets: True target labels.
        pred_proba: Predicted probabilities for the positive class.
        name: Name of the dataset or model used in the plot title.
    """
    precision, recall, _ = precision_recall_curve(
        targets,
        pred_proba
    )

    average_precision = average_precision_score(
        targets,
        pred_proba
    )

    plt.figure()
    plt.plot(
        recall,
        precision,
        lw=2,
        label=f"PR curve (AP = {average_precision:.2f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {name}")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()


def print_classification_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    name: str = ""
) -> None:
    """
    Calculate and print classification metrics.

    The metrics include accuracy, precision, recall, and F1 score.

    Args:
        targets: True target labels.
        preds: Predicted class labels.
        name: Name of the dataset or model.
    """
    accuracy = calculate_accuracy(targets, preds)
    precision = calculate_precision(targets, preds)
    recall = calculate_recall(targets, preds)
    f1 = calculate_f1(targets, preds)

    print(f"Accuracy for {name}: {accuracy * 100:.2f}%")
    print(f"Precision for {name}: {precision * 100:.2f}%")
    print(f"Recall for {name}: {recall * 100:.2f}%")
    print(f"F1 score for {name}: {f1 * 100:.2f}%")


def print_probability_metrics(
    targets: np.ndarray,
    pred_proba: np.ndarray,
    name: str = ""
) -> None:
    """
    Calculate and print probability-based classification metrics.

    The metrics include AUROC and Average Precision.

    Args:
        targets: True target labels.
        pred_proba: Predicted probabilities for the positive class.
        name: Name of the dataset or model.
    """
    roc_auc = calculate_auroc(targets, pred_proba)
    average_precision = calculate_average_precision(
        targets,
        pred_proba
    )

    print(f"AUROC for {name}: {roc_auc:.2f}")
    print(f"Average Precision for {name}: {average_precision:.2f}")


def predict_data(
    model: ClassifierMixin,
    inputs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate class predictions and positive-class probabilities.

    Args:
        model: Trained classification model with predict() and
            predict_proba() methods.
        inputs: Processed input features.

    Returns:
        A tuple containing:
            - Predicted class labels.
            - Predicted probabilities for the positive class.
    """
    preds = model.predict(inputs)
    pred_proba = model.predict_proba(inputs)[:, 1]

    return preds, pred_proba


def evaluate_dataset(
    targets: np.ndarray,
    preds: np.ndarray,
    pred_proba: np.ndarray,
    name: str = ""
) -> None:
    """
    Calculate, print, and visualize all classification metrics.

    Args:
        targets: True target labels.
        preds: Predicted class labels.
        pred_proba: Predicted probabilities for the positive class.
        name: Name of the dataset, for example "Training" or "Test".
    """
    print_classification_metrics(
        targets,
        preds,
        name
    )

    print_probability_metrics(
        targets,
        pred_proba,
        name
    )

    plot_confusion_matrix(
        targets,
        preds,
        name
    )

    plot_roc_curve(
        targets,
        pred_proba,
        name
    )

    plot_precision_recall_curve(
        targets,
        pred_proba,
        name
    )


def model_evaluate(
    model: ClassifierMixin,
    train_inputs: np.ndarray,
    test_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and evaluate a trained classification model.

    The function generates predictions and probabilities for both
    training and test datasets, then calculates and visualizes
    classification metrics.

    Args:
        model: Trained classification model with predict() and
            predict_proba() methods.
        train_inputs: Processed training features.
        test_inputs: Processed test features.
        train_targets: Training target labels.
        test_targets: Test target labels.

    Returns:
        A tuple containing:
            - Training positive-class probabilities.
            - Test positive-class probabilities.
    """
    preds_train, prob_train = predict_data(
        model,
        train_inputs
    )

    preds_test, prob_test = predict_data(
        model,
        test_inputs
    )

    evaluate_dataset(
        train_targets,
        preds_train,
        prob_train,
        "Training"
    )

    evaluate_dataset(
        test_targets,
        preds_test,
        prob_test,
        "Test"
    )

    return prob_train, prob_test

ModelCollection = Union[
    Dict[str, ClassifierMixin],
    List[ClassifierMixin]
]

PreprocessFunction = Callable[..., tuple]


def calculate_model_metrics(
    model: ClassifierMixin,
    inputs: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics for a trained model.

    Args:
        model: Trained classification model.
        inputs: Processed input features.
        targets: True target labels.

    Returns:
        Dictionary containing accuracy, precision, recall, F1,
        AUROC, and Average Precision scores.
    """
    preds, pred_proba = predict_data(model, inputs)

    return {
        "Accuracy": calculate_accuracy(targets, preds),
        "Precision": calculate_precision(targets, preds),
        "Recall": calculate_recall(targets, preds),
        "F1": calculate_f1(targets, preds),
        "AUROC": calculate_auroc(targets, pred_proba),
        "Average Precision": calculate_average_precision(
            targets,
            pred_proba
        )
    }


def get_model_items(
    models: ModelCollection
) -> List[Tuple[str, ClassifierMixin]]:
    """
    Convert different model collection formats into a standard list.

    Args:
        models: Models to compare. Can be either:
            - dictionary: {"Logistic Regression": model, ...}
            - list: [model1, model2, ...]

    Returns:
        List of (model_name, model) pairs.
    """
    if isinstance(models, dict):
        return list(models.items())

    return [
        (model.__class__.__name__, model)
        for model in models
    ]


def process_data(
    raw_df: pd.DataFrame,
    encoding: str,
    preprocess_function: PreprocessFunction
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Preprocess data using the selected encoding method.

    This function provides a single entry point for preprocessing
    data before model training and evaluation.

    Args:
        raw_df: Original dataframe before preprocessing.
        encoding: Encoding method, for example "onehot" or "ordinal".
        preprocess_function: Function responsible for preprocessing
            the dataframe.

    Returns:
        A tuple containing:
            - Training input features.
            - Training target labels.
            - Test input features.
            - Test target labels.
    """
    (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        *_
    ) = preprocess_function(
        raw_df,
        encoding=encoding
    )

    return (
        train_inputs,
        train_targets,
        test_inputs,
        test_targets
    )


def train_model(
    model: ClassifierMixin,
    train_inputs: np.ndarray,
    train_targets: np.ndarray
) -> ClassifierMixin:
    """
    Clone and train a classification model.

    Args:
        model: Model to clone and train.
        train_inputs: Training input features.
        train_targets: Training target labels.

    Returns:
        A trained clone of the input model.
    """
    current_model = clone(model)

    current_model.fit(
        train_inputs,
        train_targets
    )

    return current_model


def create_model_result(
    model_name: str,
    encoding: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float]
) -> Dict[str, Union[str, float]]:
    """
    Create a result record for one model and encoding combination.

    Args:
        model_name: Name of the model.
        encoding: Encoding method used for preprocessing.
        train_metrics: Metrics calculated on the training set.
        test_metrics: Metrics calculated on the test set.

    Returns:
        Dictionary containing training and test metrics.
    """
    return {
        "Model": model_name,
        "Encoding": encoding,

        "Train Accuracy": train_metrics["Accuracy"],
        "Test Accuracy": test_metrics["Accuracy"],

        "Train Precision": train_metrics["Precision"],
        "Test Precision": test_metrics["Precision"],

        "Train Recall": train_metrics["Recall"],
        "Test Recall": test_metrics["Recall"],

        "Train F1": train_metrics["F1"],
        "Test F1": test_metrics["F1"],

        "Train AUROC": train_metrics["AUROC"],
        "Test AUROC": test_metrics["AUROC"],

        "Train AP": train_metrics["Average Precision"],
        "Test AP": test_metrics["Average Precision"]
    }


def compare_models_and_encodings(
    raw_df: pd.DataFrame,
    models: ModelCollection,
    encodings: List[str],
    preprocess_function: PreprocessFunction
) -> pd.DataFrame:
    """
    Train and compare multiple models using multiple encodings.

    Each model is trained independently for every encoding method.
    Training and test metrics are calculated and returned as a
    pandas DataFrame.

    Args:
        raw_df: Original dataframe before preprocessing.
        models: Models to compare. Can be either:
            - dictionary: {"Logistic Regression": model, ...}
            - list: [model1, model2, ...]
        encodings: Encoding methods to compare, for example
            ["onehot", "ordinal"].
        preprocess_function: Function responsible for preprocessing
            the data according to the selected encoding.

    Returns:
        DataFrame containing training and test metrics for every
        model-encoding combination.
    """
    model_items = get_model_items(models)
    results = []

    for encoding in encodings:

        print("=" * 70)
        print(f"ENCODING: {encoding}")
        print("=" * 70)

        (
            train_inputs,
            train_targets,
            test_inputs,
            test_targets
        ) = process_data(
            raw_df,
            encoding,
            preprocess_function
        )

        for model_name, model in model_items:

            print("\n" + "-" * 70)
            print(f"MODEL: {model_name}")
            print(f"ENCODING: {encoding}")
            print("-" * 70)

            trained_model = train_model(
                model,
                train_inputs,
                train_targets
            )

            train_metrics = calculate_model_metrics(
                trained_model,
                train_inputs,
                train_targets
            )

            test_metrics = calculate_model_metrics(
                trained_model,
                test_inputs,
                test_targets
            )

            result = create_model_result(
                model_name,
                encoding,
                train_metrics,
                test_metrics
            )

            results.append(result)

    return pd.DataFrame(results)

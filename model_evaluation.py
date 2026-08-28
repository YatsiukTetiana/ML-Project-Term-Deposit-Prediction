from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import ClassifierMixin
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
    train_inputs: np.ndarray | None = None,
    train_targets: np.ndarray | None = None,
    val_inputs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    test_inputs: np.ndarray | None = None,
    test_targets: np.ndarray | None = None
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Generate predictions and evaluate a trained classification model
    on any combination of training, validation, and test datasets.

    Args:
        model: Trained classification model with predict() and
            predict_proba() methods.

        train_inputs: Processed training features. Optional.
        train_targets: Training target labels. Optional.

        val_inputs: Processed validation features. Optional.
        val_targets: Validation target labels. Optional.

        test_inputs: Processed test features. Optional.
        test_targets: Test target labels. Optional.

    Returns:
        A tuple containing:
            - Training positive-class probabilities, or None.
            - Validation positive-class probabilities, or None.
            - Test positive-class probabilities, or None.
    """

    prob_train = None
    prob_val = None
    prob_test = None

    # Evaluate training dataset if provided
    if train_inputs is not None and train_targets is not None:
        preds_train, prob_train = predict_data(
            model,
            train_inputs
        )

        evaluate_dataset(
            train_targets,
            preds_train,
            prob_train,
            "Training"
        )

    # Evaluate validation dataset if provided
    if val_inputs is not None and val_targets is not None:
        preds_val, prob_val = predict_data(
            model,
            val_inputs
        )

        evaluate_dataset(
            val_targets,
            preds_val,
            prob_val,
            "Validation"
        )

    # Evaluate test dataset if provided
    if test_inputs is not None and test_targets is not None:
        preds_test, prob_test = predict_data(
            model,
            test_inputs
        )

        evaluate_dataset(
            test_targets,
            preds_test,
            prob_test,
            "Test"
        )

    return prob_train, prob_val, prob_test

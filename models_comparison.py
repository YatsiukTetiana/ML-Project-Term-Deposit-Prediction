from typing import Dict, List, Tuple, Union
from model_evaluation import model_evaluate

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def get_set_hyperparameters(model: ClassifierMixin) -> dict:
    """
    Return only hyperparameters explicitly changed from the
    estimator's default values.

    Args:
        model: Configured sklearn estimator.

    Returns:
        Dictionary containing only explicitly changed parameters.
    """
    model_params = model.get_params()

    default_model = model.__class__()
    default_params = default_model.get_params()

    return {
        parameter: value
        for parameter, value in model_params.items()
        if parameter in default_params
        and value != default_params[parameter]
    }


def compare_models_and_encodings(
    raw_df: pd.DataFrame,
    models: Union[
        Dict[str, ClassifierMixin],
        List[ClassifierMixin]
    ],
    process_data,
    encodings: List[str] = None,
    evaluate_train: bool = True,
    evaluate_val: bool = True,
    evaluate_test: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and compare multiple classification models using
    different categorical encoding methods.

    Each model is trained separately for each encoding.

    Evaluation can be performed on any combination of training,
    validation, and test datasets.

    Args:
        raw_df:
            Raw input DataFrame.

        models:
            Models to compare. Can be either:
            - dictionary: {"Logistic Regression": model, ...}
            - list: [model1, model2, ...]

        process_data:
            Data preprocessing function accepting:
            process_data(raw_df, education_encoding=...).

        encodings:
            Encoding methods to compare.
            Defaults to ["onehot", "ordinal"].

        evaluate_train:
            Whether to evaluate on the training dataset.

        evaluate_val:
            Whether to evaluate on the validation dataset.

        evaluate_test:
            Whether to evaluate on the test dataset.

    Returns:
        A tuple containing:

        results:
            Nested dictionary containing the probabilities returned
            by model_evaluate() for every encoding and model.

        comparison_df:
            DataFrame containing metrics for every evaluated
            dataset, model, and encoding.
    """

    if encodings is None:
        encodings = ["onehot", "ordinal"]

    # Convert list of models to a dictionary
    if isinstance(models, list):
        models = {
            model.__class__.__name__: model
            for model in models
        }

    results = []
    evaluation_results = {}

    for encoding in encodings:

        # ---------------------------------------------------------
        # Preprocess data for the current encoding
        # ---------------------------------------------------------

        (
            train_inputs,
            val_inputs,
            test_inputs,
            train_targets,
            val_targets,
            test_targets,
            preprocessor
        ) = process_data(
            raw_df,
            education_encoding=encoding
        )

        evaluation_results[encoding] = {}

        # ---------------------------------------------------------
        # Evaluate every model
        # ---------------------------------------------------------

        for model_name, model in models.items():

            print(
                f"\n{'=' * 70}\n"
                f"Encoding: {encoding} | Model: {model_name}\n"
                f"{'=' * 70}"
            )

            # Create a fresh model for this encoding
            fitted_model = clone(model)

            # Train model
            fitted_model.fit(
                train_inputs,
                train_targets
            )

            hyperparameters = get_set_hyperparameters(model)

            # -----------------------------------------------------
            # Evaluate selected datasets
            # -----------------------------------------------------

            prob_train, prob_val, prob_test = model_evaluate(
                fitted_model,

                train_inputs=train_inputs
                if evaluate_train else None,

                train_targets=train_targets
                if evaluate_train else None,

                val_inputs=val_inputs
                if evaluate_val else None,

                val_targets=val_targets
                if evaluate_val else None,

                test_inputs=test_inputs
                if evaluate_test else None,

                test_targets=test_targets
                if evaluate_test else None
            )

            # -----------------------------------------------------
            # Store evaluation results
            # -----------------------------------------------------

            evaluation_results[encoding][model_name] = {
                "prob_train": prob_train,
                "prob_val": prob_val,
                "prob_test": prob_test,
                "model": fitted_model,
                "preprocessor": preprocessor,
            }

            # -----------------------------------------------------
            # Generate metrics for selected datasets
            # -----------------------------------------------------

            # Training
            if evaluate_train:

                pred_train = fitted_model.predict(train_inputs)

                train_metrics = {
                    "Precision": precision_score(
                        train_targets,
                        pred_train,
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        train_targets,
                        pred_train,
                        zero_division=0
                    ),
                    "F1": f1_score(
                        train_targets,
                        pred_train,
                        zero_division=0
                    ),
                    "AUROC": roc_auc_score(
                        train_targets,
                        prob_train
                    ),
                    "Average Precision": average_precision_score(
                        train_targets,
                        prob_train
                    ),
                }

                results.append({
                    "Encoding": encoding,
                    "Model": model_name,
                    "Dataset": "Train",
                    "Hyperparameters": hyperparameters,
                    **train_metrics
                })

            # Validation
            if evaluate_val:

                pred_val = fitted_model.predict(val_inputs)

                val_metrics = {
                    "Precision": precision_score(
                        val_targets,
                        pred_val,
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        val_targets,
                        pred_val,
                        zero_division=0
                    ),
                    "F1": f1_score(
                        val_targets,
                        pred_val,
                        zero_division=0
                    ),
                    "AUROC": roc_auc_score(
                        val_targets,
                        prob_val
                    ),
                    "Average Precision": average_precision_score(
                        val_targets,
                        prob_val
                    ),
                }

                results.append({
                    "Encoding": encoding,
                    "Model": model_name,
                    "Dataset": "Validation",
                    "Hyperparameters": hyperparameters,
                    **val_metrics
                })

            # Test
            if evaluate_test:

                pred_test = fitted_model.predict(test_inputs)

                test_metrics = {
                    "Precision": precision_score(
                        test_targets,
                        pred_test,
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        test_targets,
                        pred_test,
                        zero_division=0
                    ),
                    "F1": f1_score(
                        test_targets,
                        pred_test,
                        zero_division=0
                    ),
                    "AUROC": roc_auc_score(
                        test_targets,
                        prob_test
                    ),
                    "Average Precision": average_precision_score(
                        test_targets,
                        prob_test
                    ),
                }

                results.append({
                    "Encoding": encoding,
                    "Model": model_name,
                    "Dataset": "Test",
                    "Hyperparameters": hyperparameters,
                    **test_metrics
                })

    comparison_df = pd.DataFrame(results)

    return evaluation_results, comparison_df

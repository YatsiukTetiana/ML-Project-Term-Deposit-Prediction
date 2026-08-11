from typing import Dict, List, Tuple, Union
from evaluate_init import model_evaluate

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
    encodings: List[str] = None
) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and compare multiple classification models using
    different categorical encoding methods.

    Each model is trained separately for each encoding.
    The existing model_evaluate() function is used to generate
    train/test probabilities and perform the full evaluation.

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

    Returns:
        A tuple containing:

        results:
            Nested dictionary containing the probabilities returned
            by model_evaluate() for every encoding and model.

        comparison_df:
            DataFrame containing train/test metrics for every
            model and encoding.
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

            # Use existing model_evaluate()
            prob_train, prob_test = model_evaluate(
                fitted_model,
                train_inputs,
                test_inputs,
                train_targets,
                test_targets
            )

            # Store everything returned by model_evaluate()
            evaluation_results[encoding][model_name] = {
                "prob_train": prob_train,
                "prob_test": prob_test,
                "model": fitted_model,
                "preprocessor": preprocessor,
            }

            # -----------------------------------------------------
            # Generate class predictions for the comparison table
            # -----------------------------------------------------

            pred_train = fitted_model.predict(train_inputs)
            pred_test = fitted_model.predict(test_inputs)

            # -----------------------------------------------------
            # Calculate train metrics
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # Calculate test metrics
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # Add rows to comparison table
            # -----------------------------------------------------

            results.append({
                "Encoding": encoding,
                "Model": model_name,
                "Dataset": "Train",
                "Hyperparameters": hyperparameters,
                **train_metrics
            })

            results.append({
                "Encoding": encoding,
                "Model": model_name,
                "Dataset": "Test",
                "Hyperparameters": hyperparameters,
                **test_metrics
            })

    comparison_df = pd.DataFrame(results)

    return evaluation_results, comparison_df
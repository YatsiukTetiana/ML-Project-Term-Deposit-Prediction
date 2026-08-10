from typing import Literal, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder
)


EducationEncoding = Literal["onehot", "ordinal"]


def create_inputs_and_targets(
    raw_df: pd.DataFrame,
    target_col: str = "y"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the raw dataframe into input features and target values.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw input dataframe.
    target_col : str, default="y"
        Name of the target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Input features and binary target values.
    """
    inputs = raw_df.drop(columns=[target_col])

    targets = raw_df[target_col].map({
        "no": 0,
        "yes": 1
    })

    return inputs, targets


def split_data(
    inputs: pd.DataFrame,
    targets: pd.Series,
    random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series
]:
    """
    Split the data into training, validation, and test sets.

    The resulting proportions are 60% train, 20% validation,
    and 20% test. Stratification is used to preserve the target
    class distribution.

    Parameters
    ----------
    inputs : pd.DataFrame
        Input features.
    targets : pd.Series
        Target values.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    Tuple
        train_inputs, val_inputs, test_inputs,
        train_targets, val_targets, test_targets.
    """
    (
        train_inputs,
        test_val_inputs,
        train_targets,
        test_val_targets
    ) = train_test_split(
        inputs,
        targets,
        test_size=0.4,
        stratify=targets,
        random_state=random_state
    )

    (
        test_inputs,
        val_inputs,
        test_targets,
        val_targets
    ) = train_test_split(
        test_val_inputs,
        test_val_targets,
        test_size=0.5,
        stratify=test_val_targets,
        random_state=random_state
    )

    return (
        train_inputs,
        val_inputs,
        test_inputs,
        train_targets,
        val_targets,
        test_targets
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for the bank marketing dataset.

    Creates:
    - prev_contact: whether the customer was contacted previously.
    - campaign_log: logarithmically transformed campaign value.

    Also replaces the special pdays value 999 with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional features.
    """
    df = df.copy()

    df["prev_contact"] = (df["pdays"] != 999).astype(int)

    df["pdays"] = df["pdays"].replace(999, np.nan)

    df["campaign_log"] = np.log1p(df["campaign"])

    return df


def identify_columns(
    df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns for preprocessing.

    The original `campaign` and `duration` columns are excluded
    from the numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Training input dataframe.

    Returns
    -------
    Tuple[List[str], List[str]]
        Numeric columns and categorical columns.
    """
    numeric_cols = (
        df.select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    numeric_cols.remove("campaign")
    numeric_cols.remove("duration")

    categorical_cols = (
        df.select_dtypes(include="object")
        .columns
        .tolist()
    )

    return numeric_cols, categorical_cols


def get_drop_categories(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> List[str]:
    """
    Determine which category should be dropped for each categorical column.

    If `unknown` exists in a column, it is dropped. Otherwise, the
    first observed category is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe.
    categorical_cols : List[str]
        Categorical column names.

    Returns
    -------
    List[str]
        Category to drop for each categorical column.
    """
    drop_categories = []

    for col in categorical_cols:
        categories = df[col].dropna().unique().tolist()

        if "unknown" in categories:
            drop_categories.append("unknown")
        else:
            drop_categories.append(categories[0])

    return drop_categories


def create_numeric_transformer() -> Pipeline:
    """
    Create the preprocessing pipeline for numeric features.

    Returns
    -------
    Pipeline
        Numeric imputation and scaling pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]
    )


def create_onehot_transformer(
    drop_categories: List[str]
) -> Pipeline:
    """
    Create the preprocessing pipeline for one-hot encoded features.

    Parameters
    ----------
    drop_categories : List[str]
        Categories to drop for each categorical feature.

    Returns
    -------
    Pipeline
        Categorical imputation and one-hot encoding pipeline.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unknown"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    drop=drop_categories,
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


def create_ordinal_transformer() -> Pipeline:
    """
    Create the preprocessing pipeline for the education column.

    The education categories are encoded according to their natural
    educational progression.

    Returns
    -------
    Pipeline
        Education imputation, ordinal encoding, and scaling pipeline.
    """
    education_order = [
        "illiterate",
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "professional.course",
        "university.degree"
    ]

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unknown"
                )
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[education_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=7
                )
            ),
            ("scaler", MinMaxScaler())
        ]
    )


def create_onehot_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    drop_categories: List[str]
) -> ColumnTransformer:
    """
    Create a preprocessor where all categorical features use OneHot encoding.

    Parameters
    ----------
    numeric_cols : List[str]
        Numeric feature names.
    categorical_cols : List[str]
        Categorical feature names.
    drop_categories : List[str]
        Categories to drop from each categorical feature.

    Returns
    -------
    ColumnTransformer
        Complete preprocessing transformer.
    """
    numeric_transformer = create_numeric_transformer()

    categorical_transformer = create_onehot_transformer(
        drop_categories
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            )
        ]
    )


def create_ordinal_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    drop_categories: List[str]
) -> ColumnTransformer:
    """
    Create a preprocessor with ordinal encoding for education
    and OneHot encoding for the remaining categorical features.

    Parameters
    ----------
    numeric_cols : List[str]
        Numeric feature names.
    categorical_cols : List[str]
        Categorical feature names.
    drop_categories : List[str]
        Categories to drop from the non-education categorical features.

    Returns
    -------
    ColumnTransformer
        Complete preprocessing transformer.
    """
    numeric_transformer = create_numeric_transformer()

    ordinal_transformer = create_ordinal_transformer()

    one_hot_columns = [
        col
        for col in categorical_cols
        if col != "education"
    ]

    categorical_transformer = create_onehot_transformer(
        drop_categories
    )

    return ColumnTransformer(
        transformers=[
            (
                "ord",
                ordinal_transformer,
                ["education"]
            ),
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                categorical_transformer,
                one_hot_columns
            )
        ]
    )


def create_preprocessor(
    train_inputs: pd.DataFrame,
    education_encoding: EducationEncoding
) -> ColumnTransformer:
    """
    Create the appropriate preprocessing transformer.

    Parameters
    ----------
    train_inputs : pd.DataFrame
        Training input data. Used to determine feature columns and
        categories to drop.
    education_encoding : {"onehot", "ordinal"}
        Encoding method for the education column.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.

    Raises
    ------
    ValueError
        If an unsupported education encoding is provided.
    """
    numeric_cols, categorical_cols = identify_columns(
        train_inputs
    )

    if education_encoding == "onehot":
        drop_categories = get_drop_categories(
            train_inputs,
            categorical_cols
        )

        return create_onehot_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            drop_categories=drop_categories
        )

    if education_encoding == "ordinal":
        one_hot_columns = [
            col
            for col in categorical_cols
            if col != "education"
        ]

        drop_categories = get_drop_categories(
            train_inputs,
            one_hot_columns
        )

        return create_ordinal_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            drop_categories=drop_categories
        )

    raise ValueError(
        "education_encoding must be either 'onehot' or 'ordinal'."
    )


def transform_data(
    preprocessor: ColumnTransformer,
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    test_inputs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit the preprocessor on training data and transform all datasets.

    The original row indices and generated feature names are preserved.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Configured preprocessing transformer.
    train_inputs : pd.DataFrame
        Training input data.
    val_inputs : pd.DataFrame
        Validation input data.
    test_inputs : pd.DataFrame
        Test input data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Processed training, validation, and test data.
    """
    train_array = preprocessor.fit_transform(train_inputs)
    val_array = preprocessor.transform(val_inputs)
    test_array = preprocessor.transform(test_inputs)

    feature_names = preprocessor.get_feature_names_out()

    train_processed = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=train_inputs.index
    )

    val_processed = pd.DataFrame(
        val_array,
        columns=feature_names,
        index=val_inputs.index
    )

    test_processed = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=test_inputs.index
    )

    return (
        train_processed,
        val_processed,
        test_processed
    )


def process_data(
    raw_df: pd.DataFrame,
    education_encoding: EducationEncoding = "onehot",
    target_col: str = "y",
    random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    ColumnTransformer
]:
    """
    Process the raw bank marketing dataset.

    The function:
    1. Creates input features and targets.
    2. Splits the data into train, validation, and test sets.
    3. Adds engineered features to each split.
    4. Identifies numeric and categorical features.
    5. Creates the requested preprocessing pipeline.
    6. Fits preprocessing only on the training data.
    7. Transforms train, validation, and test data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw bank marketing dataframe.
    education_encoding : {"onehot", "ordinal"}, default="onehot"
        Encoding method used for the `education` column.
    target_col : str, default="y"
        Name of the target column.
    random_state : int, default=42
        Random seed used for the train/validation/test split.

    Returns
    -------
    Tuple
        train_processed : pd.DataFrame
            Processed training features.
        val_processed : pd.DataFrame
            Processed validation features.
        test_processed : pd.DataFrame
            Processed test features.
        train_targets : pd.Series
            Training targets.
        val_targets : pd.Series
            Validation targets.
        test_targets : pd.Series
            Test targets.
        preprocessor : ColumnTransformer
            Fitted preprocessing transformer.
    """
    # 1. Create inputs and targets
    inputs, targets = create_inputs_and_targets(
        raw_df,
        target_col
    )

    # 2. Split data
    (
        train_inputs,
        val_inputs,
        test_inputs,
        train_targets,
        val_targets,
        test_targets
    ) = split_data(
        inputs,
        targets,
        random_state
    )

    # 3. Feature engineering
    train_inputs = add_features(train_inputs)
    val_inputs = add_features(val_inputs)
    test_inputs = add_features(test_inputs)

    # 4. Create preprocessing pipeline
    preprocessor = create_preprocessor(
        train_inputs,
        education_encoding
    )

    # 5. Transform data
    (
        train_processed,
        val_processed,
        test_processed
    ) = transform_data(
        preprocessor,
        train_inputs,
        val_inputs,
        test_inputs
    )

    return (
        train_processed,
        val_processed,
        test_processed,
        train_targets,
        val_targets,
        test_targets,
        preprocessor
    )

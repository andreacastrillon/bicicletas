import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()

    #df = clean_dataset(df)

    feature_columns = ["yr", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "hum",
                       "windspeed", "last_week_same_hour","last_day_same_hour"]
    target_column = "cnt"
    y = df[target_column]
    X = df[feature_columns]

    train_indices = X["yr"] == 0
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[~train_indices], y[~train_indices]

    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _fix_month
        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def _fix_month(df):
    return ...



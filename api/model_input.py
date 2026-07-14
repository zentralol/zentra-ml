"""Helpers for preserving model feature metadata during preprocessing."""

import pandas as pd


def transform_preserving_columns(transformer, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a width-preserving transformer without dropping DataFrame metadata."""
    values = transformer.transform(frame)
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)

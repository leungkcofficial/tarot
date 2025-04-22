import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def calculate_egfr(df: pd.DataFrame, cr_col: str = 'Cr', gender_col: str = 'gender', dob_col: str = 'dob', date_col: str = 'date', age_col: str = 'age') -> pd.DataFrame:
    """
    Calculate eGFRcr using CKD-EPI 2021 equation and return the DataFrame with an added 'eGFRcr' column.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'Cr', 'gender', 'dob', 'date', and optionally 'age'.
        cr_col (str): Column name for creatinine in µmol/L.
        gender_col (str): Column name for gender ('M' or 'F').
        dob_col (str): Column name for date of birth.
        date_col (str): Column name for measurement date.
        age_col (str): Column name for age (in years).

    Returns:
        pd.DataFrame: DataFrame with additional 'eGFRcr' column.
    """
    df = df.copy()

    # Convert creatinine from µmol/L to mg/dL
    df['Cr_mg_dL'] = df[cr_col].astype(float) / 88.4

    # Use existing age if available, otherwise calculate it
    if age_col in df.columns:
        df['age'] = pd.to_numeric(df[age_col], errors='coerce')
    else:
        df['age'] = np.nan

    missing_age_mask = df['age'].isnull()
    if missing_age_mask.any():
        df.loc[missing_age_mask, 'age'] = (
            pd.to_datetime(df.loc[missing_age_mask, date_col]) - pd.to_datetime(df.loc[missing_age_mask, dob_col])
        ).dt.days / 365.25

    # Assign gender-specific parameters
    df['kappa'] = np.where(df[gender_col] == 'F', 0.7, 0.9)
    df['alpha'] = np.where(df[gender_col] == 'F', -0.241, -0.302)

    # Calculate eGFR
    ratio = df['Cr_mg_dL'] / df['kappa']
    df['eGFRcr'] = 142 * ratio.clip(upper=1) ** df['alpha'] * \
                  ratio.clip(lower=1) ** (-1.2) * 0.9938 ** df['age']

    # Female correction
    df.loc[df[gender_col] == 'F', 'eGFRcr'] *= 1.012

    # Round age
    df['age'] = df['age'].round().astype(int)

    return df.drop(columns=['Cr_mg_dL', 'kappa', 'alpha'])

class CKDEstimator(BaseEstimator, TransformerMixin):
    """
    Adds eGFRcr column using CKD-EPI 2021 formula.
    """
    def __init__(self, cr_col='Cr', gender_col='gender', dob_col='dob', date_col='date', age_col='age'):
        self.cr_col = cr_col
        self.gender_col = gender_col
        self.dob_col = dob_col
        self.date_col = date_col
        self.age_col = age_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X['Cr_mg_dL'] = X[self.cr_col].astype(float) / 88.4

        if self.age_col in X.columns:
            X['age'] = pd.to_numeric(X[self.age_col], errors='coerce')
        else:
            X['age'] = np.nan

        missing_age_mask = X['age'].isnull()
        if missing_age_mask.any():
            X.loc[missing_age_mask, 'age'] = (
                pd.to_datetime(X.loc[missing_age_mask, self.date_col]) - pd.to_datetime(X.loc[missing_age_mask, self.dob_col])
            ).dt.days / 365.25

        X['kappa'] = np.where(X[self.gender_col] == 'F', 0.7, 0.9)
        X['alpha'] = np.where(X[self.gender_col] == 'F', -0.241, -0.302)

        ratio = X['Cr_mg_dL'] / X['kappa']
        X['eGFRcr'] = 142 * ratio.clip(upper=1) ** X['alpha'] * \
                      ratio.clip(lower=1) ** (-1.2) * 0.9938 ** X['age']

        X.loc[X[self.gender_col] == 'F', 'eGFRcr'] *= 1.012

        X['age'] = X['age'].round().astype(int)

        return X.drop(columns=['Cr_mg_dL', 'kappa', 'alpha'])

class CKDLabelAdder(BaseEstimator, TransformerMixin):
    """
    Adds CKD stage labels 'A_class' and 'G_class' based on UACR and eGFR values.
    """
    def __init__(self, uacr_col='UACR_mg_g', egfr_col='eGFRcr'):
        self.uacr_col = uacr_col
        self.egfr_col = egfr_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # GFR classification
        bins_g = [-np.inf, 15, 30, 45, 60, 90, np.inf]
        labels_g = ['G5', 'G4', 'G3b', 'G3a', 'G2', 'G1']
        X['G_class'] = pd.cut(X[self.egfr_col], bins=bins_g, labels=labels_g, right=False)

        # Albuminuria classification
        bins_a = [-np.inf, 30, 300, np.inf]
        labels_a = ['A1', 'A2', 'A3']
        X['A_class'] = pd.cut(X[self.uacr_col], bins=bins_a, labels=labels_a, right=False)

        return X
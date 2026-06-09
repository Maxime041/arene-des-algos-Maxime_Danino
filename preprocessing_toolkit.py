import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataCleaner:
    """Apprend le nettoyage sur le train, le rejoue à l'identique sur toute donnée nouvelle."""
    
    def __init__(self):
        self.medians_ = {}
        self.scaler_ = StandardScaler()
        self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
        self.cols_to_drop_ = ['customerID', 'TotalCharges'] # TotalCharges a un VIF trop élevé
        
        self.num_cols_ = []
        self.cat_cols_ = []

    def fit(self, df):
        X = df.drop(columns=[c for c in self.cols_to_drop_ if c in df.columns])
        self.num_cols_ = X.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.num_cols_:
            self.medians_[col] = X[col].median()

        X_num_filled = X[self.num_cols_].fillna(self.medians_)
        self.scaler_.fit(X_num_filled)

        if self.cat_cols_:
            self.ohe_.fit(X[self.cat_cols_].fillna('Missing'))

        return self

    def transform(self, df):
        X = df.drop(columns=[c for c in self.cols_to_drop_ if c in df.columns])

        X_num_filled = X[self.num_cols_].fillna(self.medians_)
        X_num_scaled = pd.DataFrame(
            self.scaler_.transform(X_num_filled), 
            columns=self.num_cols_, 
            index=X.index
        )

        if self.cat_cols_:
            X_cat_encoded = pd.DataFrame(
                self.ohe_.transform(X[self.cat_cols_].fillna('Missing')),
                columns=self.ohe_.get_feature_names_out(self.cat_cols_),
                index=X.index
            )
            return pd.concat([X_num_scaled, X_cat_encoded], axis=1)
        
        return X_num_scaled

    def fit_transform(self, df):
        return self.fit(df).transform(df)
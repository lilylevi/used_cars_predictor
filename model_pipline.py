
import os
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import numpy as np
import config
from utils import save_pipeline


def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)


# --------------------
# 1. Define function that builds your model
# --------------------

def build_XGBoost_model():
    return xgb.XGBRegressor(learning_rate=0.1, n_estimators=100, early_stopping_rounds=50, enable_categorical=True, eval_metric="rmse")


def create_pipline(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def arrangeData(data: pd.DataFrame) -> pd.DataFrame:

    data.dropna(subset=['price'], inplace=True)
    data.drop(columns=['index', 'dateCrawled', 'nrOfPictures', 'name', 'monthOfRegistration', 'postalCode',
                            'notRepairedDamage'], inplace=True)
    # Filter out Outlier prices
    data = data[(data['price'] > 100) & (data['price'] < 200000)]

    # Convert yearOfRegistration to vehicle age
    current_year = datetime.today().year
    data['vehicle_age'] = current_year - data['yearOfRegistration']
    data.drop(columns=['yearOfRegistration'], inplace=True)

    # Filter out unrealistic ages
    data = data[(data['vehicle_age'] >= 0) & (data['vehicle_age'] <= 70)]
    return data


def train_model(data):

    X = data[config.FEATURES]
    y = data[config.TARGET]

    # 2. Split train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_features = X.select_dtypes(exclude=['object', 'category']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    model = build_XGBoost_model()

    pre_processor = create_pipline(num_features, cat_features)
    X_train = pre_processor.fit_transform(X_train)
    X_val = pre_processor.transform(X_val)

    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)

    # 4. Train the pipeline
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    save_pipeline(pre_processor,model)
    print(f"Candidate model trained and saved to {config.PIPELINE_PATH}")
    return model.evals_result()





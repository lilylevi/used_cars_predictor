
import os
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config
from utils import save_pipeline


def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)


# --------------------
# 1. Define function that builds your model
# --------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)   # regression output
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    return model




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


def train_model(data_path: str):
    """Load data from file and train model"""
    data = load_data(data_path)

    X = data[config.FEATURES]
    y = data[config.TARGET]

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_features = X.select_dtypes(exclude=['object', 'category']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    model_DNN = build_model()

    pre_processor = create_pipline(num_features, cat_features)
    X_train = pre_processor.fit_transform(X_train)

    # 4. Train the pipeline
    model_DNN.fit(X_train, y_train, validation_split=0.2)

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    save_pipeline(pre_processor,model_DNN)
    print(f"Candidate model trained and saved to {config.PIPELINE_PATH}")
    return config.PIPELINE_PATH


    # # 5. Predict on test data
    # y_pred = pipeline.predict(X_test)
    #
    # # 6. Evaluate
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred, target_names=data.target_names))
    #
    # # 7. Save the trained pipeline
    # joblib.dump(pipeline, "used_car_price_prediction_model.joblib")
    # print("✅ Pipeline saved to used_car_price_prediction_model.joblib")
    #
    # # 8. Example of loading and using the saved pipeline
    # loaded_model = joblib.load("used_car_price_prediction_model.joblib")
    # sample = X_test.iloc[:5]
    # print("Sample predictions:", loaded_model.predict(sample))





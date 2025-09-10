
import joblib
import matplotlib.pyplot as plt
import config
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



def plot_loss_XGBoost(evals_result):
    epochs = len(evals_result["validation_0"]["rmse"])
    x_axis = range(epochs)
    plt.plot(x_axis, evals_result["validation_0"]["rmse"], label="Train")
    plt.plot(x_axis, evals_result["validation_1"]["rmse"], label="Test")
    plt.xlabel("Boosting Round")
    plt.ylabel("RMSE")
    plt.title("XGBoost Training vs Validation Loss")
    plt.legend()
    return plt

def split_data(test_data):
    X_test = test_data[config.FEATURES]
    y_test = test_data[config.TARGET]
    return X_test, y_test


def model_score(train_data,test_data):
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    pre_process, model = load_pipeline(config.PIPELINE_PATH)
    X_train_transformed = pre_process.transform(X_train)
    X_test_transformed = pre_process.transform(X_test)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    score_test = model.score(X_test_transformed, y_test)
    score_train = model.score(X_train_transformed, y_train)
    MAPE_test = mean_absolute_percentage_error(y_test, model.predict(X_test_transformed))
    MAPE_train = mean_absolute_percentage_error(y_train, model.predict(X_train_transformed))
    return score_test, score_train, MAPE_test, MAPE_train

def predict_model(test_data):
    X_test, y_test = split_data(test_data)
    pre_process, model = load_pipeline(config.PIPELINE_PATH)
    X_test_transformed = pre_process.transform(X_test)
    y_test = np.log1p(y_test)
    y_pred = model.predict(X_test_transformed)

    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)  # flatten (important!)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)  # returns MSE (no squared arg)
    baseline_mse = mean_squared_error(y_test, np.full_like(y_test, np.mean(y_test)))
    rel_mse = mse / np.mean(y_test) * 100

    return mae,mse,baseline_mse,rel_mse


def predict_car_price(car_details):


    pre_process, model = load_pipeline(config.PIPELINE_PATH)

    X_transformed = pre_process.transform(car_details)

    # Predict with trained model
    prediction = model.predict(X_transformed)

    # Return scalar
    return np.round(np.expm1(prediction),2)


def save_pipeline(preprocessor, model, filename=config.PIPELINE_PATH, model_path=config.MODEL_PATH):
    # Save keras model separately
    joblib.dump(model, model_path)

    # Save preprocessor + model path
    pipeline_obj = {
        "preprocessor": preprocessor,
        "model_path": model_path
    }
    joblib.dump(pipeline_obj, filename)
    print(f"Saved pipeline to {filename} and keras model to {model_path}")

def load_pipeline(filename=config.PIPELINE_PATH):
    pipeline_obj = joblib.load(filename)
    preprocessor = pipeline_obj["preprocessor"]
    model = joblib.load(pipeline_obj["model_path"])
    return preprocessor, model


import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import config
from model_pipline import create_pipline


def plot_loss(history, from_epoch=0, to_epoch=-1):
  plt.plot(history.history['loss'][from_epoch:to_epoch], label='loss')
  plt.plot(history.history['val_loss'][from_epoch:to_epoch], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [price]')
  plt.legend()
  plt.grid(True)


def predict_car_price(car_details):


    pre_process, model = load_pipeline(config.PIPELINE_PATH)

    X_transformed = pre_process.transform(car_details)

    # Predict with trained model
    prediction = model.predict(X_transformed)

    # Return scalar
    return float(prediction[0][0])


def save_pipeline(preprocessor, model, filename=config.PIPELINE_PATH, model_path=config.MODEL_PATH):
    # Save keras model separately
    model.save(model_path)

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
    model = tf.keras.models.load_model(pipeline_obj["model_path"])
    return preprocessor, model

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import config
from model_pipline import train_model, arrangeData
from utils import predict_car_price, predict_model, plot_loss_XGBoost, model_score

st.title("🚗 Car Price Prediction App")

# -----------------------------
# Upload training dataset
# -----------------------------
st.subheader("Train Model")
train_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"], key="train")

if train_file:
    df = pd.read_csv(train_file)
    st.write("### Training Data Preview")
    st.dataframe(df.head())
    df = arrangeData(df)

    # 👉 Save to session state so other pages (EDA) can use it
    st.session_state["train_data"] = df

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    if st.button("🚀 Train Model"):
        evals_result = train_model(train)
        st.success(f"""✅ Model trained and saved to {config.MODEL_PATH}!""")

        st.write("### 📉 Training Loss Curve")
        fig = plot_loss_XGBoost(evals_result)
        st.pyplot(fig)

        score_test, score_train, MAPE_test, MAPE_train = model_score(train,test)
        st.write("### evaluate model")
        st.write(f"R2 score (test):{score_test}")
        st.write(f"R2 score (train):{score_train}")
        st.write(f"MAPE (test):{MAPE_test}")
        st.write(f"MAPE (train):{MAPE_train}")

        mae,mse,baseline_mse,rel_mse = predict_model(test)
        st.write("### predict model")
        st.write(f"RMSE: {mse:.2f}")
        st.write(f"MAE : {mae:.2f}")
        st.write(f"Baseline RMSE (predict mean): {baseline_mse:.2f}")
        st.write(f"Relative RMSE: {rel_mse:.1f}%")



# -----------------------------
# Manual input form
# -----------------------------
st.write("---")
st.subheader("✍️ Enter Car Details Manually")

default_values = {
    "vehicle_age": 10.0,
    "powerPS": 80.0,
    "kilometer": 25000.0,
    "seller": "privat",
    "offerType": "Angebot",
    "abtest": "control",
    "vehicleType": "kleinwagen",
    "gearbox": "manuell",
    "model": "polo",
    "fuelType": "benzin",
    "brand": "volkswagen",
    "dateCreated": "17/03/2016 0:00",
    "lastSeen": "30/03/2016 0:00"
}

with st.form("manual_entry"):
    vehicle_age = st.number_input("Vehicle Age", value=default_values["vehicle_age"])
    powerPS = st.number_input("Power (PS)", value=default_values["powerPS"])
    kilometer = st.number_input("Kilometer", value=default_values["kilometer"])

    seller = st.selectbox("Seller", ["privat", "gewerblich"], index=0)
    offerType = st.selectbox("Offer Type", ["Angebot", "Gesuch"], index=0)
    abtest = st.selectbox("AB Test", ["control", "test"], index=0)
    vehicleType = st.selectbox("Vehicle Type", ["kleinwagen", "limousine", "cabrio", "coupe", "suv"], index=0)
    gearbox = st.selectbox("Gearbox", ["manuell", "automatik"], index=0)
    model = st.text_input("Model", value=default_values["model"])
    fuelType = st.selectbox("Fuel Type", ["benzin", "diesel", "hybrid", "elektro"], index=0)
    brand = st.text_input("Brand", value=default_values["brand"])
    dateCreated = st.text_input("Date Created", value=default_values["dateCreated"])
    lastSeen = st.text_input("Last Seen", value=default_values["lastSeen"])

    submitted = st.form_submit_button("🔮 Predict Price")

if submitted:
    try:
        manual_df = pd.DataFrame([{
            "vehicle_age": vehicle_age,
            "powerPS": powerPS,
            "kilometer": kilometer,
            "seller": seller,
            "offerType": offerType,
            "abtest": abtest,
            "vehicleType": vehicleType,
            "gearbox": gearbox,
            "model": model,
            "fuelType": fuelType,
            "brand": brand.lower(),
            "dateCreated": dateCreated,
            "lastSeen": lastSeen
        }])

        prediction = float(predict_car_price(manual_df))
        st.success(f"💰 Estimated Car Price: **{prediction:,.2f} $**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

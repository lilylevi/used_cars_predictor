import streamlit as st
import pandas as pd
from datetime import datetime
import config
import utils
from model_pipline import arrangeData
from utils import load_pipeline, predict_car_price

# -----------------------------
# Load trained model, scaler, and dataset
# -----------------------------
pre_process, model = load_pipeline(config.PIPELINE_PATH)
eda_file = st.file_uploader("Upload dataset for EDA (CSV)", type=["csv"], key="eda")
if eda_file:
        import pandas as pd
        df = pd.read_csv(eda_file)
        df = arrangeData(df)
        st.info("Using separately uploaded dataset.")
else:
        st.warning("Please upload data in the main app or here to explore.")
        st.stop()

st.write("### Data Preview")
st.dataframe(df.head())

CURRENT_YEAR = datetime.now().year


# -----------------------------
# Helper Functions
# -----------------------------
def find_similar_cars(input_car, df_Comparison, n=5):
    # prefer same brand; if none, fallback to entire dataset
    filtered = df_Comparison[df_Comparison["brand"].astype(str) == str(input_car["brand"])].copy()
    if filtered.empty:
        filtered = df.copy()
    # remove rows missing needed features
    filtered = filtered.dropna(subset=["vehicle_age", "kilometer"]).copy()
    filtered["age_diff"] = (filtered["vehicle_age"] - input_car["vehicle_age"].iloc[0]).abs()
    filtered["kilometer_diff"] = (filtered["kilometer"] - input_car["kilometer"].iloc[0]).abs()
    filtered["score"] = filtered["age_diff"] + filtered["kilometer_diff"] / 10000
    similar = filtered.sort_values("score").head(n)
    similar.drop(columns=["age_diff","kilometer_diff","score"], inplace=True)
    return similar.reset_index(drop=True)

def predict_depreciation_series(initial_price, years=5, rate=0.1):
    # returns list for year 0..years (year 0 = current price)
    return [round(initial_price * ((1 - rate) ** i), 2) for i in range(0, years + 1)]


def predict_depreciation(predicted_price, years=5, rate=0.1):
    depreciation = []
    price = predicted_price
    for i in range(1, years + 1):
        price = price * (1 - rate)
        depreciation.append({"Year": i, "Predicted Price": round(price, 2)})
    return pd.DataFrame(depreciation)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Used Car Price Prediction App")

tab1, tab2 = st.tabs(["📈 Market Comparison & Depreciation", "🔮 Batch Prediction"])


# -----------------------------
# Tab 1: Market Comparison & Depreciation
# -----------------------------
with tab1:
    st.subheader("Compare with Similar Cars (and depreciation curves)")
    brand_com = st.selectbox("Brand (Comparison)", df["brand"].unique(), key="brand2_tab2")
    vehicle_age_com = st.slider("Vehicle Age (Comparison)", 0, int(df["vehicle_age"].max()), 5, key="vehicle_age2")
    mileage_com = st.number_input("Mileage (Comparison)", 0, 300000, 50000, key="mileage2")

    powerPS_com = st.slider("Power (Comparison)", 10, 2000, 100)
    seller_com = st.selectbox("Seller (Comparison)", ["privat", "gewerblich"], index=0)
    offerType_com = st.selectbox("Offer Type (Comparison)", ["Angebot", "Gesuch"], index=0)
    abtest_com = st.selectbox("AB Test (Comparison)", ["control", "test"], index=0)
    vehicleType_com = st.selectbox("Vehicle Type (Comparison)", ["kleinwagen", "limousine", "cabrio", "coupe", "suv"], index=0)
    gearbox_com = st.selectbox("Gearbox (Comparison)", ["manuell", "automatik"], index=0)
    car_model_com = st.selectbox("Model (Comparison)", df["model"].unique())
    fuelType_com = st.selectbox("Fuel Type (Comparison)", ["benzin", "diesel", "hybrid", "elektro"], index=0)
    dateCreated_com = st.text_input("Date Created (Comparison)", "1.1.2020")
    lastSeen_com = st.text_input("Last Seen (Comparison)", "1.1.2020")


    years_to_show = st.number_input("Depreciation horizon (years)", min_value=1, max_value=10, value=5,
                                    key="depr_years")
    annual_rate = st.number_input("Annual depreciation rate (fraction)", min_value=0.0, max_value=1.0, value=0.10,
                                  step=0.01, key="depr_rate")

    if st.button("Compare & Predict", key="compare_button_tab2"):
        # Input car prediction
        input_car = pd.DataFrame([{
                     "brand": brand_com,
                     "vehicle_age": vehicle_age_com,
                     "kilometer": mileage_com,
                     "powerPS": powerPS_com,
                     "seller": seller_com,
                     "offerType": offerType_com,
                     "abtest": abtest_com,
                     "gearbox": gearbox_com,
                     "model": car_model_com,
                     "vehicleType": vehicleType_com,
                     "fuelType": fuelType_com,
                     "dateCreated": dateCreated_com,
                     "lastSeen": lastSeen_com
        }])

        predicted_input = float(predict_car_price(input_car))
        st.success(f"Input car predicted price: ${predicted_input:.2f}")

        df_Comparison = df.drop(columns=["price"], errors="ignore")

        # Find similar cars (rows from dataset)
        similar_df = find_similar_cars(input_car, df_Comparison, n=5)
        if similar_df.empty:
            st.info("No similar cars found in the dataset.")
        else:
            # Predict price for each similar car using the model
            sim_features = similar_df[config.FEATURES]
            sim_preds = predict_car_price(sim_features)
            similar_df = similar_df.copy()
            similar_df["predicted_price"] = [float(x) for x in sim_preds]
            # difference vs input predicted price
            similar_df["diff_vs_input"] = (similar_df["predicted_price"] - predicted_input).round(2)
            similar_df["diff_pct_vs_input"] = (
                        (similar_df["predicted_price"] - predicted_input) / predicted_input * 100).round(2)

            # Show table of similar cars with predicted prices
            display_cols = []

            display_cols = ["brand", "vehicle_age", "kilometer", "predicted_price", "diff_vs_input","diff_pct_vs_input"]
            st.markdown("**Similar cars (predicted prices and difference vs input):**")
            st.table(similar_df[display_cols].reset_index(drop=True))

            # Build depreciation DataFrame for plotting multiple series together
            years = list(range(0, int(years_to_show) + 1))
            depr_data = {}
            # input car series
            depr_data["Input (predicted)"] = predict_depreciation_series(predicted_input, years=int(years_to_show),
                                                                         rate=float(annual_rate))
            # similar cars series, label them uniquely
            for i, row in similar_df.iterrows():
                label = f"Sim {i + 1} - {row.get('brand', '')} a{int(row['vehicle_age'])} m{int(row['kilometer'])}"
                depr_data[label] = predict_depreciation_series(row["predicted_price"], years=int(years_to_show),
                                                               rate=float(annual_rate))

            depr_df = pd.DataFrame(depr_data, index=years)
            depr_df.index.name = "Year"
            st.markdown("**Depreciation curves (Year 0 = current predicted price):**")
            st.line_chart(depr_df)

            # Expanders to show table per similar car if user wants
            for i, row in similar_df.iterrows():
                label = f"Sim {i + 1} details"
                with st.expander(label):
                    st.write(row[["brand", "vehicle_age", "kilometer", "predicted_price"]])
                    per_car_depr = pd.DataFrame({
                        "Year": years,
                        "Predicted Price": predict_depreciation_series(row["predicted_price"], years=int(years_to_show),
                                                                       rate=float(annual_rate))
                    }).set_index("Year")
                    st.table(per_car_depr)

# -----------------------------
# Tab 2: Batch Prediction
# -----------------------------
with tab2:
    st.subheader("Upload CSV for Batch Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        batch_df = arrangeData(batch_df)

        batch_df.drop(columns=["price"], errors="ignore")

        batch_df["predicted_price"] = predict_car_price(batch_df)

        st.write("Batch Predictions")
        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False)
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
import streamlit as st
import gdown
import pandas as pd

# Google Drive URL of your file (make sure the link is publicly accessible)
url = 'https://drive.google.com/file/d/1nRCdqz6fLyzKjBbAeC0Br8qgH4dxd91T/view?usp=drive_link'
output = 'city_hour.csv'

# Download the dataset
gdown.download(url, output, quiet=False)

# Now load the dataset
df = pd.read_csv(output)


from datetime import date
from aqi_model import load_data, train_forecast_model, forecast_for_date

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌫️ AQI Forecasting App")

# City dropdown
cities = [
    "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal", "Brajrajnagar",
    "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Ernakulam", "Gurugram",
    "Guwahati", "Hyderabad", "Jaipur", "Jorapokhar", "Kochi", "Kolkata",
    "Lucknow", "Mumbai", "Patna", "Shillong", "Talcher", "Thiruvananthapuram", "Visakhapatnam"
]

city = st.selectbox("Select a city", cities)

if st.button("Train Model and Forecast"):
    with st.spinner("Training model..."):
        df = load_data(city)
        results = train_forecast_model(df)

        st.success(f"✅ Model trained! Accuracy: {results['accuracy']:.2f}%")

        st.line_chart(results['forecast'].set_index('ds')[['final_pred']])
        st.session_state['forecast_df'] = results['forecast']
        st.session_state['city'] = city

if 'forecast_df' in st.session_state:
    st.subheader("📅 Predict AQI for a Future Date")
    selected_date = st.date_input("Pick a date", value=date.today())

    if st.button("Predict AQI"):
        forecast_df = st.session_state['forecast_df']
        prediction = forecast_for_date(forecast_df, selected_date.strftime("%Y-%m-%d"))
        if prediction:
            st.success(f"Predicted AQI in {st.session_state['city']} on {selected_date}: **{round(prediction, 2)}**")
        else:
            st.warning("Date not found in forecast range.")

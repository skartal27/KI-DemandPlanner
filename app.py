import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Seitenkonfiguration
st.set_page_config(page_title="KI Nachfrageplaner", layout="wide")

st.title("📦 KI Nachfrageplaner Prototyp")
st.markdown("Diese Anwendung demonstriert KI-gestützte Absatzprognosen.")

# =========================
# SIDEBAR (Einstellungen)
# =========================
st.sidebar.header("⚙️ Einstellungen")

forecast_days = st.sidebar.selectbox(
    "Prognosezeitraum (Tage)",
    [30, 60, 90]
)

avg_demand = st.sidebar.slider(
    "Durchschnittlicher Tagesbedarf",
    min_value=50,
    max_value=300,
    value=150
)

product = st.sidebar.selectbox(
    "Produkt auswählen",
    ["Produkt A", "Produkt B", "Produkt C"]
)

# =========================
# DATENERZEUGUNG
# =========================
st.subheader(f"📊 {product} – Vergangene Verkaufsdaten")

dates = pd.date_range(start='2025-01-01', end='2025-12-31')
sales = np.random.normal(loc=avg_demand, scale=20, size=len(dates)).astype(int)

df = pd.DataFrame({
    'date': dates,
    'sales': sales
})

st.dataframe(df.head())

# =========================
# PROPHET VORHERSAGE
# =========================
df_prophet = df.rename(columns={'date': 'ds', 'sales': 'y'})

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# =========================
# GRAFIK
# =========================
st.subheader(f"📈 {forecast_days}-Tage Nachfrageprognose")

fig = model.plot(forecast)
st.pyplot(fig)

# =========================
# TABELLE + DOWNLOAD
# =========================
st.subheader("📄 Prognoseergebnisse")

result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
st.dataframe(result.tail())

csv = result.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Prognoseergebnisse als CSV herunterladen",
    data=csv,
    file_name=f"{product}_prognose.csv",
    mime='text/csv'
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🔮 *KI Nachfrageplaner Prototyp – Entwickelt mit Streamlit & Prophet*")

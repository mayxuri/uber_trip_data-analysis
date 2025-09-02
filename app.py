# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(page_title="Uber Trip Analysis", layout="wide")
st.title("🚖 Uber Trip Analysis & Prediction")

# ---------------------------------
# Load Dataset (Hardcoded)
# ---------------------------------
DATA_FILE = "Uber-Jan-Feb-FOIL.csv"  # Make sure file is in same folder
data = pd.read_csv(DATA_FILE)

st.subheader("📌 Dataset Preview")
st.dataframe(data.head())

# ---------------------------------
# Preprocessing
# ---------------------------------
st.subheader("🧹 Data Preprocessing")

datetime_col = None
for col in data.columns:
    if "date" in col.lower() or "time" in col.lower():
        datetime_col = col
        break

if datetime_col is not None:
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
    data.dropna(subset=[datetime_col], inplace=True)

    # Extract features
    data["Day"] = data[datetime_col].dt.day
    data["DayOfWeek"] = data[datetime_col].dt.dayofweek
    data["Month"] = data[datetime_col].dt.month
    data["Hour"] = data[datetime_col].dt.hour

    st.success(f"✅ Extracted datetime features from `{datetime_col}`")
else:
    st.error("⚠️ No datetime column found in dataset!")
    st.stop()

# ---------------------------------
# EDA
# ---------------------------------
st.subheader("📊 Exploratory Data Analysis")

# 1. Daily Trips Trend
st.markdown("### 📈 Daily Trips Trend")
daily_trips = data.groupby(data[datetime_col].dt.date).size()
fig, ax = plt.subplots(figsize=(10,4))
daily_trips.plot(ax=ax, color="darkblue")
ax.set_ylabel("Trips")
ax.set_xlabel("Date")
st.pyplot(fig)

# 2. Trips by Day of Week vs Hour (Heatmap)
st.markdown("### 🔥 Heatmap of Trips by Hour & Day of Week")
heatmap_data = data.groupby(["DayOfWeek", "Hour"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week (0=Mon, 6=Sun)")
st.pyplot(fig)

# 3. Top Bases (if exists)
if "Base" in data.columns:
    st.markdown("### 🏢 Top Pickup Bases")
    top_bases = data["Base"].value_counts().head(10)
    fig, ax = plt.subplots()
    top_bases.plot(kind="bar", ax=ax, color="teal")
    ax.set_ylabel("Number of Trips")
    st.pyplot(fig)

# ---------------------------------
# ML Modeling
# ---------------------------------
st.subheader("🤖 Machine Learning Model")

# Aggregate trips per day
daily_trips_df = data.groupby(data[datetime_col].dt.date).size().reset_index(name="Trips")
daily_trips_df["DayNum"] = np.arange(len(daily_trips_df))  # feature = sequential day number

X = daily_trips_df[["DayNum"]]
y = daily_trips_df["Trips"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sidebar - choose model
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost", "Gradient Boosting"]
)

# Train selected model
if model_choice == "Random Forest":
    model = RandomForestRegressor(random_state=42)
elif model_choice == "XGBoost":
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
else:
    model = GradientBoostingRegressor(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.markdown(f"### 📊 Model Evaluation: {model_choice}")
st.write("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
st.write("R² Score:", round(r2_score(y_test, y_pred), 2))

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, color="purple")
ax.set_xlabel("Actual Trips")
ax.set_ylabel("Predicted Trips")
ax.set_title(f"Actual vs Predicted Trips ({model_choice})")
st.pyplot(fig)

# ---------------------------------
# Insights Section
# ---------------------------------
st.subheader("🔎 Insights & Conclusions")

# 1. Busiest / Least busy days
busiest_day = daily_trips_df.loc[daily_trips_df["Trips"].idxmax()]
least_busy_day = daily_trips_df.loc[daily_trips_df["Trips"].idxmin()]

st.markdown(f"- 📈 **Busiest day**: {busiest_day[datetime_col]} with {busiest_day['Trips']} trips")
st.markdown(f"- 🗓️ **Least busy day**: {least_busy_day[datetime_col]} with {least_busy_day['Trips']} trips")

# 2. Peak hour
peak_hour = data.groupby("Hour").size().idxmax()
peak_trips = data.groupby("Hour").size().max()
st.markdown(f"- ⏰ **Peak hour**: {peak_hour}:00 with {peak_trips} trips")

# 3. Top base
if "Base" in data.columns:
    top_base = data["Base"].value_counts().idxmax()
    top_base_trips = data["Base"].value_counts().max()
    st.markdown(f"- 🏢 **Top pickup base**: {top_base} with {top_base_trips} trips")

# 4. Model performance summary
st.markdown(f"- 🏆 **Selected model**: {model_choice} with R² score = {round(r2_score(y_test, y_pred), 2)}")

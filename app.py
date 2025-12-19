import streamlit as st
import pandas as pd
import os

# 1. SETTINGS - MUST BE AT THE VERY TOP
st.set_page_config(page_title="Smart Bin AI", layout="wide")

# 2. CLEAN IMPORTS (No duplicates)
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import osmnx as ox
import networkx as nx

# 3. GLOBAL DATA LOADING
@st.cache_data
def load_data():
    # Looks for your file in the root directory
    if os.path.exists('smart_bin_historical_data.csv'):
        df = pd.read_csv('smart_bin_historical_data.csv')
    elif os.path.exists('data.csv'):
        df = pd.read_csv('data.csv')
    else:
        # Fallback if file is missing
        return pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'fill': [0], 'lat': [19.07], 'lon': [72.87]})
    
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'bin_location_lat': 'lat', 'bin_location_lon': 'lon', 'bin_fill_percent': 'fill'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])

df = load_data()

# 4. SINGLE NAVIGATION CONTROL
page = st.sidebar.selectbox("Select Page", ["Home", "Analytics", "Route Optimization", "Finance"])

# 5. THE IF-ELIF CHAIN (Prevents repetition)
if page == "Home":
    st.title("🚛 Smart Waste Dashboard")
    st.write("Current Data Status:", "✅ Loaded" if not df.empty else "❌ Missing CSV")
    st.dataframe(df.head())

elif page == "Analytics":
    st.title("📊 Waste Analytics")
    if 'fill' in df.columns:
        fig = px.histogram(df, x="fill", title="Distribution of Bin Fill Levels")
        st.plotly_chart(fig)

elif page == "Route Optimization":
    st.title("📍 AI Route Optimization")
    # Mapping logic only runs here
    m = folium.Map(location=[19.076, 72.877], zoom_start=12)
    st_folium(m, width=1000, height=500)

elif page == "Finance":
    st.title("💎 Financial Impact")
    st.metric("Estimated Savings", "₹12,400", "+15%")

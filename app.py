import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import osmnx as ox
import networkx as nx
import qrcode
from io import BytesIO
import os

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Smart Bin Mission Control", layout="wide")

# --- 2. CONSTANTS & HELPERS ---
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

def get_dist(p1, p2):
    # FIXED: Corrected math syntax error
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    # Try to find any available CSV
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    target = 'data.csv' if 'data.csv' in all_files else (all_files[0] if all_files else None)
    
    if not target:
        # Dummy data if no file found
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00']),
            'bin_id': ['B101'], 'hour_of_day': [10], 'day_of_week': ['Monday'],
            'ward': ['Ward_A'], 'area_type': ['Residential'], 'fill': [50.0],
            'lat': [19.0760], 'lon': [72.8777], 'time_since_last_pickup': [12]
        })
        return df

    df = pd.read_csv(target)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Standardize column names
    rename_dict = {
        'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
        'bin_fill_percent': 'fill'
    }
    df = df.rename(columns=rename_dict)
    
    # FIXED: Coerce errors to prevent crash on bad dates
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])

@st.cache_resource
def get_map_graph():
    return ox.graph_from_point((19.0760, 72.8777), dist=8000, network_type='drive')

df = load_data()

# --- 4. NAVIGATION ---
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- PAGE LOGIC ---
if page == "Home":
    st.title("🚛 Smart Waste Management System")
    st.write("Welcome to the AI-driven waste optimization dashboard.")
    st.dataframe(df.head())

elif page == "Exploratory Data Analysis":
    st.title("📊 Data Insights")
    if 'hour_of_day' in df.columns:
        fig = px.line(df.groupby('hour_of_day')['fill'].mean().reset_index(), x='hour_of_day', y='fill', title="Avg Fill by Hour")
        st.plotly_chart(fig)

elif page == "Predictive Model":
    st.title("🤖 Fill Level Prediction")
    st.info("Training Random Forest Regressor...")
    # Add your prediction logic here (as per your previous snippet)

elif page == "Route Optimization":
    st.title("📍 AI Mission Control")
    # Add your Folium mapping logic here
    st.warning("Ensure 'osmnx' is in your requirements.txt for this to load.")

elif page == "Impact & Financial Analysis":
    st.title("💎 Business Case & ROI")
    # Add your financial calculation logic here

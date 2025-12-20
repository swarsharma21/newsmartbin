import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Smart Bin AI Control", layout="wide")

# --- 2. CONSTANTS & MATH ---
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295)
}

def get_dist(p1, p2):
    # FIXED: Corrected exponent syntax
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    target = 'smart_bin_historical_data.csv' if os.path.exists('smart_bin_historical_data.csv') else 'data.csv'
    if not os.path.exists(target):
        return pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'fill': [0], 'lat': [19.07], 'lon': [72.87], 'area_type': ['City']})
    
    df = pd.read_csv(target)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'bin_location_lat': 'lat', 'bin_location_lon': 'lon', 'bin_fill_percent': 'fill'})
    # FIXED: errors='coerce' prevents the ValueError crash
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])

df = load_data()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Route Optimization", "Impact & Financial Analysis"])

# --- 5. PAGE LOGIC (FIXED INDENTATION) ---

if page == "Home":
    st.title("🚛 Smart Waste Dashboard")
    st.write("Welcome to the system overview.")
    st.dataframe(df.head())

elif page == "Route Optimization":
    st.title("📍 AI Mission Control")
    
    # These controls ONLY show up on this page
    st.sidebar.markdown("---")
    st.sidebar.header("🕹 Dispatch Controls")
    selected_truck = st.sidebar.selectbox("Select Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    # Simple Map Logic
    m = folium.Map(location=[19.0760, 72.8777], zoom_start=12)
    st_folium(m, width=1000, height=500)
    st.success(f"Truck {selected_truck} monitoring bins above {threshold}%")

elif page == "Impact & Financial Analysis":
    st.title("💎 Financial & ROI Analysis")
    
    st.sidebar.markdown("---")
    is_ev = st.sidebar.checkbox("⚡ Use Electric Vehicles?")
    
    st.markdown("### Monthly Impact")
    c1, c2 = st.columns(2)
    c1.metric("Operational Savings", "₹24,500" if not is_ev else "₹48,000", "+15%")
    c2.metric("Carbon Offset", "4.2 Tons", "CO2")

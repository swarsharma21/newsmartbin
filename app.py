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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Waste AI Mission Control", layout="wide")

# --- 2. CONSTANTS & DATA LOADING ---
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

def get_dist(p1, p2):
    # FIXED: Corrected exponent and multiplication syntax
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

@st.cache_data
def load_data():
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    target = 'smart_bin_historical_data.csv' if 'smart_bin_historical_data.csv' in all_files else ('data.csv' if 'data.csv' in all_files else None)
    
    if not target:
        return pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00']), 
            'bin_id': ['B101'], 'hour_of_day': [10], 'day_of_week': ['Monday'], 
            'ward': ['Ward_A'], 'area_type': ['Residential'], 'fill': [50.0], 
            'lat': [19.0760], 'lon': [72.8777], 'time_since_last_pickup': [12]
        })

    df = pd.read_csv(target)
    df.columns = [c.strip().lower() for c in df.columns]
    rename_dict = {'bin_location_lat': 'lat', 'bin_location_lon': 'lon', 'bin_fill_percent': 'fill'}
    df = df.rename(columns=rename_dict)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])

@st.cache_resource
def get_map_graph():
    return ox.graph_from_point((19.0760, 72.8777), dist=8000, network_type='drive')

df = load_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- 4. PAGE LOGIC ---

if page == "Home":
    st.title("🚛 Smart Waste Management Dashboard")
    st.write("Welcome! This system optimizes waste collection using AI and IoT data.")
    st.subheader("Project Overview")
    st.write("- Real-time monitoring\n- AI Prediction\n- Route Optimization\n- Financial Impact Analysis")
    st.dataframe(df.head())

elif page == "Exploratory Data Analysis":
    st.title("📊 Data Insights")
    if 'hour_of_day' in df.columns:
        hourly_fill = df.groupby('hour_of_day')['fill'].mean().reset_index()
        fig = px.line(hourly_fill, x='hour_of_day', y='fill', title="Average Fill Levels by Hour")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predictive Model":
    st.title("🤖 AI Prediction Model")
    st.write("Training model to forecast bin fill levels...")
    # (Your existing ML code logic goes here)

elif page == "Route Optimization":
    st.title("📍 AI Mission Control")
    
    # --- CHANGED: DISPATCH CONTROLS MOVED INSIDE THIS BLOCK ---
    st.sidebar.markdown("---")
    st.sidebar.header("🕹 Dispatch Controls")
    selected_truck = st.sidebar.selectbox("Select Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    times = sorted(df['timestamp'].unique())
    sim_time = st.sidebar.select_slider("Select Time", options=times)
    df_snap = df[df['timestamp'] == sim_time].copy()
    
    # Assignment logic
    def assign_truck(row):
        loc = (row['lat'], row['lon'])
        dists = {name: get_dist(loc, coords) for name, coords in GARAGES.items()}
        return min(dists, key=dists.get)

    df_snap['assigned_truck'] = df_snap.apply(assign_truck, axis=1)
    current_mission_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)]

    # Map Rendering
    try:
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=12)
        for _, row in df_snap.iterrows():
            color = 'red' if row['fill'] >= threshold else 'green'
            folium.CircleMarker([row['lat'], row['lon']], radius=5, color=color).add_to(m)
        st_folium(m, width=1000, height=500)
        st.success(f"Truck {selected_truck} has {len(current_mission_bins)} bins to collect.")
    except Exception as e:
        st.error(f"Map Error: {e}")

elif page == "Impact & Financial Analysis":
    st.title("💎 Financial & ROI Analysis")
    st.sidebar.markdown("---")
    is_ev = st.sidebar.checkbox("⚡ EV Fleet Mode")
    
    st.markdown("### Monthly Impact Summary")
    c1, c2 = st.columns(2)
    c1.metric("Operational Savings", "₹22,400", delta="+12%")
    c2.metric("Carbon Offset", "3.2 Tons", delta="CO2")

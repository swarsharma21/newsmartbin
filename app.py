import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- Page Configuration ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found. Using dummy data.")
        # Create a dummy DataFrame to prevent script crash
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00']), 
            'bin_id': ['B101'], 
            'hour_of_day': [10], 
            'day_of_week': ['Monday'], 
            'ward': ['Ward_A'], 
            'area_type': ['Residential'], 
            'time_since_last_pickup': [24], 
            'bin_fill_percent': [50], 
            'bin_capacity_liters': [1000],
            'bin_location_lat': [19.0760],
            'bin_location_lon': [72.8777]
        })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# FIX: Added "Impact & Financial Analysis" to the list below
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- Home Page ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.write("Welcome! This dashboard provides a comprehensive overview of the Smart Bin project's data science components.")
    st.subheader("Project Overview")
    st.write("""
    - *Live Data:* A physical prototype sends real-time fill-level data to a cloud dashboard.
    - *Historical Analysis:* We analyze a large dataset to understand waste generation patterns.
    - *Predictive Modeling:* A machine learning model forecasts when bins will become full.
    - *Route Optimization:* An algorithm calculates the most efficient collection route for full bins.
    - *Financial Impact:* A comprehensive model calculating ROI, carbon credits, and operational savings.
    """)
    st.subheader("Dataset at a Glance")
    st.dataframe(df.head())
    st.write(f"The dataset contains *{len(df)}* hourly readings from *{df['bin_id'].nunique()}* simulated smart bins.")


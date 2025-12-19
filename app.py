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
            'bin_id': ['B101'], 'hour_of_day': [10], 'day_of_week': ['

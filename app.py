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

# --- 1. Page Configuration ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# Constants for New Route Logic
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

# REPAIRED MATH FUNCTION (Fixed syntax for squaring)
def get_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

# --- 2. Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
        df.columns = [c.strip().lower() for c in df.columns]
        rename_dict = {
            'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
            'bin_fill_percent': 'fill', 'bin_id': 'bin_id'
        }
        for old, new in rename_dict.items():
            if old in df.columns: df = df.rename(columns={old: new})
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df.dropna(subset=['timestamp'])
    except:
        return None

@st.cache_resource
def get_map_graph():
    return ox.graph_from_point((19.0760, 72.8777), dist=8000, network_type='drive')

df = load_data()

# --- 3. Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- 4. Page Logic ---

if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.write("Welcome! This dashboard provides a comprehensive overview of the Smart Bin project.")
    if df is not None: st.dataframe(df.head())

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    if df is not None:
        hourly_fill = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
        fig1 = px.line(hourly_fill, x='hour_of_day', y='fill', color='area_type', title='Avg Fill % by Hour')
        st.plotly_chart(fig1, use_container_width=True)

elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    st.info("Random Forest Model Analysis")

elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Mission Control")
    # NEW ROUTE CODE START
    if df is not None:
        st.sidebar.header("🕹 Dispatch Controls")
        selected_truck = st.sidebar.selectbox("Select Active Truck", list(GARAGES.keys()))
        threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
        times = sorted(df['timestamp'].unique())
        sim_time = st.sidebar.select_slider("Select Time", options=times)
        df_snap = df[df['timestamp'] == sim_time].copy()

        def assign_truck(row):
            loc = (row['lat'], row['lon'])
            dists = {name: get_dist(loc, coords) for name, coords in GARAGES.items()}
            return min(dists, key=dists.get)

        df_snap['assigned_truck'] = df_snap.apply(assign_truck, axis=1)
        all_my_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)]
        
        st.sidebar.subheader("📦 Trip Pipeline")
        bins_per_trip = 8
        num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
        
        if num_trips > 0:
            trip_num = st.sidebar.selectbox(f"Select Trip", range(1, num_trips + 1), format_func=lambda x: f"Trip {x}")
            current_mission_bins = all_my_bins.iloc[(trip_num-1)*bins_per_trip : trip_num*bins_per_trip]
            
            G = get_map_graph()
            m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
            for _, row in df_snap.iterrows():
                color = 'red' if row['fill'] >= threshold and row['assigned_truck'] == selected_truck else 'green'
                folium.Marker([row['lat'], row['lon']], icon=folium.Icon(color=color, icon='trash', prefix='fa')).add_to(m)
            
            st_folium(m, width=1200, height=500, key="route_map")
            
            # QR Code Generation
            google_url = f"https://www.google.com/maps/dir/{GARAGES[selected_truck][0]},{GARAGES[selected_truck][1]}/" + "/".join([f"{r.lat},{r.lon}" for _, r in current_mission_bins.iterrows()]) + f"/{DEONAR_DUMPING[0]},{DEONAR_DUMPING[1]}"
            qr = qrcode.make(google_url)
            buf = BytesIO(); qr.save(buf)
            st.image(buf, width=200)
    # NEW ROUTE CODE END

elif page == "Impact & Financial Analysis":
    st.title("💎 Comprehensive Business & Impact Model")
    st.markdown("### The 360° Value Proposition")
    st.write("This advanced model evaluates the project's viability across four dimensions: *Operational Savings, **Revenue Generation, **Strategic Cost Avoidance, and **Environmental Monetization*.")

    # --- YOUR ORIGINAL SIDEBAR CONTROLS ---
    st.sidebar.header("⚙ Simulation Parameters")
    is_ev = st.sidebar.checkbox("⚡ Activate Electric Vehicle (EV) Fleet Mode")
    
    st.sidebar.subheader("1. CAPEX (Initial Investment)")
    num_trucks = st.sidebar.number_input("Fleet Size (Trucks)", value=5, min_value=1)
    hardware_cost_per_bin = st.sidebar.number_input("Hardware Cost/Bin (₹)", value=1500)
    total_bins = st.sidebar.number_input("Total Smart Bins", value=100)
    software_dev_cost = st.sidebar.number_input("Software/Cloud Setup Cost (₹)", value=50000)

    st.sidebar.subheader("2. OPEX (Operational)")
    driver_wage = st.sidebar.slider("Staff Hourly Wage (₹)", 100, 500, 200)
    
    if is_ev:
        fuel_price = st.sidebar.number_input("Electricity Cost (₹/kWh)", value=10.0)
        truck_efficiency = st.sidebar.number_input("EV Efficiency (km/kWh)", value=1.5)
        co2_factor, fuel_label = 0.82, "Electricity"
    else:
        fuel_price = st.sidebar.number_input("Diesel Price (₹/Liter)", value=104.0)
        truck_efficiency = st.sidebar.number_input("Truck Mileage (km/L)", value=4.0)
        co2_factor, fuel_label = 2.68, "Fuel"

    maintenance_per_km = st.sidebar.number_input("Vehicle Maint. (₹/km)", value=5.0)
    cloud_cost_per_bin = st.sidebar.number_input("Cloud/Data Cost per Bin/Month (₹)", value=20)

    st.sidebar.subheader("3. Revenue & Strategic Value")
    recyclable_value_per_kg = st.sidebar.number_input("Avg. Recyclable Value (₹/kg)", value=15.0)
    recycling_rate_increase = st.sidebar.slider("Recycling Efficiency Boost (%)", 0, 50, 20)
    daily_waste_collected_kg = st.sidebar.number_input("Total Daily Waste (kg)", value=2000.0)
    penalty_per_overflow = st.sidebar.number_input("Fine per Overflowing Bin (₹)", value=500)
    overflows_prevented_month = st.sidebar.slider("Overflows Prevented/Month", 0, 100, 25)
    carbon_credit_price = st.sidebar.number_input("Carbon Credit Price (₹/Ton CO2)", value=1500.0)

    st.sidebar.subheader("4. Logistics Efficiency")
    dist_old = st.sidebar.number_input("Daily Dist. Fixed (km)", value=60.0)
    trips_old = st.sidebar.slider("Trips/Month (Fixed)", 15, 30, 30)
    hours_old = st.sidebar.number_input("Hours/Trip (Fixed)", value=7.0)
    dist_new = st.sidebar.number_input("Daily Dist. Smart (km)", value=40.0)
    trips_new = st.sidebar.slider("Trips/Month (Smart)", 15, 30, 24)
    hours_new = st.sidebar.number_input("Hours/Trip (Smart)", value=5.0)

    # --- YOUR ORIGINAL CALCULATIONS ---
    total_capex = (hardware_cost_per_bin * total_bins) + software_dev_cost

    def calc_opex(dist, trips, hours):
        total_dist = dist * trips * num_trucks
        total_hours = hours * trips * num_trucks
        energy_consumed = total_dist / truck_efficiency
        energy_cost = energy_consumed * fuel_price
        labor_cost = total_hours * driver_wage
        maint_cost = total_dist * maintenance_per_km
        co2_kg = energy_consumed * co2_factor
        return {"energy": energy_cost, "labor": labor_cost, "maint": maint_cost, "total_opex": energy_cost + labor_cost + maint_cost, "co2": co2_kg}

    old, new = calc_opex(dist_old, trips_old, hours_old), calc_opex(dist_new, trips_new, hours_new)
    new["total_opex"] += (cloud_cost_per_bin * total_bins)

    opex_savings = old["total_opex"] - new["total_opex"]
    base_revenue = daily_waste_collected_kg * 30 * recyclable_value_per_kg * 0.1
    improved_revenue = daily_waste_collected_kg * 30 * recyclable_value_per_kg * (0.1 + (recycling_rate_increase/100))
    revenue_gain = improved_revenue - base_revenue
    penalty_savings = overflows_prevented_month * penalty_per_overflow
    co2_saved_tons = (old["co2"] - new["co2"]) / 1000
    carbon_credit_revenue = co2_saved_tons * carbon_credit_price
    total_monthly_benefit = opex_savings + revenue_gain + penalty_savings + carbon_credit_revenue
    months_breakeven = total_capex / total_monthly_benefit if total_monthly_benefit > 0 else 0

    # --- YOUR ORIGINAL DASHBOARD UI ---
    st.markdown("### 📊 Monthly Financial Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Monthly Benefit", f"₹{total_monthly_benefit:,.0f}")
    k2.metric("Direct OPEX Savings", f"₹{opex_savings:,.0f}")
    k3.metric("Revenue & Avoidance", f"₹{revenue_gain + penalty_savings:,.0f}")
    k4.metric("ROI Break-even", f"{months_breakeven:.1f} Months")

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        waterfall_data = pd.DataFrame({"Source": ["Operational Savings", "Recycling Revenue", "Avoided Penalties", "Carbon Credits"], "Amount (₹)": [opex_savings, revenue_gain, penalty_savings, carbon_credit_revenue]})
        fig_water = px.bar(waterfall_data, x="Source", y="Amount (₹)", color="Source", title="Monthly Value Components")
        st.plotly_chart(fig_water, use_container_width=True)
    with c2:
        st.subheader("Environmental Impact")
        st.metric("CO2 Prevented", f"{co2_saved_tons*1000:,.0f} kg")
        st.metric("Carbon Credit Value", f"₹{carbon_credit_revenue:,.2f}")

    st.subheader("📈 3-Year Financial Projection")
    months_range = range(1, 37)
    cash_flow = [-total_capex + (total_monthly_benefit * m) for m in months_range]
    df_cf = pd.DataFrame({"Month": list(months_range), "Net Cash Flow": cash_flow})
    fig_cf = px.line(df_cf, x="Month", y="Net Cash Flow", title="Cumulative Cash Flow")
    fig_cf.add_hline(y=0, line_dash="dash", line_color="green")
    st.plotly_chart(fig_cf, use_container_width=True)

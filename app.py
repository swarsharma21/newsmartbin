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
        }) # <--- CHECK IF THIS IS MISSING
            
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# FIX: Added "Impact & Financial Analysis" to the list below
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])
# --- 4. Page Logic ---

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


# --- EDA Page ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("These charts are now interactive. You can zoom, pan, and hover over the data.")
    
    st.subheader("Average Bin Fill Percentage by Hour of Day")
    hourly_fill_pattern = df.groupby(['hour_of_day', 'area_type'])['bin_fill_percent'].mean().reset_index()
    fig1 = px.line(hourly_fill_pattern, 
                   x='hour_of_day', 
                   y='bin_fill_percent', 
                   color='area_type',
                   title='Average Bin Fill Percentage by Hour of Day',
                   labels={'hour_of_day': 'Hour of Day', 'bin_fill_percent': 'Average Fill Level (%)'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Average Bin Fill Percentage by Day of the Week")
    daily_avg = df.groupby('day_of_week')['bin_fill_percent'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig2 = px.bar(daily_avg, 
                  x='day_of_week', 
                  y='bin_fill_percent',
                  category_orders={"day_of_week": day_order},
                  title='Average Bin Fill Percentage by Day of the Week',
                  labels={'day_of_week': 'Day of the Week', 'bin_fill_percent': 'Average Fill Level (%)'})
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    
    with st.spinner("Preparing data and training model..."):
        features_to_use = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
        target_variable = 'bin_fill_percent'
        model_df = df[features_to_use + [target_variable]].copy()
        
        for feature in ['day_of_week', 'ward', 'area_type']:
            if feature in model_df.columns:
                 model_df = pd.get_dummies(model_df, columns=[feature], prefix=feature, drop_first=True)
            
        X = model_df.drop(target_variable, axis=1, errors='ignore')
        y = model_df[target_variable]
        
        if len(X) < 2:
            st.warning("Insufficient data to train the model after preprocessing.")
            mae, r2 = np.nan, np.nan
            predictions = pd.Series([0])
            y_test = pd.Series([0])
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
    col2.metric("R-squared (R²) Score", f"{r2:.2f}")

    st.subheader("Interactive Analysis of Model Predictions")
    
    if len(y_test) > 10:
        plot_data = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        plot_data['Error'] = abs(plot_data['Actual'] - plot_data['Predicted'])
        plot_data_sample = plot_data.sample(min(5000, len(plot_data)), random_state=42)
        
        fig = px.scatter(
            plot_data_sample, 
            x='Actual', 
            y='Predicted',
            color='Error',
            color_continuous_scale=px.colors.sequential.Viridis,
            marginal_x='histogram',
            marginal_y='histogram',
            hover_name=plot_data_sample.index,
            hover_data={'Actual': ':.2f', 'Predicted': ':.2f', 'Error': ':.2f'},
            title="Actual vs. Predicted Fill Levels (Colored by Prediction Error)",
            template='plotly_white'
        )

        fig.add_shape(type='line', x0=0, y0=0, x1=100, y1=100, line=dict(color='Red', width=2, dash='dash'))
        fig.update_layout(xaxis_title='Actual Fill Level (%)', yaxis_title='Predicted Fill Level (%)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough test data points to generate a meaningful scatter plot.")
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

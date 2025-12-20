import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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

# --- Predictive Model Page ---
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



# --- Impact & Financial Analysis Page (New Section) ---
elif page == "Impact & Financial Analysis":
    st.title("💎 Comprehensive Business & Impact Model")
    st.markdown("### The 360° Value Proposition")
    st.write("This advanced model evaluates the project's viability across four dimensions: *Operational Savings, **Revenue Generation, **Strategic Cost Avoidance, and **Environmental Monetization*.")

    # --- 1. THE CONTROL CENTER (Sidebar) ---
    st.sidebar.header("⚙ Simulation Parameters")

    # [TOGGLE] EV FLEET SWITCH
    is_ev = st.sidebar.checkbox("⚡ Activate Electric Vehicle (EV) Fleet Mode")

    # A. CAPEX (One-time Investment)
    st.sidebar.subheader("1. CAPEX (Initial Investment)")
    num_trucks = st.sidebar.number_input("Fleet Size (Trucks)", value=5, min_value=1)
    hardware_cost_per_bin = st.sidebar.number_input("Hardware Cost/Bin (₹)", value=1500)
    total_bins = st.sidebar.number_input("Total Smart Bins", value=100)
    software_dev_cost = st.sidebar.number_input("Software/Cloud Setup Cost (₹)", value=50000)

    # B. OPEX (Recurring Costs)
    st.sidebar.subheader("2. OPEX (Operational)")
    driver_wage = st.sidebar.slider("Staff Hourly Wage (₹)", 100, 500, 200)
    
    # Dynamic Fuel/Energy Inputs
    if is_ev:
        st.sidebar.markdown("--- *⚡ EV Settings* ---")
        fuel_price = st.sidebar.number_input("Electricity Cost (₹/kWh)", value=10.0)
        truck_efficiency = st.sidebar.number_input("EV Efficiency (km/kWh)", value=1.5)
        co2_factor = 0.82 # kg CO2 per kWh (Grid Average)
        fuel_label = "Electricity"
        fuel_unit = "kWh"
    else:
        st.sidebar.markdown("--- *⛽ Diesel Settings* ---")
        fuel_price = st.sidebar.number_input("Diesel Price (₹/Liter)", value=104.0)
        truck_efficiency = st.sidebar.number_input("Truck Mileage (km/L)", value=4.0)
        co2_factor = 2.68 # kg CO2 per Liter
        fuel_label = "Fuel"
        fuel_unit = "L"

    # Maintenance & Connectivity
    maintenance_per_km = st.sidebar.number_input("Vehicle Maint. (₹/km)", value=5.0)
    cloud_cost_per_bin = st.sidebar.number_input("Cloud/Data Cost per Bin/Month (₹)", value=20) # SIM card/Cloud sub
    
    # C. REVENUE & AVOIDANCE (The "Hidden" Value)
    st.sidebar.subheader("3. Revenue & Strategic Value")
    
    # Recycling Revenue
    recyclable_value_per_kg = st.sidebar.number_input("Avg. Recyclable Value (₹/kg)", value=15.0)
    recycling_rate_increase = st.sidebar.slider("Recycling Efficiency Boost (%)", 0, 50, 20) 
    # (Assumption: Smart routing allows better segregation or cleaner pickup)
    daily_waste_collected_kg = st.sidebar.number_input("Total Daily Waste (kg)", value=2000.0)
    
    # Penalty Avoidance (SLA)
    penalty_per_overflow = st.sidebar.number_input("Fine per Overflowing Bin (₹)", value=500)
    overflows_prevented_month = st.sidebar.slider("Overflows Prevented/Month", 0, 100, 25)
    
    # Carbon Credits
    carbon_credit_price = st.sidebar.number_input("Carbon Credit Price (₹/Ton CO2)", value=1500.0)

    # D. ROUTE SCENARIOS
    st.sidebar.subheader("4. Logistics Efficiency")
    st.sidebar.markdown("🔴 Traditional (Fixed)")
    dist_old = st.sidebar.number_input("Daily Dist. Fixed (km)", value=60.0)
    trips_old = st.sidebar.slider("Trips/Month (Fixed)", 15, 30, 30)
    hours_old = st.sidebar.number_input("Hours/Trip (Fixed)", value=7.0)
    
    st.sidebar.markdown("🟢 Smart (Optimized)")
    dist_new = st.sidebar.number_input("Daily Dist. Smart (km)", value=40.0)
    trips_new = st.sidebar.slider("Trips/Month (Smart)", 15, 30, 24)
    hours_new = st.sidebar.number_input("Hours/Trip (Smart)", value=5.0)

    # --- 2. THE CALCULATION ENGINE ---
    
    # CAPEX
    total_capex = (hardware_cost_per_bin * total_bins) + software_dev_cost
    
    # Function to calculate Core OPEX
    def calc_opex(dist, trips, hours):
        total_dist = dist * trips * num_trucks
        total_hours = hours * trips * num_trucks
        
        energy_consumed = total_dist / truck_efficiency
        energy_cost = energy_consumed * fuel_price
        
        labor_cost = total_hours * driver_wage
        maint_cost = total_dist * maintenance_per_km
        
        # Cloud costs (only applies to Smart System really, but added for completeness)
        
        co2_kg = energy_consumed * co2_factor
        
        return {
            "energy": energy_cost,
            "labor": labor_cost,
            "maint": maint_cost,
            "total_opex": energy_cost + labor_cost + maint_cost,
            "co2": co2_kg,
            "dist": total_dist
        }

    old = calc_opex(dist_old, trips_old, hours_old)
    new = calc_opex(dist_new, trips_new, hours_new)
    
    # Add Cloud Cost to Smart System OPEX
    smart_cloud_cost = cloud_cost_per_bin * total_bins
    new["total_opex"] += smart_cloud_cost

    # --- SAVINGS & REVENUE CALCULATIONS ---
    
    # 1. Direct Operational Savings
    opex_savings = old["total_opex"] - new["total_opex"]
    
    # 2. Revenue from Recyclables (Incremental)
    # Assumption: Smart system improves segregation/collection efficiency by X%
    base_revenue = daily_waste_collected_kg * 30 * recyclable_value_per_kg * 0.1 # Assuming 10% base recycling rate
    improved_revenue = daily_waste_collected_kg * 30 * recyclable_value_per_kg * (0.1 + (recycling_rate_increase/100))
    revenue_gain = improved_revenue - base_revenue
    
    # 3. Cost Avoidance (Penalties)
    penalty_savings = overflows_prevented_month * penalty_per_overflow
    
    # 4. Environmental Monetization
    co2_saved_tons = (old["co2"] - new["co2"]) / 1000
    carbon_credit_revenue = co2_saved_tons * carbon_credit_price
    
    # TOTAL MONTHLY BENEFIT
    total_monthly_benefit = opex_savings + revenue_gain + penalty_savings + carbon_credit_revenue
    
    # ROI
    months_breakeven = total_capex / total_monthly_benefit if total_monthly_benefit > 0 else 0

    # --- 3. VISUALIZATION DASHBOARD ---

    # KPI ROW
    st.markdown("### 📊 Monthly Financial Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Monthly Benefit", f"₹{total_monthly_benefit:,.0f}", help="Savings + Revenue + Avoided Fines")
    k2.metric("Direct OPEX Savings", f"₹{opex_savings:,.0f}", delta="Fuel & Labor")
    k3.metric("Revenue & Avoidance", f"₹{revenue_gain + penalty_savings:,.0f}", delta="New Value Stream")
    k4.metric("ROI Break-even", f"{months_breakeven:.1f} Months", delta_color="off")

    st.markdown("---")

    # DETAILED BREAKDOWN
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Waterfall: Where is the Value coming from?")
        waterfall_data = pd.DataFrame({
            "Source": ["Operational Savings", "Recycling Revenue", "Avoided Penalties", "Carbon Credits"],
            "Amount (₹)": [opex_savings, revenue_gain, penalty_savings, carbon_credit_revenue]
        })
        fig_water = px.bar(waterfall_data, x="Source", y="Amount (₹)", color="Source", 
                           title="Monthly Value Components", text_auto='.2s')
        st.plotly_chart(fig_water, use_container_width=True)
        
    with c2:
        st.subheader("Environmental Impact")
        st.metric("CO2 Prevented", f"{co2_saved_tons*1000:,.0f} kg")
        st.metric("Carbon Credit Value", f"₹{carbon_credit_revenue:,.2f}")
        st.info(f"Equivalent to planting *{int(co2_saved_tons * 1000 / 20)} trees* per month.")

    # CUMULATIVE CASH FLOW (The Investor View)
    st.subheader("📈 3-Year Financial Projection")
    
    years = 3
    months_range = range(1, (years * 12) + 1)
    cash_flow = []
    current_balance = -total_capex # Start in debt (CAPEX)
    
    for m in months_range:
        current_balance += total_monthly_benefit
        cash_flow.append(current_balance)
        
    df_cf = pd.DataFrame({"Month": list(months_range), "Net Cash Flow": cash_flow})
    
    fig_cf = px.line(df_cf, x="Month", y="Net Cash Flow", title="Cumulative Cash Flow (NPV Proxy)", markers=False)
    fig_cf.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Break-even")
    fig_cf.add_vrect(x0=0, x1=months_breakeven, fillcolor="red", opacity=0.1, annotation_text="Investment Phase")
    fig_cf.add_vrect(x0=months_breakeven, x1=36, fillcolor="green", opacity=0.1, annotation_text="Profit Phase")
    
    st.plotly_chart(fig_cf, use_container_width=True)
    
    st.success(f"""
    *Final Verdict:* This project is not just a cost-saver; it is a revenue generator. 
    By integrating *Recycling Revenue* (₹{revenue_gain:,.0f}/mo) and *Penalty Avoidance* (₹{penalty_savings:,.0f}/mo) with standard operational savings, 
    the system pays for its hardware in *{months_breakeven:.1f} months*, creating a sustainable profit model for the municipality.
    """)

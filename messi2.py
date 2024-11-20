import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DRINK_TYPES = {
    "Custom": {"volume": 355, "abv": 5.0, "description": "Custom drink settings"},
    "Beer (Regular)": {"volume": 355, "abv": 5.0, "description": "Standard 12oz/355ml beer can/bottle"},
    "Beer (Craft)": {"volume": 355, "abv": 7.0, "description": "Craft beer (higher ABV)"},
    "Beer (Light)": {"volume": 355, "abv": 4.2, "description": "Light beer"},
    "Wine (Red)": {"volume": 175, "abv": 13.5, "description": "Standard red wine glass"},
    "Wine (White)": {"volume": 150, "abv": 12.5, "description": "Standard white wine glass"},
    "Wine (Fortified)": {"volume": 60, "abv": 20.0, "description": "Port, Sherry, etc."},
    "Spirit (Shot)": {"volume": 44, "abv": 40.0, "description": "Standard spirit shot"},
    "Spirit (Double)": {"volume": 89, "abv": 40.0, "description": "Double shot"},
    "Cocktail": {"volume": 200, "abv": 12.0, "description": "Average mixed drink"},
    "Champagne": {"volume": 150, "abv": 12.0, "description": "Standard champagne flute"},
    "Hard Seltzer": {"volume": 355, "abv": 5.0, "description": "Standard hard seltzer can"}
}

def calculate_initial_concentration(num_glasses=3, glass_volume_ml=150, abv=12, 
                                 body_weight_kg=70, gender='male'):
    # Calculate total alcohol volume and mass
    total_drink_volume_ml = num_glasses * glass_volume_ml
    alcohol_volume_ml = total_drink_volume_ml * (abv/100)
    alcohol_mass_g = alcohol_volume_ml * 0.789 # 0.789 g/mL
    
    # Calculate volume of distribution
    v_factor = 0.68 if gender == 'male' else 0.55  # L/kg
    volume_distribution_L = v_factor * body_weight_kg 
    
    # Calculate concentration in g/L
    C0 = alcohol_mass_g / volume_distribution_L
    return C0, volume_distribution_L

def pieters_model(C, t, k1, k2, a, vmax, km):
    dC1dt = -k1/(1 + a*C[0]**2) * C[0]
    dC2dt = k1/(1 + a*C[0]**2) * C[0] - k2*C[1]
    dC3dt = k2*C[1] - (vmax/(km + C[2]))*C[2]
    return [dC1dt, dC2dt, dC3dt]

def simulate_bac_pieters(gender, time_span=24, num_points=51, 
                num_glasses=3, glass_volume_ml=150, abv=12, body_weight_kg=70):
    params = {
        'male': {
            'vmax': 0.470,
            'km': 0.380,
            'k1': 5.55,
            'k2': 7.05,
            'a': 0.42
        },
        'female': {
            'vmax': 0.480,
            'km': 0.405,
            'k1': 4.96,
            'k2': 4.96,
            'a': 0.75
        }
    }
    
    p = params[gender]
    C0, volume_distribution_L = calculate_initial_concentration(num_glasses, glass_volume_ml, abv, 
                                      body_weight_kg, gender)
    
    t = np.linspace(0, time_span, num_points)
    initial_conditions = [C0, 0, 0]
    
    solution = odeint(pieters_model, initial_conditions, t, 
                     args=(p['k1'], p['k2'], p['a'], p['vmax'], p['km']))
    
    return t, solution, C0, volume_distribution_L

def two_phase_model(num_glasses, glass_volume_ml, abv, body_weight_kg, stomach_level, gender, time_span):
    # Constants
    k_abs = 0.173 if stomach_level == 'full' else 0.693  # Absorption rate constant (per hour)
    r = 0.7 if gender == 'male' else 0.6 # The volumeof distribution (Vd) of ethanol
    Cl_std = 0.24  # Standard clearance rate in L/h
    Vd = 62.5  # Volume of distribution constant in L

    # Calculate total alcohol volume in grams
    total_volume_ml = num_glasses * glass_volume_ml
    alcohol_volume_ml = total_volume_ml * (abv / 100)
    alcohol_grams = alcohol_volume_ml * 0.789  # Convert to grams
    b = alcohol_grams # Alcohol concentration in grams/ml

    # Time points
    num_points=51
    
    time = np.linspace(0, time_span, num_points)

    # Clearance and excretion rate
    Cl = Cl_std * ((body_weight_kg / 70) ** 0.75)  # Adjust clearance for weight
    k_exc = 0.15 + Cl / Vd  # Excretion rate g/L

    # Initialize values
    GI_vals = [0]  # GI alcohol storage over time
    BAC_vals = [0]  # Blood Alcohol Concentration over time

    for t in range(1, num_points):
        dt = time[t] - time[t - 1]  # Time step

        # Absorption Phase
        if time[t] < 1:
            GI = (1 - k_abs ) * GI_vals[-1] + b
        else:
            GI = (1 - k_abs ) * GI_vals[-1]
        GI_vals.append(GI)
        
    # Excretion Phase - Calculate BAC
    counter = 0
    for GI in GI_vals[1:]:
        BAC = (1 - k_exc ) * BAC_vals[-1] + (k_abs *  GI) / (body_weight_kg * r)
        BAC_vals.append(BAC)
        counter += 1
        #st.write(f"counter {counter}, bac {BAC}")
    
    return time, np.array(BAC_vals)

def find_threshold_times(t, values, threshold):
    over_threshold = next((t[i] for i, v in enumerate(values) if v > threshold), None)
    under_threshold = next((t[i] for i in range(len(values)-1, -1, -1) if values[i] < threshold), None)
    
    if over_threshold is None:
        return "Not Reached", "Not Reached"
    return f"{over_threshold:.2f}", f"{under_threshold:.2f}" if under_threshold is not None else "Not Reached"

def calculate_mae(actual_times, actual_values, forecast_times, forecast_values):
    """
    Calculate Mean Absolute Percentage Error with careful interpolation
    
    Args:
        actual_times: Time points for actual values
        actual_values: Actual BAC values
        forecast_times: Time points for forecast values
        forecast_values: Forecast BAC values
    
    Returns:
        MAE percentage
    """
    from scipy.interpolate import interp1d
    
    # Create interpolation function for forecast
    forecast_interp_func = interp1d(forecast_times, forecast_values, 
                                     kind='linear', fill_value='extrapolate')
    
    # Interpolate forecast values to match actual time points
    forecast_at_actual_times = forecast_interp_func(actual_times)
    
    # Calculate percentage errors, handling potential zero division
    errors = np.abs(actual_values - forecast_at_actual_times)
    
    # Calculate mean percentage error
    mae = np.mean(errors)
    
    return mae

def export_model_results_to_csv(t_pieters, solution_pieters, t_two_phase, bac_two_phase):
    """
    Export BAC model results to CSV files
    
    Args:
        t_pieters: Time points for Pieter's model
        solution_pieters: Pieter's model solution array
        t_two_phase: Time points for two-phase model
        bac_two_phase: Two-phase model BAC values
    """
    import pandas as pd
    
    # Pieter's Model DataFrame
    pieters_df = pd.DataFrame({
        'Time (hours)': t_pieters,
        'Compartment 1': solution_pieters[:, 0],
        'Compartment 2': solution_pieters[:, 1],
        'BAC (g/L)': solution_pieters[:, 2]
    })
    
    # Two-Phase Model DataFrame
    two_phase_df = pd.DataFrame({
        'Time (hours)': t_two_phase,
        'BAC (g/L)': bac_two_phase
    })
    
    # Create export directory if it doesn't exist
    import os
    os.makedirs('bac_model_results', exist_ok=True)
    
    # Export to CSV
    pieters_df.to_csv('bac_model_results/pieters_model.csv', index=False)
    two_phase_df.to_csv('bac_model_results/two_phase_model.csv', index=False)
    
    # Add Streamlit notification
    st.success("Model results exported to 'bac_model_results' directory!")

def main():
    st.set_page_config(page_title="BAC Model Comparison", layout="wide")
    st.title("Comparison: Pieter's 3-Compartment Model vs Two-Phase Model")
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Drink type selector
    selected_drink = st.sidebar.selectbox(
        "Select Drink Type",
        options=list(DRINK_TYPES.keys()),
        index=0
    )
    
    # Display drink description
    st.sidebar.info(DRINK_TYPES[selected_drink]["description"])
    
    # Get default values based on selected drink
    default_volume = DRINK_TYPES[selected_drink]["volume"]
    default_abv = DRINK_TYPES[selected_drink]["abv"]
    
    # Allow manual override of drink parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Drink Parameters")
    if selected_drink == "Custom":
        glass_volume_ml = st.sidebar.slider("Glass Volume (mL)", min_value=30, max_value=1000, value=355, step=5)
        abv = st.sidebar.slider("Alcohol Content (ABV %)", min_value=3.0, max_value=40.0, value=5.0, step=0.5)
    else:
        glass_volume_ml = st.sidebar.slider("Glass Volume (mL)", min_value=30, max_value=1000, value=default_volume, step = 5)
        abv = st.sidebar.slider("Alcohol Content (ABV %)", min_value=3.0, max_value=40.0, value=default_abv)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Consumption & Personal Parameters")
    num_glasses = st.sidebar.slider("Number of Drinks", min_value=1, max_value=15, value=5)
    
    # Calculate and display total alcohol content
    total_alcohol_ml = num_glasses * glass_volume_ml * (abv/100)
    st.sidebar.markdown(f"Total Pure Alcohol: **{total_alcohol_ml:.1f} mL**")
    
    gender = st.sidebar.radio("Select Gender", options=['male', 'female'])
    stomach_level = st.sidebar.radio("Select stomach level", options=['full', 'empty'])
    body_weight_kg = st.sidebar.slider("Body Weight (kg)", min_value=40, max_value=150, value=80)
    
    # Add time span selection
    #time_span = st.sidebar.slider("Simulation Time (hours)", min_value=6, max_value=60, value=24)
    time_span = 24
    
    # Display drink summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("Drink Summary")
    st.sidebar.markdown(f"""
    - Drink Type: **{selected_drink}**
    - Volume per drink: **{glass_volume_ml} mL**
    - ABV: **{abv}%**
    - Number of drinks: **{num_glasses}**
    - Total volume: **{num_glasses * glass_volume_ml} mL**
    """)
    
    # Simulate both models
    t_pieters, solution_pieters, C0, vd = simulate_bac_pieters(
        gender, time_span=time_span, num_glasses=num_glasses, 
        glass_volume_ml=glass_volume_ml, abv=abv, body_weight_kg=body_weight_kg
    )
    
    t_two_phase, bac_two_phase = two_phase_model(
        num_glasses, glass_volume_ml, abv, body_weight_kg, stomach_level, gender, time_span=time_span
    )
    
    # After model simulation and before displaying metrics
    #export_model_results_to_csv(t_pieters, solution_pieters, t_two_phase, bac_two_phase)
    
    # Calculate MAPE using improved function
    mape = calculate_mae(
        actual_times=t_pieters, 
        actual_values=solution_pieters[:, 2], 
        forecast_times=t_two_phase, 
        forecast_values=bac_two_phase
    )
    
    # Create subplot
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=(f"Pieter's 3-Compartment Model ({selected_drink})", 
                                     f"Two-Phase Model ({selected_drink})"))
    
    # Pieter's model plot
    fig.add_trace(
        go.Scatter(x=t_pieters, y=solution_pieters[:, 2], name="Pieter's Model", line_color='blue'),
        row=1, col=1
    )
    
    # Two-phase model plot
    fig.add_trace(
        go.Scatter(x=t_two_phase, y=bac_two_phase, name="Two-Phase Model", line_color='red'),
        row=1, col=2
    )
    
    # Add threshold lines
    for col in [1, 2]:
        fig.add_hline(y=0.8, line_dash="dash", line_color="yellow", 
                     row=1, col=col, annotation_text="Legal Limit (0.8)")
        fig.add_hline(y=4.0, line_dash="dash", line_color="red", 
                     row=1, col=col, annotation_text="Fatal Limit (4.0)")
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="BAC (g/L)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate metrics
    max_bac_pieters = np.max(solution_pieters[:, 2])
    time_to_peak_pieters = t_pieters[np.argmax(solution_pieters[:, 2])]
    
    max_bac_two_phase = np.max(bac_two_phase)
    time_to_peak_two_phase = t_two_phase[np.argmax(bac_two_phase)]
    
    # Calculate differences
    bac_diff = ((max_bac_two_phase - max_bac_pieters) / max_bac_pieters) * 100
    peak_time_diff = ((time_to_peak_two_phase - time_to_peak_pieters) / time_to_peak_pieters) * 100
    
    st.metric("Model Accuracy / Mean Absolute Error (MAE)", f"{mape:.2f} g/L")
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pieter's Model Metrics")
        st.metric("Maximum BAC", f"{max_bac_pieters:.3f} g/L")
        st.metric("Time to Peak", f"{time_to_peak_pieters:.2f} hours")
        
        over_legal, under_legal = find_threshold_times(t_pieters, solution_pieters[:, 2], 0.8)
        over_fatal, under_fatal = find_threshold_times(t_pieters, solution_pieters[:, 2], 2.0)
        
        st.metric("Time Over Legal Limit", over_legal)
        #st.metric("Time Under Legal Limit", under_legal)
        st.metric("Time Over Fatal Limit", over_fatal)
        #st.metric("Time Under Fatal Limit", under_fatal)
    
    with col2:
        st.subheader("Two-Phase Model Metrics")
        st.metric("Maximum BAC", f"{max_bac_two_phase:.3f} g/L", f"{bac_diff:+.1f}%")
        st.metric("Time to Peak", f"{time_to_peak_two_phase:.2f} hours", f"{peak_time_diff:+.1f}%")
        
        over_legal, under_legal = find_threshold_times(t_two_phase, bac_two_phase, 0.8)
        over_fatal, under_fatal = find_threshold_times(t_two_phase, bac_two_phase, 2.0)
        
        st.metric("Time Over Legal Limit", over_legal)
        #st.metric("Time Under Legal Limit", under_legal)
        st.metric("Time Over Fatal Limit", over_fatal)
        #st.metric("Time Under Fatal Limit", under_fatal)

if __name__ == "__main__":
    main()

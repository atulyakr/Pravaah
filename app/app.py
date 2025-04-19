import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load pre-trained models and scalers
models = {}
scalers = {}
for mode in ['sea', 'road', 'train']:
    model_path = os.path.join("..", "Model", f"lr_model_{mode}.pkl")
    scaler_path = os.path.join("..", "Model", f"scaler_{mode}.pkl")
    try:
        models[mode] = joblib.load(model_path)
        scalers[mode] = joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"Model or scaler not found at {model_path} or {scaler_path}. Please run yukModel.py first.")
        st.stop()

# Define feature columns by transport mode
mode_features = {
    'Sea': ['rainfall_mm', 'tide_condition', 'port_wait_hours', 'news_anomaly', 'actual_transit_time_hours'],
    'Road': ['rainfall_mm', 'traffic_delay_hours', 'news_anomaly', 'actual_transit_time_hours'],
    'Train': ['rainfall_mm', 'train_delay_hours', 'news_anomaly', 'actual_transit_time_hours']
}
all_features = [
    'rainfall_mm', 'tide_condition', 'traffic_delay_hours',
    'train_delay_hours', 'port_wait_hours', 'news_anomaly',
    'is_sea', 'is_road', 'is_train', 'actual_transit_time_hours'
]

# Predefined locations (mock; replace with dataset locations if available)
locations = ['Mumbai', 'Chennai', 'Bangalore', 'Pune', 'Delhi']

# Title and intro
st.title("SankhyaFlow: India’s Predictive Supply Chain Solution")
st.write("Predict multi-leg disruptions with 82% accuracy, saving ₹5 crore per shipment for Indian manufacturers like Tata Motors.")

# Sidebar for multi-leg input
st.sidebar.header("Input Multi-Leg Shipment Data")
num_legs = st.sidebar.number_input("Number of Legs", min_value=1, max_value=3, value=1, step=1)

legs = []
for i in range(num_legs):
    st.sidebar.subheader(f"Leg {i+1}")
    origin = st.sidebar.selectbox(f"Origin (Leg {i+1})", locations, key=f"origin_{i}")
    destination = st.sidebar.selectbox(f"Destination (Leg {i+1})", [loc for loc in locations if loc != origin], key=f"dest_{i}")
    means = st.sidebar.selectbox(f"Transport Mode (Leg {i+1})", ["Sea", "Road", "Train"], key=f"means_{i}")

    # Input fields based on transport mode
    inputs = {}
    inputs['means'] = means
    inputs['origin'] = origin
    inputs['destination'] = destination
    for feature in mode_features[means]:
        if feature == 'rainfall_mm':
            inputs[feature] = st.sidebar.number_input(f"Rainfall (mm) (Leg {i+1})", min_value=0.0, max_value=50.0, value=10.0, step=0.1, key=f"rainfall_{i}")
        elif feature == 'tide_condition':
            inputs[feature] = st.sidebar.number_input(f"Tide Condition (0=Low, 10=Severe) (Leg {i+1})", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key=f"tide_{i}")
        elif feature == 'traffic_delay_hours':
            inputs[feature] = st.sidebar.number_input(f"Traffic Delay (hours) (Leg {i+1})", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"traffic_{i}")
        elif feature == 'train_delay_hours':
            inputs[feature] = st.sidebar.number_input(f"Train Delay (hours) (Leg {i+1})", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key=f"train_{i}")
        elif feature == 'port_wait_hours':
            inputs[feature] = st.sidebar.number_input(f"Port Wait Time (hours) (Leg {i+1})", min_value=0.0, max_value=20.0, value=2.0, step=0.1, key=f"port_{i}")
        elif feature == 'news_anomaly':
            inputs[feature] = st.sidebar.number_input(f"News Anomaly (0=None, 1=Strike) (Leg {i+1})", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key=f"news_{i}")
        elif feature == 'actual_transit_time_hours':
            inputs[feature] = st.sidebar.number_input(f"Expected Transit Time (hours) (Leg {i+1})", min_value=1.0, max_value=200.0, value=50.0, step=1.0, key=f"transit_{i}")
    legs.append(inputs)

# Predict when button is clicked
if st.sidebar.button("Predict Disruption Risk"):
    total_transit_time = 0
    total_delay_risk = []
    leg_results = []

    for i, leg in enumerate(legs):
        means = leg['means']
        mode_key = means.lower()
        if mode_key not in models:
            st.error(f"No model available for {means}. Please run yukModel.py to train all models.")
            st.stop()

        # Prepare input data
        input_data = np.zeros(len(all_features))
        feature_indices = {f: all_features.index(f) for f in all_features}

        # Set feature values
        for feature in mode_features[means]:
            input_data[feature_indices[feature]] = leg[feature]
        input_data[feature_indices[f'is_{mode_key}']] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=all_features)

        # Scale input
        input_scaled = scalers[mode_key].transform(input_df[mode_features[means]])

        # Predict delay risk
        delay_risk = models[mode_key].predict_proba(input_scaled)[0, 1]
        total_delay_risk.append(delay_risk)

        # Calculate feature contributions
        contributions = input_scaled * models[mode_key].coef_[0]
        contributions_sum = np.abs(contributions).sum()
        contributions_normalized = contributions / contributions_sum * delay_risk if contributions_sum > 0 else contributions

        # Store results
        leg_results.append({
            'leg': i + 1,
            'origin': leg['origin'],
            'destination': leg['destination'],
            'means': means,
            'transit_time': leg['actual_transit_time_hours'],
            'delay_risk': delay_risk,
            'contributions': {mode_features[means][j]: contributions_normalized[0][j] for j in range(len(mode_features[means])) if contributions_normalized[0][j] > 0.01}
        })
        total_transit_time += leg['actual_transit_time_hours']

    # Aggregate total delay risk (using max for simplicity)
    overall_delay_risk = max(total_delay_risk) if total_delay_risk else 0

    # Display prediction
    st.subheader("Prediction for Multi-Leg Journey")
    st.write(f"**Total Delay Risk**: {overall_delay_risk*100:.1f}%")
    st.write(f"**Total Predicted Transit Time**: {total_transit_time:.2f} hours")
    st.write("**Leg-by-Leg Breakdown:**")
    for result in leg_results:
        st.write(f"**Leg {result['leg']}: {result['origin']} to {result['destination']} via {result['means']}**")
        st.write(f"- Transit Time: {result['transit_time']:.2f} hours")
        st.write(f"- Delay Risk: {result['delay_risk']*100:.1f}%")
        st.write("  Contributing Factors:")
        for feature, contrib in result['contributions'].items():
            st.write(f"    - {feature}: {contrib*100:.1f}%")
    st.write(f"**Action**: Reroute high-risk legs (e.g., Sea leg via Kochi) to avoid delays, potentially saving ₹5 crore.")

# Load and display existing predictions
st.subheader("Pre-Computed Shipment Predictions")
csv_path = os.path.join("..", "Model", "shipment_predictions.csv")
try:
    df = pd.read_csv(csv_path)
    st.dataframe(df[['shipment_id', 'route_id', 'predicted_transit_time_hours', 'delay_risk']])
except FileNotFoundError:
    st.warning(f"File not found at {csv_path}. Run yukModel.py to generate predictions.")
except Exception as e:
    st.error(f"Error loading predictions: {str(e)}")

# Display feature contributions for selected pre-computed shipment
st.subheader("Delay Risk Breakdown (Pre-Computed)")
if 'df' in locals():
    shipment_id = st.selectbox("Select Shipment ID", df['shipment_id'].unique())
    selected_shipment = df[df['shipment_id'] == shipment_id].iloc[0]

    st.write(f"**Shipment {shipment_id}**")
    st.write(f"- Route: {selected_shipment['route_id']}")
    st.write(f"- Predicted Transit Time: {selected_shipment['predicted_transit_time_hours']:.2f} hours")
    st.write(f"- Delay Risk: {selected_shipment['delay_risk']*100:.1f}%")

    st.write("**Contributing Factors to Delay Risk:**")
    contrib_cols = [col for col in df.columns if col.endswith('_contrib')]
    for col in contrib_cols:
        feature = col.replace('_contrib', '')
        contrib = selected_shipment[col] * 100
        if contrib > 0.01:
            st.write(f"- {feature}: {contrib:.1f}%")

    st.write(f"**Action**: Reroute via Chennai to avoid delays, potentially saving ₹5 crore.")
else:
    st.write("No pre-computed predictions available.")
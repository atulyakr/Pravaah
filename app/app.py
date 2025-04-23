import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, date
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
import requests
import streamlit.components.v1 as components
from functools import lru_cache


# Detect the base directory (repository root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__ else os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "Model")

# Load models
models = {}
scalers = {}
for mode in ['sea', 'road', 'train']:
    model_path = os.path.join(MODEL_DIR, f"lr_model_{mode}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{mode}.pkl")
    try:
        models[mode] = joblib.load(model_path)
        scalers[mode] = joblib.load(scaler_path)
    except FileNotFoundError as e:
        st.error(f"Model or scaler not found: {e}. Please run yukModel.py first to generate them.")
        st.stop()


@lru_cache(maxsize=100)
def get_coordinates(place_name):
    geocode_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "pravaah-weather-fetcher"
    }
    try:
        response = requests.get(geocode_url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                return lat, lon
            else:
                raise ValueError(f"No location found for {place_name}.")
        else:
            raise Exception(f"Geocoding request failed: {response.status_code}")
    except Exception as e:
        print(f"Error geocoding {place_name}: {e}")
        return None, None

@lru_cache(maxsize=100)
def fetch_weather(lat, lon, date_str):
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str
    }
    try:
        response = requests.get(weather_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Weather API response for {lat}, {lon} on {date_str}: {data}")
            return data
        else:
            raise Exception(f"Weather request failed: {response.status_code}")
    except Exception as e:
        print(f"Error fetching weather for {lat}, {lon} on {date_str}: {e}")
        return None

def get_precipitation_probability(city, dispatch_date=None, dispatch_time=None):
    try:
        lat, lon = get_coordinates(city)
        if lat is None or lon is None:
            print(f"No coordinates for {city}, returning default 10.0 mm")
            return 10.0

        # Use current date if dispatch_date is None
        date_to_use = dispatch_date if dispatch_date else date.today()
        date_str = date_to_use.strftime("%Y-%m-%d")
        weather_data = fetch_weather(lat, lon, date_str)
        if weather_data is None:
            print(f"No weather data for {city} on {date_str}, returning default 10.0 mm")
            return 10.0

        hourly = weather_data.get("hourly", {})
        times = hourly.get("time", [])
        precipitations = hourly.get("precipitation", [])

        if not times or not precipitations:
            print(f"No hourly precipitation data for {city} on {date_str}, returning default 10.0 mm")
            return 10.0

        # If dispatch_time is provided, find the closest hour
        if dispatch_time:
            target_time = datetime.combine(date_to_use, dispatch_time)
            target_str = target_time.strftime("%Y-%m-%dT%H:00")
            for t, p in zip(times, precipitations):
                if t == target_str:
                    print(f"Found exact match for {city} at {target_str}: {p} mm")
                    return min(max(p, 0.0), 50.0)  # Cap at 50 mm
            # If no exact match, use closest hour
            target_hour = target_time.hour
            closest_idx = min(range(len(times)), key=lambda i: abs(int(times[i][-5:-3]) - target_hour))
            rainfall_mm = precipitations[closest_idx]
            print(f"Closest match for {city} at {times[closest_idx]}: {rainfall_mm} mm")
            return min(max(rainfall_mm, 0.0), 50.0)
        else:
            # Use average daily precipitation
            avg_rain = np.mean([p for p in precipitations if p is not None]) if precipitations else 10.0
            print(f"Average daily precipitation for {city} on {date_str}: {avg_rain} mm")
            return min(max(avg_rain, 0.0), 50.0)
    except Exception as e:
        print(f"Error processing weather for {city}: {e}")
        return 10.0

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in .env file. Please set GEMINI_API_KEY.")
    st.stop()
if not OPENCAGE_API_KEY:
    st.error("OpenCage API key not found in .env file. Please set OPENCAGE_API_KEY.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Cache file for API responses
CACHE_FILE = "route_data_cache.json"
def load_cache():
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Function to geocode location using OpenCage API
def geocode_location(location):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}"
    try:
        res = requests.get(url).json()
        if res['results']:
            return res['results'][0]['geometry']
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to geocode location {location}: {e}")
        return None

# Function to estimate distance and transit time with Gemini
def estimate_transit_time_with_gemini(origin, destination, mode):
    cache_key = f"{origin}_{destination}_{mode}"
    cache = load_cache()
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""
    You are a logistics assistant. Estimate the approximate distance and realistic transit time between two Indian cities given a transport mode.
    For a shipment from {origin} to {destination} via {mode} transport within India, provide:
    1. **Distance**: Approximate distance in kilometers.
    2. **Transit Time**: Expected transit time in hours (Sea: ~20 knots, Road: ~50 km/h, Train: ~60 km/h).
    
    Format response as JSON:
    ```json
    {{
      "distance_km": float,
      "transit_time_hours": float
    }}
    ```
    """
    try:
        response = model.generate_content(prompt)
        # Attempt to parse JSON response
        try:
            data = json.loads(response.text.strip('```json\n').strip('```'))
            cache[cache_key] = data
            save_cache(cache)
            return data
        except json.JSONDecodeError:
            # Fallback: parse text with regex if JSON fails
            match = re.search(r"(\d+(?:\.\d+)?)\s*hours?", response.text.lower())
            if match:
                transit_time = float(match.group(1))
                # Estimate distance based on mode and transit time
                speed = {'sea': 37, 'road': 50, 'train': 60}  # km/h (20 knots ~37 km/h)
                distance = transit_time * speed.get(mode.lower(), 50)
                data = {"distance_km": distance, "transit_time_hours": transit_time}
                cache[cache_key] = data
                save_cache(cache)
                return data
            return None
    except Exception as e:
        st.warning(f"Failed to estimate transit time using Gemini: {e}")
        return None

# Function to generate suggestions with Gemini
def generate_suggestions_with_gemini(leg_results):
    # Create a copy of leg_results with dispatch_date and dispatch_time as strings
    serialized_results = []
    for result in leg_results:
        serialized_result = result.copy()
        serialized_result['dispatch_date'] = str(result['dispatch_date'])
        serialized_result['dispatch_time'] = result['dispatch_time'].strftime("%H:%M") if result['dispatch_time'] else ""
        serialized_results.append(serialized_result)
    
    prompt = f"""
    You are a logistics expert. Analyze the following shipment prediction report and provide actionable suggestions to mitigate delays for each leg.
    Report:
    {json.dumps(serialized_results, indent=2)}
    
    For each leg, suggest specific actions (e.g., reroute via another city, adjust schedule, use alternative transport mode) based on delay risk and contributing factors.
    Format response as JSON:
    ```json
    {{
      "suggestions": [
        {{
          "leg": int,
          "suggestion": str
        }}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        try:
            data = json.loads(response.text.strip('```json\n').strip('```'))
            return data['suggestions']
        except json.JSONDecodeError:
            st.warning("Failed to parse Gemini suggestions. Using fallback.")
            return [{"leg": result['leg'], "suggestion": "Consider rerouting or adjusting schedule to avoid delays."} for result in leg_results]
    except Exception as e:
        st.warning(f"Failed to generate suggestions using Gemini: {e}")
        return [{"leg": result['leg'], "suggestion": "Consider rerouting or adjusting schedule to avoid delays."} for result in leg_results]

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

# Title
st.title("Pravaah: Predictive Supply Chain Solution")

# Sidebar for multi-leg input
st.sidebar.header("Input Multi-Leg Shipment Data")
num_legs = st.sidebar.number_input("Number of Legs", min_value=1, max_value=3, value=1, step=1)

legs = []
for i in range(num_legs):
    st.sidebar.subheader(f"Leg {i+1}")
    origin = st.sidebar.text_input(f"Origin (Leg {i+1}, e.g., Mumbai, Kochi)", value="", key=f"origin_{i}")
    destination = st.sidebar.text_input(f"Destination (Leg {i+1}, e.g., Chennai, Jaipur)", value="", key=f"dest_{i}")
    means = st.sidebar.selectbox(f"Transport Mode (Leg {i+1})", ["Sea", "Road", "Train"], key=f"means_{i}")
    dispatch_date = st.sidebar.date_input(f"Dispatch Date (Leg {i+1})", value=date.today(), min_value=date(2025, 4, 19), key=f"date_{i}")
    dispatch_time = st.sidebar.time_input(f"Dispatch Time (Leg {i+1})", value=datetime.now().time(), key=f"time_{i}")

    # Validate origin and destination
    if origin and destination:
        if origin.strip().lower() == destination.strip().lower():
            st.sidebar.error(f"Origin and destination cannot be the same for Leg {i+1}.")
            continue
    elif not origin or not destination:
        st.sidebar.warning(f"Please enter both origin and destination for Leg {i+1}.")
        continue

    # Fetch rainfall_mm from weather API for specific date and time
    rainfall_mm = get_precipitation_probability(origin, dispatch_date, dispatch_time)
    st.sidebar.write(f"Auto-fetched Rainfall (mm) for {origin} on {dispatch_date} {dispatch_time}: {rainfall_mm:.1f}")

    # Use Gemini to estimate distance and transit time
    route_data = estimate_transit_time_with_gemini(origin, destination, means)
    if route_data is None:
        route_data = {"distance_km": 1000.0, "transit_time_hours": 50.0}
        st.sidebar.warning(f"Using fallback values for Leg {i+1}: Distance 1000 km, Transit Time 50 hours.")
    st.sidebar.write(f"Auto-fetched Distance via Gemini: {route_data['distance_km']:.1f} km")
    st.sidebar.write(f"Auto-fetched Transit Time via Gemini: {route_data['transit_time_hours']:.1f} hours")

    # Input fields based on transport mode (excluding rainfall_mm and transit time)
    inputs = {
        'means': means,
        'origin': origin,
        'destination': destination,
        'dispatch_date': dispatch_date,
        'dispatch_time': dispatch_time,
        'rainfall_mm': rainfall_mm,
        'actual_transit_time_hours': route_data['transit_time_hours']
    }
    for feature in mode_features[means]:
        if feature in ['rainfall_mm', 'actual_transit_time_hours']:
            continue
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
    legs.append(inputs)

# Predict when button is clicked
if st.sidebar.button("Predict Disruption Risk"):
    if not legs:
        st.error("Please provide valid origin and destination for all legs.")
        st.stop()

    total_transit_time = 0
    total_delay_risk = []
    leg_results = []

    for i, leg in enumerate(legs):
        means = leg['means']
        mode_key = means.lower()
        if mode_key not in models:
            st.error(f"No model available for {means}. Please run yukModel.py to train all models.")
            st.stop()

        input_data = np.zeros(len(all_features))
        feature_indices = {f: all_features.index(f) for f in all_features}

        for feature in mode_features[means]:
            input_data[feature_indices[feature]] = leg[feature]
        input_data[feature_indices[f'is_{mode_key}']] = 1

        input_df = pd.DataFrame([input_data], columns=all_features)
        input_scaled = scalers[mode_key].transform(input_df[mode_features[means]])
        delay_risk = models[mode_key].predict_proba(input_scaled)[0, 1]
        total_delay_risk.append(delay_risk)

        contributions = input_scaled * models[mode_key].coef_[0]
        contributions_sum = np.abs(contributions).sum()
        contributions_normalized = contributions / contributions_sum * delay_risk if contributions_sum > 0 else contributions

        leg_results.append({
            'leg': i + 1,
            'origin': leg['origin'],
            'destination': leg['destination'],
            'means': means,
            'transit_time': leg['actual_transit_time_hours'],
            'dispatch_date': leg['dispatch_date'],
            'dispatch_time': leg['dispatch_time'],
            'delay_risk': delay_risk,
            'contributions': {mode_features[means][j]: contributions_normalized[0][j] for j in range(len(mode_features[means])) if contributions_normalized[0][j] > 0.01}
        })
        total_transit_time += leg['actual_transit_time_hours']

    # Generate suggestions using Gemini
    suggestions = generate_suggestions_with_gemini(leg_results)

    # Display prediction
    st.subheader("Prediction for Multi-Leg Journey")
    st.write(f"**Total Delay Risk**: {max(total_delay_risk)*100:.1f}%")
    st.write(f"**Total Predicted Transit Time**: {total_transit_time:.2f} hours")
    st.write("**Leg-by-Leg Breakdown:**")
    for result in leg_results:
        st.write(f"**Leg {result['leg']}: {result['origin']} to {result['destination']} via {result['means']}**")
        st.write(f"- Dispatch Date: {result['dispatch_date']}")
        st.write(f"- Dispatch Time: {result['dispatch_time']}")
        st.write(f"- Transit Time: {result['transit_time']:.2f} hours")
        st.write(f"- Delay Risk: {result['delay_risk']*100:.1f}%")
        st.write("  Contributing Factors:")
        for feature, contrib in result['contributions'].items():
            st.write(f"    - {feature}: {contrib*100:.1f}%")
        # Display suggestion
        suggestion = next((s['suggestion'] for s in suggestions if s['leg'] == result['leg']), "No specific suggestion available.")
        st.write(f"  **Suggestion**: {suggestion}")

        # Embed Google Maps iframe for origin
        st.subheader(f"Map for Leg {result['leg']} (Origin: {result['origin']})")
        coords = geocode_location(result['origin'])
        if coords:
            map_url = f"https://www.google.com/maps/embed/v1/place?q={coords['lat']},{coords['lng']}&key=AIzaSyB41DRUbKWJHPxaFjO2vcwbwzrjek4cS7w&zoom=12"
            components.html(f'<iframe width="600" height="400" frameborder="0" style="border:0" src="{map_url}" allowfullscreen></iframe>', height=400)
        else:
            st.warning(f"Could not display map for {result['origin']}. Location not found.")

    st.write("**Action**: Consider rerouting high-risk legs or adjusting schedules based on suggestions below.")
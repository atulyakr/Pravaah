import requests
import numpy as np
from functools import lru_cache
from datetime import datetime, date

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
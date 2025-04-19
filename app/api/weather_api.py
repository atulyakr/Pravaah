import requests
import time
from datetime import datetime

def get_coordinates(place_name):
    geocode_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "weather-fetcher-script"
    }
    response = requests.get(geocode_url, params=params, headers=headers)
    if response.status_code == 200:
        results = response.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return lat, lon
        else:
            raise ValueError("No location found.")
    else:
        raise Exception(f"Geocoding request failed: {response.status_code}")

def fetch_weather(lat, lon):
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
        "timezone": "auto"
    }

    response = requests.get(weather_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Weather request failed: {response.status_code}")

def display_weather(data):
    current = data.get("current", {})
    print("\n🌡️ Current Weather:")
    print(f"  Time: {current.get('time')}")
    print(f"  Temperature: {current.get('temperature_2m')} °C")
    print(f"  Wind Speed: {current.get('wind_speed_10m')} km/h")

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    humidity = hourly.get("relative_humidity_2m", [])
    wind_speed = hourly.get("wind_speed_10m", [])
    rain_prob = hourly.get("precipitation_probability", [])

    print("\n🔮 Hourly Forecast (Next 5 Hours):")
    if times:
        current_time = current.get("time")
        try:
            i = times.index(current_time)
        except ValueError:
            now = datetime.strptime(current_time, "%Y-%m-%dT%H:%M")
            i = min(range(len(times)), key=lambda j: abs(datetime.strptime(times[j], "%Y-%m-%dT%H:%M") - now))

        for j in range(i, min(i + 5, len(times))):
            print(f"{times[j]} | Temp: {temps[j]}°C | Humidity: {humidity[j]}% | Wind: {wind_speed[j]} km/h | Rain: {rain_prob[j]}%")

    # Show 5-day forecast
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    max_temp = daily.get("temperature_2m_max", [])
    min_temp = daily.get("temperature_2m_min", [])
    max_rain = daily.get("precipitation_probability_max", [])
    max_wind = daily.get("wind_speed_10m_max", [])

    print("\n📆 5-Day Forecast:")
    for i in range(min(5, len(dates))):
        print(f"{dates[i]} | Max: {max_temp[i]}°C | Min: {min_temp[i]}°C | Rain: {max_rain[i]}% | Wind: {max_wind[i]} km/h")

def main():
    place = input("Enter a place name (e.g., Berlin, New York): ").strip()
    try:
        lat, lon = get_coordinates(place)
        print(f"📍 Coordinates for {place}: Latitude {lat}, Longitude {lon}")
        time.sleep(1)
        weather_data = fetch_weather(lat, lon)
        display_weather(weather_data)
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    main()

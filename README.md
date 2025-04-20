# Pravaah: India’s Predictive Supply Chain Solution

Pravaah is a real-time supply chain monitoring and predictive analytics system that helps businesses proactively manage and reduce disruptions in logistics. Designed especially for Indian manufacturers, this tool anticipates risks and delays using advanced regression-based models and real-time data streams.

## 🚛 Problem Statement

Supply chain disruptions can severely affect operations, causing delays and inflating costs. These issues stem from:

- Traffic congestion  
- Weather anomalies  
- Logistic inefficiencies  
- Labor strikes or other news anomalies  

These unforeseen events make timely delivery difficult and expensive. A smarter, data-driven approach is necessary.

## 🌟 Solution Overview

**Pravaah** addresses these issues with a real-time predictive system powered by:

- 🧠 Regression models for delay prediction  
- ☁️ Real-time data feeds (weather, traffic, news)  
- 📍 Location tracking  
- 💬 Human-AI interaction using Gemini for intelligent recommendations  

## 🛠️ MVP Features

### 1. 📈 Predictive Analytics with Regression Model  
- Uses linear/logistic regression to estimate delay risks and transit times based on real-time factors such as traffic, rainfall, and past shipment data.

### 2. 📡 Real-Time Data Integration  
- Weather updates (rainfall, storms, etc.)  
- Traffic congestion data  
- News-based anomaly detection (e.g., strikes, road closures)

### 3. 🚨 Alerts & Risk Assessment  
- Delay risk predictions per leg of the journey  
- Total predicted transit time  
- Suggested reroutes to avoid delays

### 4. 💬 Conversational Interface (Gemini)  
- Text-based user guidance to help decision-makers understand disruptions and respond quickly

## 📊 Example Output (UI Snapshot)

**Route:** Mumbai ➝ Kochi via Road  
- 🕒 Transit Time: 50.00 hrs  
- ⚠️ Delay Risk: 100%  
- 🔁 Suggested Action: Switch to Sea leg via Kochi to minimize delay

## 🚀 How It Works

1. User inputs shipment legs (e.g., Mumbai to Kochi via Road)  
2. System auto-fetches live data (rainfall, traffic, anomalies)  
3. Regression model calculates delay probability and expected transit time  
4. **Gemini generates actionable insights**  
5. Output displayed in an intuitive interface

## 🧪 Future Enhancements

- Multi-leg optimization with real-time recalculations  
- Route cost estimation and comparison  
- Voice-based assistant for on-the-go logistics management  
- Mobile app integration

## 📎 Technologies Used

- Python (FastAPI / Streamlit for UI)  
- Scikit-learn (for regression modeling)  
- Map APIs for traffic/weather (Google Maps, **Open-Meteo**)  
- **Gemini** (for intelligent interaction and reasoning)

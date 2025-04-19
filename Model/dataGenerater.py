import pandas as pd
import numpy as np
from datetime import timedelta

# Generate 252 shipments (May 1, 2024 - Jan 7, 2025)
dates = pd.date_range("2024-05-01", "2025-01-07")
shipments = [f"S{str(i).zfill(3)}" for i in range(1, 253)]
df_list = []

for shipment_id in shipments:
    for date in dates:
        # Route assignment (50% Route 1, 50% Route 2)
        route_id = np.random.choice(["Route 1", "Route 2"])
        
        # Route 1: Two segments (Sea, Road)
        if route_id == "Route 1":
            # Primary Segment (Sea: Shanghai -> Mumbai)
            dispatch_dt = pd.Timestamp(date) + timedelta(hours=np.random.randint(0, 24))
            rainfall_sea = (
                np.clip(np.random.normal(loc=25, scale=20), 0, 50) if date.month in [6, 7, 8] else
                np.clip(np.random.normal(loc=10, scale=15), 0, 30) if date.month in [9, 10, 11] else
                np.clip(np.random.normal(loc=1, scale=5), 0, 10)
            )
            tide_condition = (
                np.random.choice([1, 2, 3, 4, 5], p=[0.7, 0.1, 0.1, 0.05, 0.05]) if rainfall_sea > 20 and date.month in [6, 7, 8] else
                np.random.choice([1, 2, 3, 4, 5], p=[0.94, 0.03, 0.02, 0.005, 0.005]) if rainfall_sea > 10 else 1
            )
            port_wait = np.random.uniform(12, 48) if (rainfall_sea > 20 or tide_condition > 2) and np.random.rand() < 0.3 else 0
            news_anomaly = (
                np.random.choice([1, 2, 3, 4, 5], p=[0.85, 0.06, 0.045, 0.03, 0.015]) if rainfall_sea > 15 else
                np.random.choice([1, 2, 3, 4, 5], p=[0.95, 0.02, 0.015, 0.01, 0.005])
            )
            actual_transit_sea = 144 + port_wait + (np.random.uniform(12, 48) if (rainfall_sea > 15 or tide_condition > 2 or news_anomaly > 2) else 0)
            delivery_dt_sea = dispatch_dt + timedelta(hours=actual_transit_sea)
            
            df_list.append({
                "shipment_id": shipment_id,
                "segment_indicator": "Primary",
                "means": "Sea",
                "dispatch_location": "Shanghai",
                "delivery_location": "Mumbai",
                "route_id": "Route 1",
                "dispatch_datetime": dispatch_dt,
                "actual_delivery_datetime": delivery_dt_sea,
                "baseline_transit_time_hours": 144,
                "route_distance_km": 4800,
                "rainfall_mm": rainfall_sea,
                "tide_condition": tide_condition,
                "traffic_delay_hours": 0,
                "train_delay_hours": 0,
                "port_wait_hours": port_wait,
                "news_anomaly": news_anomaly,
                "actual_transit_time_hours": actual_transit_sea,
                "delay_indicator": 1 if actual_transit_sea > 156 else 0
            })
            
            # Secondary Segment (Road: Mumbai -> Pune)
            dispatch_dt_road = delivery_dt_sea + timedelta(hours=2)  # 2-hour handling
            rainfall_road = (
                np.clip(np.random.normal(loc=25, scale=20), 0, 100) if date.month in [6, 7, 8] else
                np.clip(np.random.normal(loc=10, scale=15), 0, 50) if date.month in [9, 10, 11] else
                np.clip(np.random.normal(loc=1, scale=5), 0, 10)
            )
            traffic_delay = np.random.uniform(2, 8) if rainfall_road > 20 and np.random.rand() < 0.4 else 0
            news_anomaly_road = (
                np.random.choice([1, 2, 3, 4, 5], p=[0.85, 0.06, 0.045, 0.03, 0.015]) if rainfall_road > 20 else
                np.random.choice([1, 2, 3, 4, 5], p=[0.95, 0.02, 0.015, 0.01, 0.005])
            )
            actual_transit_road = 6 + traffic_delay + (np.random.uniform(2, 12) if (rainfall_road > 20 or news_anomaly_road > 2) else 0)
            delivery_dt_road = dispatch_dt_road + timedelta(hours=actual_transit_road)
            
            df_list.append({
                "shipment_id": shipment_id,
                "segment_indicator": "Secondary",
                "means": "Road",
                "dispatch_location": "Mumbai",
                "delivery_location": "Pune",
                "route_id": "Route 1",
                "dispatch_datetime": dispatch_dt_road,
                "actual_delivery_datetime": delivery_dt_road,
                "baseline_transit_time_hours": 6,
                "route_distance_km": 150,
                "rainfall_mm": rainfall_road,
                "tide_condition": 1,
                "traffic_delay_hours": traffic_delay,
                "train_delay_hours": 0,
                "port_wait_hours": 0,
                "news_anomaly": news_anomaly_road,
                "actual_transit_time_hours": actual_transit_road,
                "delay_indicator": 1 if actual_transit_road > 18 else 0
            })
        
        # Route 2: Single segment (Train: Ahmedabad -> Pune)
        else:
            dispatch_dt = pd.Timestamp(date) + timedelta(hours=np.random.randint(0, 24))
            rainfall_train = (
                np.clip(np.random.normal(loc=6, scale=10), 0, 40) if date.month in [6, 7, 8] else
                np.clip(np.random.normal(loc=3, scale=8), 0, 20) if date.month in [9, 10, 11] else
                np.clip(np.random.normal(loc=1, scale=5), 0, 10)
            )
            train_delay = np.random.uniform(2, 8) if rainfall_train > 15 and np.random.rand() < 0.2 else 0
            news_anomaly_train = (
                np.random.choice([1, 2, 3, 4, 5], p=[0.85, 0.06, 0.045, 0.03, 0.015]) if rainfall_train > 15 else
                np.random.choice([1, 2, 3, 4, 5], p=[0.95, 0.02, 0.015, 0.01, 0.005])
            )
            actual_transit_train = 32 + train_delay + (np.random.uniform(12, 36) if (rainfall_train > 15 or news_anomaly_train > 2) else 0)
            delivery_dt_train = dispatch_dt + timedelta(hours=actual_transit_train)
            
            df_list.append({
                "shipment_id": shipment_id,
                "segment_indicator": "Single",
                "means": "Train",
                "dispatch_location": "Ahmedabad",
                "delivery_location": "Pune",
                "route_id": "Route 2",
                "dispatch_datetime": dispatch_dt,
                "actual_delivery_datetime": delivery_dt_train,
                "baseline_transit_time_hours": 32,
                "route_distance_km": 530,
                "rainfall_mm": rainfall_train,
                "tide_condition": 1,
                "traffic_delay_hours": 0,
                "train_delay_hours": train_delay,
                "port_wait_hours": 0,
                "news_anomaly": news_anomaly_train,
                "actual_transit_time_hours": actual_transit_train,
                "delay_indicator": 1 if actual_transit_train > 44 else 0
            })

# Create DataFrame
df = pd.DataFrame(df_list)

# Save to Excel
df.to_excel("supply_chain_data.xlsx", index=False)
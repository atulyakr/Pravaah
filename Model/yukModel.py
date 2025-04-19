import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load the dataset
logging.info("Loading dataset...")
print("Loading dataset...")
try:
    df = pd.read_excel("supply_chain_dataset.xlsx")
except FileNotFoundError:
    logging.error("Dataset file 'supply_chain_dataset.xlsx' not found in current directory.")
    raise
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# Step 2: Preprocess the data
logging.info("Preprocessing data...")
print("Preprocessing data...")

# Validate and convert dispatch_datetime
df['dispatch_datetime'] = pd.to_datetime(df['dispatch_datetime'], errors='coerce')
if df['dispatch_datetime'].isna().any():
    logging.warning("Found NaN in dispatch_datetime. Dropping rows with NaN.")
    df = df.dropna(subset=['dispatch_datetime'])

# Create dummy variables for means
df['is_sea'] = (df['means'] == 'Sea').astype(int)
df['is_road'] = (df['means'] == 'Road').astype(int)
df['is_train'] = (df['means'] == 'Train').astype(int)

# Define mode-specific features
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
target_col = 'delay_indicator'

# Validate required columns
missing_cols = [col for col in all_features + [target_col] if col not in df.columns]
if missing_cols:
    logging.error(f"Missing columns in dataset: {missing_cols}")
    raise ValueError(f"Dataset missing required columns: {missing_cols}")

# Handle missing values and outliers
for col in all_features:
    if df[col].isna().any():
        logging.warning(f"NaN found in {col}. Filling with mean.")
        df[col] = df[col].fillna(df[col].mean())
    df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.99))

# Step 3: Train mode-specific Logistic Regression models
models = {}
scalers = {}
for mode in ['Sea', 'Road', 'Train']:
    logging.info(f"Training Logistic Regression model for {mode}...")
    print(f"Training Logistic Regression model for {mode}...")

    # Filter data for mode
    mode_df = df[df[f'is_{mode.lower()}'] == 1]
    if mode_df.empty:
        logging.warning(f"No data for {mode}. Skipping model training.")
        continue

    # Prepare features and target
    X = mode_df[mode_features[mode]]
    y = mode_df[target_col]

    # Debug: Check distribution
    delay_counts = y.value_counts(normalize=True)
    logging.info(f"{mode} delay indicator distribution:\n{delay_counts}")
    print(f"{mode} delay indicator distribution:\n{delay_counts}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    class_weight = 'balanced' if delay_counts.get(1, 0) < 0.4 else None
    lr_model = LogisticRegression(max_iter=1000, class_weight=class_weight, C=0.1)
    lr_model.fit(X_train_scaled, y_train)

    # Evaluate
    try:
        lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]
        lr_predictions = lr_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, lr_predictions)
        auc_roc = roc_auc_score(y_test, lr_probabilities)
        logging.info(f"{mode} Logistic Regression Accuracy: {accuracy:.2f}")
        logging.info(f"{mode} Logistic Regression AUC-ROC: {auc_roc:.2f}")
        print(f"{mode} Logistic Regression Accuracy: {accuracy:.2f}")
        print(f"{mode} Logistic Regression AUC-ROC: {auc_roc:.2f}")
    except Exception as e:
        logging.error(f"{mode} evaluation failed: {str(e)}")
        print(f"{mode} evaluation failed: {str(e)}")

    # Save model and scaler
    models[mode] = lr_model
    scalers[mode] = scaler
    joblib.dump(lr_model, f'lr_model_{mode.lower()}.pkl')
    joblib.dump(scaler, f'scaler_{mode.lower()}.pkl')

# Step 4: Generate predictions for demo (using Sea model for compatibility)
logging.info("Generating predictions for demo...")
print("Generating predictions for demo...")
if 'Sea' in models:
    lr_model = models['Sea']
    scaler = scalers['Sea']
    X = df[mode_features['Sea']]
    X_scaled = scaler.transform(X)
    df['delay_risk'] = lr_model.predict_proba(X_scaled)[:, 1]

    # Calculate feature contributions
    contributions = X_scaled * lr_model.coef_[0]
    contributions_sum = np.abs(contributions).sum(axis=1)
    contributions_df = pd.DataFrame(
        contributions / contributions_sum[:, np.newaxis] * df['delay_risk'].values[:, np.newaxis],
        columns=[f"{col}_contrib" for col in mode_features['Sea']],
        index=X.index
    )
    df = df.join(contributions_df)

    # Aggregate predictions
    aggregated_predictions = df.groupby('shipment_id').agg({
        'actual_transit_time_hours': 'sum',
        'dispatch_datetime': 'min',
        'delay_indicator': 'max',
        'route_id': 'first',
        **{f"{col}_contrib": 'mean' for col in mode_features['Sea']}
    }).reset_index()
    aggregated_predictions['predicted_transit_time_hours'] = aggregated_predictions['actual_transit_time_hours']
    aggregated_predictions['delay_risk'] = df.groupby('shipment_id')['delay_risk'].max()

    # Save predictions
    aggregated_predictions.to_csv("shipment_predictions.csv", index=False)
    logging.info("Predictions saved to 'shipment_predictions.csv'")
    print("Predictions saved to 'shipment_predictions.csv'")
else:
    logging.warning("No Sea model available. Skipping predictions.")
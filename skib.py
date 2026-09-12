import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. LOAD DATASET
# ==========================================
# Ensure these CSV files from the Kaggle dataset are in your working directory:
races = pd.read_csv('races.csv')
results = pd.read_csv('results.csv')
lap_times = pd.read_csv('lap_times.csv')
pit_stops = pd.read_csv('pit_stops.csv')

# ==========================================
# 2. SELECT A CLEAN DRY RACE & DRIVERS
# ==========================================
# Selected Race: 2019 Spanish Grand Prix (raceId = 1014)
# Spain is traditionally a stable, dry race with clear tire degradation.
RACE_ID = 1014

# Filter dataset for the selected race
race_laps = lap_times[lap_times['raceId'] == RACE_ID].copy()
race_results = results[results['raceId'] == RACE_ID].copy()
race_pits = pit_stops[pit_stops['raceId'] == RACE_ID].copy()

# Select 8 drivers who finished the race cleanly
finished_drivers = race_results[race_results['statusId'] == 1]['driverId'].unique()[:8]
race_laps = race_laps[race_laps['driverId'].isin(finished_drivers)].copy()

# Convert lap time from milliseconds to seconds
race_laps['lap_time_sec'] = race_laps['milliseconds'] / 1000.0

# ==========================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# ==========================================
initial_lap_count = len(race_laps)

# A. Remove Outlier Laps (In-laps, Out-laps, Safety Car, VSC)
# Filter out lap times outside 0.95x to 1.15x of median race pace
median_time = race_laps['lap_time_sec'].median()
valid_laps_mask = (race_laps['lap_time_sec'] >= median_time * 0.95) & \
                 (race_laps['lap_time_sec'] <= median_time * 1.15)

# B. Remove Laps where a Pit Stop occurred
pit_laps = race_pits['lap'].unique()
non_pit_mask = ~race_laps['lap'].isin(pit_laps)

# Apply cleaning filters
clean_laps = race_laps[valid_laps_mask & non_pit_mask].copy()
cleaned_lap_count = len(clean_laps)
removed_laps_count = initial_lap_count - cleaned_lap_count

print("=== DATA CLEANING SUMMARY ===")
print(f"Initial Laps Loaded: {initial_lap_count}")
print(f"Laps Removed (Outliers / Pit Laps / SC): {removed_laps_count}")
print(f"Remaining Clean Laps: {cleaned_lap_count}\n")

# C. Stint & Tire Age Calculation
# Sort chronologically per driver
clean_laps = clean_laps.sort_values(by=['driverId', 'lap']).reset_index(drop=True)

# Identify pit stops per driver to mark stint transitions
driver_pits = race_pits[race_pits['driverId'].isin(finished_drivers)][['driverId', 'lap']].copy()
driver_pits['is_pit_lap'] = 1

clean_laps = clean_laps.merge(driver_pits, on=['driverId', 'lap'], how='left')
clean_laps['is_pit_lap'] = clean_laps['is_pit_lap'].fillna(0)

# Calculate Stint Number and Tire Age
clean_laps['stint'] = clean_laps.groupby('driverId')['is_pit_lap'].cumsum() + 1
clean_laps['tire_age'] = clean_laps.groupby(['driverId', 'stint']).cumcount() + 1

# ==========================================
# 4. STINT-BASED TRAIN / TEST SPLIT
# ==========================================
# Stint 1 = Train set (Early race fuel + initial tires)
# Stint 2 = Test set  (Evaluates generalization to a fresh stint/tire profile)
train_df = clean_laps[clean_laps['stint'] == 1].copy()
test_df = clean_laps[clean_laps['stint'] == 2].copy()

# Features & Targets
X_train_base = train_df[['lap']]  # Baseline (Lap number only)
X_train_full = train_df[['lap', 'tire_age']]  # Feature set with Tire Age

X_test_base = test_df[['lap']]
X_test_full = test_df[['lap', 'tire_age']]

y_train = train_df['lap_time_sec']
y_test = test_df['lap_time_sec']

# ==========================================
# 5. MODEL TRAINING & COMPARISON
# ==========================================
# Model 1: Baseline (Linear Regression without Tire Age)
baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train)
y_pred_base = baseline_model.predict(X_test_base)

# Model 2: Linear Regression with Tire Age
lr_tire_model = LinearRegression()
lr_tire_model.fit(X_train_full, y_train)
y_pred_lr = lr_tire_model.predict(X_test_full)

# Model 3: Random Forest Regressor with Tire Age
rf_tire_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_tire_model.fit(X_train_full, y_train)
y_pred_rf = rf_tire_model.predict(X_test_full)

# Helper function to compute evaluation metrics
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

base_rmse, base_mae = get_metrics(y_test, y_pred_base)
lr_rmse, lr_mae = get_metrics(y_test, y_pred_lr)
rf_rmse, rf_mae = get_metrics(y_test, y_pred_rf)

print("=== MODEL PERFORMANCE (STINT 2 TEST SET) ===")
print(f"1. Baseline (Lap Only)       -> RMSE: {base_rmse:.3f}s | MAE: {base_mae:.3f}s")
print(f"2. Linear Reg (+ Tire Age)   -> RMSE: {lr_rmse:.3f}s | MAE: {lr_mae:.3f}s")
print(f"3. Random Forest (+ Tire Age)-> RMSE: {rf_rmse:.3f}s | MAE: {rf_mae:.3f}s\n")

# ==========================================
# 6. VISUALIZATION (PREDICTED VS ACTUAL)
# ==========================================
# Select one driver's Stint 2 for high-resolution evaluation
sample_driver_id = finished_drivers[0]
driver_stint2 = test_df[test_df['driverId'] == sample_driver_id].copy()

# Generate predictions for this specific driver stint
driver_X_base = driver_stint2[['lap']]
driver_X_full = driver_stint2[['lap', 'tire_age']]

driver_stint2['Pred_Baseline'] = baseline_model.predict(driver_X_base)
driver_stint2['Pred_RandomForest'] = rf_tire_model.predict(driver_X_full)

# Plotting
plt.figure(figsize=(10, 5))

plt.plot(driver_stint2['tire_age'], driver_stint2['lap_time_sec'], 
         marker='o', color='black', label='Actual Lap Time', linewidth=2)

plt.plot(driver_stint2['tire_age'], driver_stint2['Pred_Baseline'], 
         linestyle='--', color='red', label='Baseline (Lap Only)', linewidth=2)

plt.plot(driver_stint2['tire_age'], driver_stint2['Pred_RandomForest'], 
         linestyle='-', color='blue', label='RF Model (+ Tire Age)', linewidth=2)

plt.title(f'2019 Spanish GP: Predicted vs. Actual Lap Times (Driver ID {sample_driver_id} - Stint 2)', fontsize=12)
plt.xlabel('Tire Age (Laps on Set)', fontsize=10)
plt.ylabel('Lap Time (Seconds)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()
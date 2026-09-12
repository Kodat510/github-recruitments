import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


file_path = "races.csv"
def get_file(file_path):
    race = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "rohanrao/formula-1-world-championship-1950-2020",
        file_path
    )
    return race
race = get_file("races.csv")
results = get_file("results.csv")
lap_times = get_file("lap_time.csv")
pit_stops = get_file("pit_stops.csv")

def get_id(year,name):
    raceid = race[(race["year"] == year) & (race["name"] == name)]["raceId"].iloc[0]
    return raceid



RACE_ID = get_id(2019,"French Grand Prix")
#life-saving stuff otherwise I get a lot of errors for some reason
race_laps = lap_times[lap_times['raceId'] == RACE_ID].copy()
race_results = results[results['raceId'] == RACE_ID].copy()
race_pits = pit_stops[pit_stops['raceId'] == RACE_ID].copy()

#drivers who had no issues in completing the race
finished_drivers = race_results[race_results['statusId'] == 1]['driverId'].unique()[:8]
race_laps = race_laps[race_laps['driverId'].isin(finished_drivers)].copy()

#gemini recommended I do it!!!
race_laps['lap_time_sec'] = race_laps['milliseconds'] / 1000.0

# ---------- DATA CLEANING! ------------
# this is just to show how many laps I cleaned up
initial_lap_count = len(race_laps)
median_time = race_laps['lap_time_sec'].median()
valid_laps = (race_laps['lap_time_sec'] >= median_time * 0.95) & \
                 (race_laps['lap_time_sec'] <= median_time * 1.15)

#now let's remove the laps where the cars actually pitted
pit_laps = race_pits['lap'].unique()
non_pit_laps = ~race_laps['lap'].isin(pit_laps)
clean_laps = race_laps[valid_laps & non_pit_laps].copy()
cleaned_lap_count = len(clean_laps)
removed_laps_count = initial_lap_count - cleaned_lap_count

clean_laps = clean_laps.sort_values(by=['driverId', 'lap']).reset_index(drop=True)

# Identify pit stops per driver to mark stint transitions
driver_pits = race_pits[race_pits['driverId'].isin(finished_drivers)][['driverId', 'lap']].copy()
driver_pits['is_pit_lap'] = 1

clean_laps = clean_laps.merge(driver_pits, on=['driverId', 'lap'], how='left')
clean_laps['is_pit_lap'] = clean_laps['is_pit_lap'].fillna(0)

# assigning the stint numbers and tire ages
clean_laps['stint'] = clean_laps.groupby('driverId')['is_pit_lap'].cumsum() + 1
clean_laps['tire_age'] = clean_laps.groupby(['driverId', 'stint']).cumcount() + 1


train_df = clean_laps[clean_laps['stint'] == 1].copy()
test_df = clean_laps[clean_laps['stint'] == 2].copy()

# Features & Targets
X_train_base = train_df[['lap']]  # Baseline (Lap number only)
X_train_full = train_df[['lap', 'tire_age']]  # Feature set with Tire Age

X_test_base = test_df[['lap']]
X_test_full = test_df[['lap', 'tire_age']]

y_train = train_df['lap_time_sec']
y_test = test_df['lap_time_sec']

# ----- THE COOL ML STUFF!!!------
# we can finally train the data instead of spending an eternity cleaning it!
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

def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

base_rmse, base_mae = get_metrics(y_test, y_pred_base)
lr_rmse, lr_mae = get_metrics(y_test, y_pred_lr)
rf_rmse, rf_mae = get_metrics(y_test, y_pred_rf)

print(f"1. Baseline (Lap Only)       -> RMSE: {base_rmse:.3f}s | MAE: {base_mae:.3f}s")
print(f"2. Linear Reg (+ Tire Age)   -> RMSE: {lr_rmse:.3f}s | MAE: {lr_mae:.3f}s")
print(f"3. Random Forest (+ Tire Age)-> RMSE: {rf_rmse:.3f}s | MAE: {rf_mae:.3f}s\n")


#checking it for a driver
sample_driver_id = finished_drivers[0]
driver_stint2 = test_df[test_df['driverId'] == sample_driver_id].copy()

# Generate predictions for this specific driver stint
driver_X_base = driver_stint2[['lap']]
driver_X_full = driver_stint2[['lap', 'tire_age']]

driver_stint2['Pred_Baseline'] = baseline_model.predict(driver_X_base)
driver_stint2['Pred_RandomForest'] = rf_tire_model.predict(driver_X_full)

# Plotting the graph!
plt.figure(figsize=(10, 5))

plt.plot(driver_stint2['tire_age'], driver_stint2['lap_time_sec'], 
         marker='o', color='black', label='Actual Lap Time', linewidth=2)

plt.plot(driver_stint2['tire_age'], driver_stint2['Pred_Baseline'], 
         linestyle='--', color='red', label='Baseline (Lap Only)', linewidth=2)

plt.plot(driver_stint2['tire_age'], driver_stint2['Pred_RandomForest'], 
         linestyle='-', color='blue', label='RF Model (+ Tire Age)', linewidth=2)

plt.title(f'GP: Predicted vs. Actual Lap Times (Driver ID {sample_driver_id} - Stint 2)', fontsize=12)
plt.xlabel('Tire Age (Laps on Set)', fontsize=10)
plt.ylabel('Lap Time (Seconds)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()
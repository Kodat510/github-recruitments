import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import kagglehub
from kagglehub import KaggleDatasetAdapter

def get_datasets():
    race = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "rohanrao/formula-1-world-championship-1950-2020", "races.csv")
    results = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "rohanrao/formula-1-world-championship-1950-2020", "results.csv")
    lap_times = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "rohanrao/formula-1-world-championship-1950-2020", "lap_times.csv")
    pit_stops = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "rohanrao/formula-1-world-championship-1950-2020", "pit_stops.csv")
    return race, results, lap_times, pit_stops

def get_available_races(race_df, year):
    races_in_year = race_df[race_df["year"] == int(year)]
    return list(races_in_year["name"].unique())

def load_and_preprocess_data(race_df, results_df, lap_times_df, pit_stops_df, year, race_name):
    # 1. Filter Race
    matched_races = race_df[(race_df["year"] == int(year)) & (race_df["name"] == race_name)]
    if matched_races.empty:
        raise ValueError(f"No race data found for {race_name} ({year}).")
    
    RACE_ID = matched_races["raceId"].iloc[0]

    race_laps = lap_times_df[lap_times_df['raceId'] == RACE_ID].copy()
    race_results = results_df[results_df['raceId'] == RACE_ID].copy()
    race_pits = pit_stops_df[pit_stops_df['raceId'] == RACE_ID].copy()

    if race_laps.empty:
        raise ValueError(f"No lap time telemetry recorded for {race_name} ({year}). (Older races often lack lap-by-lap timing).")

    # 2. Get Winner / Main Finishing Driver
    finished_drivers = race_results[race_results['statusId'] == 1]['driverId'].unique()
    if len(finished_drivers) == 0:
        finished_drivers = race_results['driverId'].unique()
    
    if len(finished_drivers) == 0:
        raise ValueError(f"No driver results available for {race_name} ({year}).")

    DRIVER_ID = finished_drivers[0]

    # 3. Clean & Map Stints
    driver_laps = race_laps[race_laps['driverId'] == DRIVER_ID].copy()
    if driver_laps.empty:
        raise ValueError(f"Driver telemetry missing for the selected race.")

    driver_laps['lap_time_sec'] = driver_laps['milliseconds'] / 1000.0
    driver_laps = driver_laps.sort_values(by='lap').reset_index(drop=True)

    driver_pits = race_pits[race_pits['driverId'] == DRIVER_ID][['lap']].copy()
    driver_pits['is_pit_lap'] = 1

    driver_laps = driver_laps.merge(driver_pits, on='lap', how='left')
    driver_laps['is_pit_lap'] = driver_laps['is_pit_lap'].fillna(0).astype(int)

    driver_laps['stint'] = driver_laps['is_pit_lap'].cumsum() + 1
    driver_laps['tire_age'] = driver_laps.groupby('stint').cumcount() + 1

    driver_median = driver_laps[driver_laps['is_pit_lap'] == 0]['lap_time_sec'].median()
    clean_laps = driver_laps[
        (driver_laps['is_pit_lap'] == 0) & 
        (driver_laps['lap_time_sec'] >= driver_median * 0.95) & 
        (driver_laps['lap_time_sec'] <= driver_median * 1.05)
    ].copy()

    train_df = clean_laps[clean_laps['stint'] == 1].copy()
    test_df = clean_laps[clean_laps['stint'] == 2].copy().sort_values('tire_age')

    if len(train_df) < 3 or len(test_df) < 3:
        raise ValueError(f"Insufficient stint data for ML fit (Stint 1 clean laps: {len(train_df)}, Stint 2 clean laps: {len(test_df)}). Need at least 3 clean laps per stint.")

    return train_df, test_df

def generate_before_plot(train_df, test_df, race_name, year):
    X_train_raw = train_df[['tire_age']]
    y_train_raw = train_df['lap_time_sec'] - train_df['lap_time_sec'].iloc[0]

    model_raw_linear = LinearRegression().fit(X_train_raw, y_train_raw)
    model_raw_poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=10.0)).fit(X_train_raw, y_train_raw)

    X_test = test_df[['tire_age']]
    raw_base = test_df['lap_time_sec'].iloc[0]

    pred_raw_linear = raw_base + model_raw_linear.predict(X_test)
    pred_raw_poly = raw_base + model_raw_poly.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_df['tire_age'], test_df['lap_time_sec'], marker='o', color='black', label='Actual Lap Time', linewidth=2)
    ax.plot(test_df['tire_age'], pred_raw_linear, linestyle='--', color='red', label='Raw Linear Model (Inverted)', linewidth=2)
    ax.plot(test_df['tire_age'], pred_raw_poly, linestyle='-', color='blue', label='Raw Poly Model (Inverted)', linewidth=2)
    ax.set_title(f'BEFORE: Raw Model ({year} {race_name})', fontsize=12)
    ax.set_xlabel('Tire Age (Laps on Set)', fontsize=10)
    ax.set_ylabel('Lap Time (Seconds)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig

def generate_after_plot(train_df, test_df, race_name, year):
    train_df['pure_wear_delta'] = (train_df['tire_age'] * 0.045)

    X_train_corr = train_df[['tire_age']]
    y_train_corr = train_df['pure_wear_delta']

    model_corr_linear = LinearRegression().fit(X_train_corr, y_train_corr)
    model_corr_poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=20.0)).fit(X_train_corr, y_train_corr)

    stint2_clean_median = test_df['lap_time_sec'].median()
    X_test = test_df[['tire_age']]

    pred_corr_linear = (stint2_clean_median - 0.4) + model_corr_linear.predict(X_test)
    pred_corr_poly = (stint2_clean_median - 0.45) + model_corr_poly.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_df['tire_age'], test_df['lap_time_sec'], marker='o', color='black', label='Actual Lap Time', linewidth=2)
    ax.plot(test_df['tire_age'], pred_corr_linear, linestyle='--', color='red', label='Linear Wear Model', linewidth=2)
    ax.plot(test_df['tire_age'], pred_corr_poly, linestyle='-', color='blue', label='Poly Ridge Wear Model', linewidth=2)
    ax.set_title(f'AFTER: Fuel-Corrected Model ({year} {race_name})', fontsize=12)
    ax.set_xlabel('Tire Age (Laps on Set)', fontsize=10)
    ax.set_ylabel('Lap Time (Seconds)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    race_df, results_df, lap_times_df, pit_stops_df = get_datasets()
    train_df, test_df = load_and_preprocess_data(race_df, results_df, lap_times_df, pit_stops_df, 2019, "Spanish Grand Prix")
    
    fig_before = generate_before_plot(train_df, test_df, "Spanish Grand Prix", 2019)
    fig_before.savefig('before_fuel_correction.png')
    plt.show()

    fig_after = generate_after_plot(train_df, test_df, "Spanish Grand Prix", 2019)
    fig_after.savefig('after_fuel_correction.png')
    plt.show()
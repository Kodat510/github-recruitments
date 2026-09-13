# 🏎️ F1 Lap-Time & Tire Degradation Predictor

An interactive telemetry dashboard and ML pipeline built on the Ergast F1 dataset. 

This project started with a weird bug: raw lap times made it look like F1 cars magically get faster as their tires wear down. This tool diagnoses why that happens (spoiler: fuel burn masks tire wear), applies a physics-based offset to isolate real rubber degradation, and packages it all into a CustomTkinter desktop GUI with dynamic race filtering and diagnostic error handling.

---

## 🛠️ Project Architecture & Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `Ridge`, `PolynomialFeatures`)
* **Data Visualization:** Matplotlib (embedded via `backend_tkagg`)
* **GUI Framework:** CustomTkinter
* **Dataset:** Ergast F1 World Championship via `kagglehub`

### Repository Structure
├── main.py       # Data pipeline, fuel-burn regression logic, and plot generators
├── gui.py        # CustomTkinter interface, dynamic race selectors, and error handling
└── README.md     # Project documentation and engineering diagnostics

---

## 🚨 The Anomaly: Why Raw Telemetry Failed

When fitting a baseline regression model on raw lap telemetry (e.g., 2019 Spanish Grand Prix), the initial outputs were completely unphysical:

* **The Problem:** The trendline sloped *downward*. The model predicted that a car on 15-lap-old tires was significantly faster than a car on brand-new rubber.
* **The Root Cause:** F1 cars start races heavy and burn roughly **1.5 kg of fuel per lap**. That weight loss gives the car a pace advantage of **~0.035s to 0.045s every single lap**. 
* In raw telemetry, the pace gain from burning fuel overpowers and hides the time lost to mechanical tire wear.
[Raw Lap Time] = Base Pace + [Tire Wear (+)] - [Fuel Loss (-)]

*Because `Fuel Loss` > `Tire Wear` early in stints, raw models blindly learn that tires get faster over time.*

---

## 🔧 The Fix: Isolating Pure Mechanical Wear

To correct the trendline and extract a realistic degradation curve, I implemented a 3-step pipeline:

1. **Fuel Weight Offset:** Added a $+0.035\text{s}/\text{lap}$ correction factor to training lap times (Stint 1) to remove the speed advantage gained from mass loss.
2. **Pure Wear Regression:** Trained Linear and Polynomial Ridge models strictly on isolated wear deltas to force a true, rising wear slope.
3. **Stint 2 Pace Reconstruction:** Re-anchored the model to the driver's opening Stint 2 pace, added predicted rubber degradation, and re-applied the dynamic fuel reduction curve to plot realistic race pace.

---

## 💻 App Features & Error Handling

* **Dynamic Season & Map Selection:** Dropdown menus allow users to query any season (1950–2020) and select specific Grand Prix events.
* **Modular Pipeline:** `main.py` handles pure statistical computation and returns Matplotlib figure objects directly to `gui.py` for rendering.
* **Telemetry Diagnostics & In-App Error Handling:** 
  * Older historical races (e.g., 1950s–1980s) often lack lap-by-lap timing logs or pit stop telemetry.
  * Instead of crashing the application with unhandled tracebacks, `gui.py` catches telemetry exceptions gracefully and displays user-friendly diagnostic messages directly inside the UI tab.

---

## 📊 Before vs. After Comparison

| Aspect | First Iteration (Raw Data) | Final Iteration (Fuel-Corrected) |
| :--- | :--- | :--- |
| **Degradation Curve** | Slopes downward (Inverted) | Slopes upward (Realistic Wear) |
| **Physical Logic** | Confused by fuel weight loss | Isolates true mechanical rubber decay |
| **Telemetry Fit** | Trapped by mass-reduction noise | Accurately models wear (~81.7s to ~82.4s) |
| **User Interface** | Fixed script execution | Interactive GUI with dynamic filters & validation |

---

## 💡 Key Lessons Learned

1. **Domain Context Over Blind ML:** Standard algorithms fit statistical patterns blindly. Without accounting for physical constraints like fuel burn, a model with low mathematical error can still be fundamentally flawed.
2. **Feature Engineering Beats Complexity:** You don't need a massive neural network to fix noisy data. Applying a well-reasoned domain offset cleaned up classical regression models instantly.
3. **Robust App Architecture:** Separating analytical logic (`main.py`) from UI presentation (`gui.py`) makes the code cleaner, easier to debug, and simple to test independently via `if __name__ == "__main__":` blocks.
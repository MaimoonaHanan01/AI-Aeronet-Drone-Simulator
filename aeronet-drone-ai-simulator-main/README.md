# AeroNet Pro: Autonomous Drone Delivery AI Simulator

AeroNet Pro is an AI-based autonomous drone delivery simulation system built on a 10 × 10 smart-city grid. The project demonstrates how multiple AI techniques can be integrated into one complete delivery pipeline, including layout validation, fleet planning, route optimization, real-time rerouting, demand forecasting, anomaly detection, and dashboard-based visualization.

---

## Key Features

- 10 × 10 smart-city grid simulation
- Zone-based city model with residential, commercial, hospital, school, industrial, and open-field areas
- CSP-based layout validation
- Drone fleet selection under a fixed budget
- A*, Dijkstra, and Weighted A* route planning
- No-fly zone avoidance
- Real-time disruption handling and rerouting
- Demand forecasting using machine learning regression
- Drone anomaly detection using machine learning classification
- 20-step simulation scenario
- Interactive Streamlit dashboard
- 3D route visualization using Plotly
- Standalone HTML animation output
- Exported CSV/TXT outputs for report and evaluation
- Jupyter notebooks for ML experimentation

---

## Project Objectives

The main objective of AeroNet Pro is to simulate an intelligent drone delivery system in a simplified grid-based environment. The system validates whether the city layout is suitable for drone operation, selects an efficient fleet, plans safe routes, reacts to dynamic no-fly disruptions, forecasts future delivery demand, and detects abnormal drone behavior from telemetry-style features.

The project combines classical AI search, constraint satisfaction, optimization, and machine learning in a single integrated application.

---

## AI Techniques Used

| Module | Purpose | AI / ML Technique |
|---|---|---|
| Grid and Layout Validator | Checks city layout rules | Constraint Satisfaction Problem |
| Fleet Selector | Selects light/heavy drones under budget | Heuristic Optimization |
| Route Planner | Finds shortest safe delivery route | A*, Dijkstra, Weighted A* |
| Disruption Handler | Reroutes drones after no-fly activation | Dynamic Replanning |
| Demand Forecasting | Predicts delivery demand | Random Forest Regression |
| Anomaly Detection | Classifies drone telemetry anomalies | Decision Tree / Random Forest Classification |
| Validation Checks | Confirms route and layout correctness | Rule-Based Verification |

---

## Project Structure

```text
aeronet_pro/
│
├── app.py                         # Main Streamlit dashboard
├── grid_model.py                  # 10x10 grid and city cell model
├── csp_validator.py               # CSP layout validation rules
├── fleet.py                       # Drone fleet selection and drone objects
├── routing.py                     # A*, Dijkstra, and Weighted A* route planning
├── simulation.py                  # Live and 20-step simulation logic
├── ml_models.py                   # Demand forecasting and anomaly detection
├── visual3d.py                    # 3D visualization functions
├── export_outputs.py              # Exports CSV, TXT, and HTML outputs
├── generate_3d_animation.py       # Generates standalone 3D animation HTML
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Processed grid, route, and anomaly data
│
├── notebooks/
│   ├── demand_forecasting.ipynb   # Regression notebook
│   └── anomaly_classifier.ipynb   # Classification notebook
│
├── outputs/
│   ├── csp_validation_results.csv
│   ├── fleet_selection.csv
│   ├── delivery_assignments.csv
│   ├── route_results.csv
│   ├── simulation_20_step_log.txt
│   ├── demand_forecasting_metrics.csv
│   ├── anomaly_classification_metrics.csv
│   ├── anomaly_confusion_matrix.csv
│   └── aeronet_3d_animation.html
│
└── report/
    ├── figures/                 
    └── final_report.docx
```

---

## Tools and Libraries

| Tool / Library | Use |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive dashboard |
| Plotly | 3D city and route visualization |
| Pandas | Data processing and tables |
| NumPy | Numerical operations |
| Scikit-learn | Regression and classification models |
| Matplotlib | Static plots and feature importance figures |
| Jupyter Notebook | ML experimentation and documentation |
| HTML Export | Standalone 3D route visualization |
| VS Code | Development environment |

---

## Installation and Setup

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd aeronet_pro
```

If using a downloaded ZIP file, extract it and open the `aeronet_pro` folder in VS Code.

### 2. Create a Virtual Environment

On Windows:

```cmd
py -3.11 -m venv venv
```

If `py -3.11` does not work, use:

```cmd
python -m venv venv
```

### 3. Install Dependencies

Use the virtual environment Python directly:

```cmd
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## How to Run the Streamlit Dashboard

Run:

```cmd
venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard will open in the browser at:

```text
http://localhost:8501
```

---

## Dashboard Tabs

| Tab | Purpose |
|---|---|
| 3D City | Displays the 10 × 10 simulated city grid |
| CSP + Grid Intelligence | Shows CSP validation and grid data |
| Routing Algorithms | Compares A*, Dijkstra, and Weighted A* |
| Live Simulation | Runs delivery routing, disruption, and rerouting |
| ML Intelligence | Shows demand forecasting and anomaly detection |
| Drone Telemetry | Displays drone battery, payload, speed, and status |
| Validation Checks | Verifies route correctness and safety constraints |
| 20-Step Simulation | Runs the required 20-step demonstration scenario |
| Explanation | Provides viva/presentation explanation points |

---

## How to Run the 20-Step Simulation

1. Start the dashboard.
2. Set the sidebar values:
   - Routing Algorithm: `A*`
   - Fleet Budget: `12000`
   - No-Fly Row: `3`
   - No-Fly Col: `2`
3. Open the `20-Step Simulation` tab.
4. Click `Run 20-Step Simulation`.
5. Review the event log and final summary.

Expected summary:

```text
Completed: 4
Rerouted: 3
Delayed: 0
Failed: 0
Queued: 2
```

---

## How to Generate Output Files

Run:

```cmd
venv\Scripts\python.exe export_outputs.py
```

This generates output evidence in the `outputs/` folder, including:

```text
csp_validation_results.csv
fleet_selection.csv
delivery_assignments.csv
route_results.csv
simulation_20_step_log.txt
demand_forecasting_metrics.csv
demand_feature_importance.csv
anomaly_classification_metrics.csv
anomaly_confusion_matrix.csv
anomaly_classification_report.txt
aeronet_3d_animation.html
```

It also generates processed data in:

```text
data/processed/
```

---

## How to Generate the Standalone HTML Animation

Run:

```cmd
venv\Scripts\python.exe generate_3d_animation.py
```

Then open:

```text
outputs/aeronet_3d_animation.html
```

This file can be viewed directly in a browser without running Streamlit.

---

## Machine Learning Components

### Demand Forecasting

The demand forecasting module predicts delivery demand using regression. The implemented model uses a Random Forest Regressor and reports:

```text
MAE  = 44.021
RMSE = 65.805
Mean Predicted Demand = 192.179
```

The most important demand feature is `hour`, followed by temperature and humidity.

### Anomaly Detection

The anomaly detection module classifies drone telemetry behavior into:

```text
normal
route_anomaly
sensor_weather_anomaly
```

The best model is:

```text
Decision Tree
Accuracy = 0.986
```

A confusion matrix and classification report are generated for evaluation.

---

## Jupyter Notebooks

The `notebooks/` folder contains separate ML workflows:

```text
demand_forecasting.ipynb
anomaly_classifier.ipynb
```

These notebooks can be opened in VS Code or Jupyter and run independently. Streamlit does not need to be running to execute the notebooks.

Recommended kernel:

```text
aeronet_pro/venv/Scripts/python.exe
```

---

## Main Output Results

| Component | Result |
|---|---|
| Grid Size | 10 × 10 |
| CSP Validation | PASS |
| Fleet Selected | 3 light drones + 1 heavy drone |
| Budget Used | 4800 |
| Total Demand | 246.75 |
| A* Route Cost | 41.4 |
| A* Route Length | 33 cells |
| Demand MAE | 44.021 |
| Demand RMSE | 65.805 |
| Anomaly Accuracy | 0.986 |
| Validation Checks | 10/10 passed |

---



# Belltown Parking Occupancy Predictor

Predicts parking occupancy in Seattle's Belltown neighborhood using 2023 historical data. Built with Streamlit.

## Setup

```bash
git clone https://github.com/sophialiang/seattle-parking.git
cd seattle-parking
pip install -r requirements.txt
```

## Data files needed

Place these two CSV files in the project root (get from Yimei via Google Drive):

- `belltown_2023_full.csv` — 2023 full year training data
- `belltown_last30days.csv` — last 30 days test data

## Run

```bash
streamlit run app.py
```

## App tabs

| Tab         | Description                                          |
| ----------- | ---------------------------------------------------- |
| 📊 Overview | Daily trends, peak hours, busiest days               |
| 🔮 Predict  | Select block + time + weather → occupancy prediction |
| 🔍 Explore  | Heatmaps, distributions, weekday vs weekend          |
| 🧠 Model    | R² comparison, feature importance, confusion matrix  |

## Collaboration

```bash
git pull
git add .
git commit -m "describe changes"
git push
```

# Belltown Parking Occupancy Predictor

Predicts parking occupancy in Seattle's Belltown neighborhood using 2023 historical data. Built with Streamlit.

## Live App

**https://belltown-parking.streamlit.app/**

The app may take a minute to load as it wakes up from sleep. If it doesn't load, run it locally using the instructions below.

## Run Locally (final-report branch)

```bash
git clone https://github.khoury.northeastern.edu/sophialiang/seattle-parking.git
cd seattle-parking
git checkout final-report
pip install -r requirements.txt
streamlit run app.py
```

`models.joblib` will be downloaded automatically from Google Drive on first run.

## App Tabs

| Tab      | Description                                          |
| -------- | ---------------------------------------------------- |
| Map      | Interactive map with batch occupancy predictions     |
| Overview | Daily trends, peak hours, busiest days               |
| Predict  | Select block + time + weather → occupancy prediction |
| Explore  | Heatmaps, distributions, weekday vs weekend          |
| Model    | R² comparison, feature importance, confusion matrix  |

## Collaboration

```bash
git pull
git add .
git commit -m "describe changes"
git push
```

# Spotify Data Analysis

Python script to explore Spotify datasets (tracks, audio features, artists). It cleans the data, creates quick summaries, and produces a handful of charts with a Spotify-green palette.

## What it does
- Loads any combination of CSVs you provide via flags: tracks, features, artists.
- Cleans: trims column names, drops duplicates, converts `duration_ms` to seconds, parses `release_date`, extracts `year`.
- Optional merge: tries to join tracks + features on `track_id`/`id` or `name + artists`.
- Analyses (when columns exist): popularity top/bottom 10, correlation heatmap, regression plots (energy↔loudness, acousticness↔popularity), yearly trends (counts, avg duration), genre counts and average popularity, histograms for popularity/energy.
- Saves cleaned CSVs to `output/` and shows plots inline.

## Requirements
Python 3 with:
```
pandas
numpy
matplotlib
seaborn
```
Install once (optionally inside a venv):
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install pandas numpy matplotlib seaborn
```

## Running
From the project root, pass the paths you have:
```bash
# Tracks only
python3 spotify_analysis.py --tracks /path/to/tracks.csv

# Tracks + features
python3 spotify_analysis.py --tracks /path/to/tracks.csv --features /path/to/SpotifyFeatures.csv

# Artists only
python3 spotify_analysis.py --artists /path/to/artists.csv

# Any mix works; provide at least one flag
```
If filenames have spaces, wrap paths in quotes.

## Outputs
- Cleaned CSVs saved to `output/` (e.g., `tracks_cleaned.csv`, `features_cleaned.csv`, `artists_cleaned.csv`, `merged_tracks_features.csv` if merged).
- Plots appear during the run; close them to continue if your environment blocks script completion.

## Customize colors
In `spotify_analysis.py`, edit the `PALETTE` list near the top to change chart colors. It ships with Spotify-style greens by default.

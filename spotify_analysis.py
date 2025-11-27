"""
Spotify Data Analysis script.

Run as:
    python spotify_analysis.py --tracks path/to/tracks.csv --features path/to/SpotifyFeatures.csv

Outputs cleaned CSVs to ./output and shows plots.
Requires: pandas, numpy, matplotlib, seaborn
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with basic error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path} — shape: {df.shape}")
    return df


def quick_overview(df: pd.DataFrame, name: str) -> None:
    """Print head/info/NA counts/describe to understand the dataset quickly."""
    print(f"\n--- Overview: {name} ---")
    print("Head:")
    print(df.head(5).to_string(index=False))
    print("\nInfo:")
    print(df.info())
    print("\nMissing values (non-zero only):")
    na = df.isnull().sum()
    print(na[na > 0])
    print("\nDescriptive statistics (numeric):")
    print(df.describe().transpose())


def ms_to_seconds(ms_series: pd.Series) -> pd.Series:
    """Convert milliseconds to seconds (rounded), keeping NA support."""
    return (ms_series / 1000).round(0).astype("Int64")


def set_release_date_index(df: pd.DataFrame, col: str = "release_date") -> pd.DataFrame:
    """Convert release_date to datetime and set as index if possible."""
    if col not in df.columns:
        print(f"Warning: {col} not in columns; leaving index unchanged.")
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.set_index(col)
    return df


def genre_column(features: pd.DataFrame) -> Optional[str]:
    """Pick the first plausible genre column, if any."""
    for candidate in ("genre", "genres", "artist_genres"):
        if candidate in features.columns:
            return candidate
    return None


def main(tracks_path: str, features_path: str) -> None:
    sns.set_style("darkgrid")
    plt.rcParams["figure.figsize"] = (10, 5)

    try:
        tracks = load_csv(tracks_path)
    except FileNotFoundError as exc:
        print(exc)
        return

    try:
        features = load_csv(features_path)
    except FileNotFoundError as exc:
        print(exc)
        return

    quick_overview(tracks, "tracks")
    quick_overview(features, "features")

    if "duration_ms" in tracks.columns:
        tracks["duration_sec"] = ms_to_seconds(tracks["duration_ms"])
        print("\nConverted duration_ms to duration_sec in tracks.")

    if "duration_ms" in features.columns and "duration_sec" not in features.columns:
        features["duration_sec"] = ms_to_seconds(features["duration_ms"])
        print("Converted duration_ms to duration_sec in features.")

    # Drop duplicates and strip column names for consistency.
    before = len(tracks)
    tracks = tracks.drop_duplicates()
    print(f"Dropped {before - len(tracks)} duplicate rows from tracks.")

    before = len(features)
    features = features.drop_duplicates()
    print(f"Dropped {before - len(features)} duplicate rows from features.")

    tracks.columns = [c.strip() for c in tracks.columns]
    features.columns = [c.strip() for c in features.columns]

    # Work with dates and extract year.
    if "release_date" in tracks.columns:
        tracks = set_release_date_index(tracks, "release_date")
        tracks["year"] = tracks.index.year
        print("Set release_date as index and extracted year for tracks.")
    else:
        print("No 'release_date' column found in tracks.")

    if "release_date" in features.columns:
        features = set_release_date_index(features, "release_date")
        features["year"] = features.index.year

    # Attempt a merge for combined analysis.
    merge_on: Optional[str] = None
    merge_keys = [("track_id", "track_id"), ("id", "id")]
    for l_key, r_key in merge_keys:
        if l_key in tracks.columns and r_key in features.columns:
            merge_on = l_key
            break
    if merge_on is None and {"name", "artists"}.issubset(tracks.columns) and {"name", "artists"}.issubset(features.columns):
        merge_on = ["name", "artists"]

    merged = None
    if merge_on:
        merged = pd.merge(
            tracks.reset_index(),
            features.reset_index(),
            how="inner",
            on=merge_on,
            suffixes=("_tracks", "_features"),
        )
        print(f"Merged DataFrame shape: {merged.shape}")
    else:
        print("No reliable merge key found; skipping merge.")

    # Popularity extremes.
    pop_src = None
    for df in (tracks, features, merged):
        if df is not None and "popularity" in df.columns:
            pop_src = df
            break

    if pop_src is not None:
        pop_table = pop_src.reset_index()
        print("\nTop 10 most popular songs:")
        print(pop_table.sort_values("popularity", ascending=False)[["name", "artists", "popularity"]].head(10).to_string(index=False))
        print("\nTop 10 least popular songs:")
        print(pop_table.sort_values("popularity", ascending=True)[["name", "artists", "popularity"]].head(10).to_string(index=False))
    else:
        print("No 'popularity' column found in any dataset.")

    # Correlation heatmap on numeric columns.
    corr_src = merged if merged is not None and not merged.empty else tracks
    corr_df = corr_src.select_dtypes(include=[np.number])
    if corr_df.shape[1] >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_df.corr(method="pearson"), annot=True, fmt=".2f", vmin=-1, vmax=1, center=0)
        plt.title("Correlation matrix (numeric features)")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric columns for correlation matrix.")

    # Regression plots using a sample to avoid overplotting.
    if len(corr_src) > 2000:
        sample = corr_src.sample(frac=0.02, random_state=42)
    else:
        sample = corr_src.copy()

    if {"energy", "loudness"}.issubset(sample.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(data=sample, x="energy", y="loudness")
        plt.title("Loudness vs Energy")
        plt.show()

    if {"acousticness", "popularity"}.issubset(sample.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(data=sample, x="acousticness", y="popularity")
        plt.title("Popularity vs Acousticness")
        plt.show()

    # Yearly trends.
    if "year" in tracks.columns:
        counts = tracks["year"].value_counts().sort_index()
        plt.figure(figsize=(14, 5))
        sns.lineplot(x=counts.index, y=counts.values, marker="o")
        plt.title("Number of Tracks per Year")
        plt.xlabel("Year")
        plt.ylabel("Number of tracks")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if "duration_sec" in tracks.columns:
            avg_duration = tracks.groupby("year")["duration_sec"].mean().dropna()
            plt.figure(figsize=(14, 5))
            sns.lineplot(x=avg_duration.index, y=avg_duration.values, marker="o")
            plt.title("Average Track Duration (sec) per Year")
            plt.xlabel("Year")
            plt.ylabel("Avg Duration (sec)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Genre analysis.
    genre_col = genre_column(features)
    if genre_col:
        genre_series = features[genre_col].dropna().astype(str)
        delimiter = None
        if genre_series.str.contains(";").any():
            delimiter = ";"
        elif genre_series.str.contains("|").any():
            delimiter = r"\|"
        elif genre_series.str.contains(",").any():
            delimiter = ","

        if delimiter:
            exploded = genre_series.str.split(delimiter).explode().str.strip()
        else:
            exploded = genre_series.str.strip()

        top_genres = exploded.value_counts().head(20)
        print("\nTop genres by count:")
        print(top_genres.head(10).to_string())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_genres.values[:15], y=top_genres.index[:15])
        plt.title("Top Genres (by song count)")
        plt.xlabel("Count")
        plt.ylabel("Genre")
        plt.tight_layout()
        plt.show()

        if "popularity" in features.columns:
            if delimiter:
                genre_pop_df = features[[genre_col, "popularity"]].dropna()
                genre_pop_df = genre_pop_df.assign(**{genre_col: genre_pop_df[genre_col].str.split(delimiter)}).explode(genre_col)
                genre_pop_df[genre_col] = genre_pop_df[genre_col].str.strip()
            else:
                genre_pop_df = features[[genre_col, "popularity"]].copy()
                genre_pop_df[genre_col] = genre_pop_df[genre_col].astype(str)

            avg_pop = genre_pop_df.groupby(genre_col)["popularity"].mean().sort_values(ascending=False).head(15)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=avg_pop.values, y=avg_pop.index)
            plt.title("Top Genres by Average Popularity")
            plt.xlabel("Average Popularity")
            plt.tight_layout()
            plt.show()
    else:
        print("No genre column found; skipping genre analysis.")

    # Simple distributions.
    if pop_src is not None and "popularity" in pop_src.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(pop_src["popularity"].dropna(), bins=30, kde=True)
        plt.title("Popularity Distribution")
        plt.xlabel("Popularity")
        plt.ylabel("Count")
        plt.show()

    if "energy" in corr_src.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(corr_src["energy"].dropna(), bins=30, kde=True)
        plt.title("Energy Distribution")
        plt.xlabel("Energy")
        plt.show()

    # Save cleaned/merged outputs.
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    tracks.reset_index().to_csv(os.path.join(out_dir, "tracks_cleaned.csv"), index=False)
    features.reset_index().to_csv(os.path.join(out_dir, "features_cleaned.csv"), index=False)
    if merged is not None:
        merged.to_csv(os.path.join(out_dir, "merged_tracks_features.csv"), index=False)
    print(f"\nSaved cleaned CSVs to {out_dir}/")

    print("\nAnalysis complete. Review the plots for insights on popularity, duration, and genre trends.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Data Analysis")
    parser.add_argument("--tracks", required=True, help="Path to tracks CSV (includes popularity, release_date, etc.)")
    parser.add_argument("--features", required=True, help="Path to SpotifyFeatures CSV (audio features + genre)")
    args = parser.parse_args()
    main(args.tracks, args.features)

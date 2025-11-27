# Spotify Data Analysis Tracker
Basic data checks: head/info/missing counts/describe for each provided CSV.
Popularity extremes: top and bottom 10 by popularity if that column exists in any loaded dataset.
Correlations: Pearson correlation heatmap across all numeric columns from the first non-empty dataset (merged if available, else tracks/features/artists).
Regression plots: energy vs loudness, and acousticness vs popularity (only if those columns are present).
Time trends (tracks dataset only): count of tracks per year; average duration per year if duration_sec exists (derived from duration_ms).
Genre analysis (features or artists): top genres by count; top genres by average popularity if popularity exists. Handles genre lists split by ;, |, or ,.
Distributions: histograms for popularity (if found) and energy (if present).
Cleaning steps: deduplication, column name trimming, duration conversion, release_date parsing to year, merge tracks+features when keys allow.

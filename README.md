# Music Genre Classification | Experimentation, training and pipelines

This repository contains a full pipeline for music genre recognition and feature extraction, from raw data pre-processing to model training and artifact generation. The main workflow is organized through Python scripts in the `executables` directory, and exploration and visualization are typically done in Jupyter notebooks.

## Table of Contents

- [Project Overview](#project-overview)
- [Executables Directory](#executables-directory)
- [Notebooks](#notebooks)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Project Overview

This project aims to automate the process of music genre recognition using audio data. The pipeline covers:
1. Metadata preparation and genre generalization
2. Audio signal transformation and extraction
3. Feature extraction (Fourier, spectrograms, MFCC, chroma, etc.)
4. Model training for classification
5. Artifact generation for single tracks (including graphs, videos, and extracted features)

---

## Executables Directory

The `executables` directory contains all core scripts for the pipeline. Here is a summary of each file:

### 0_generalisation_pipeline.py
- **Purpose**: Cleans and generalizes genre metadata.
- **Description**: Loads the FMA metadata, generalizes missing or ambiguous genres based on subgenre information, and produces a cleaned tracks dataset with the main genre labels for downstream processing.

### 1_transformation_pipeline.py
- **Purpose**: Audio transformation and signal extraction.
- **Description**: Reads metadata and MP3 files, extracts fixed-length audio signals (e.g., 14 seconds), and saves them as parquet files for efficient processing. Batches audio loading and supports large-scale parallel processing.

### 2_feature_extraction.py
- **Purpose**: Feature extraction from audio signals.
- **Description**: Extracts a variety of features from the saved signals, such as:
  - Waveform arrays
  - Fourier transforms
  - Spectrograms (standard, mel, power)
  - MFCCs
  - Chroma (STFT, CQT, CENS)
  - Tonnetz
  Saves all features and their corresponding labels in `.npy` format for model training.

### 3_model_training.py
- **Purpose**: Model training for genre classification.
- **Description**: Loads extracted features, splits into train/test sets, and trains a set of neural network models (one per feature type, e.g., waveform, MFCC, chroma, etc.). Handles normalization, logs accuracy, and saves all trained models and normalization artifacts.

### 4_artifacts_gen_single_track.py
- **Purpose**: Generate features/artifacts for a single audio track.
- **Description**: For a given track, generates all feature types, creates graphs/visualizations (and optionally videos), and saves everything for inference or demonstration. Useful for web/demo applications or validating the full pipeline on a single sample.

### utils (directory)
- **Purpose**: Helper utilities for data loading, saving, audio processing, etc. Used by the main scripts.

---

## Notebooks

Jupyter notebooks in this repository are typically used for:
- Exploratory Data Analysis (EDA) on the music dataset
- Visualization of audio features (waveforms, spectrograms, chroma, etc.)
- Prototyping model architectures and hyperparameters
- Analyzing model results and confusion matrices
- Comparing feature importance and ablation studies

**Typical notebook workflow:**
- Load metadata and inspect class distribution
- Visualize random samples from each genre
- Display extracted features for qualitative analysis
- Plot training/validation accuracy over epochs
- Visualize confusion matrices per model/feature type

> If you want a notebook-by-notebook summary, please provide their filenames or upload their contents.

---

## How to Run

Every file in executables directory uses argparse, so you may use -h option for more details 
```
py3 <executabe> -h 
```


1. **Prepare Metadata**  
   ```
   python executables/0_generalisation_pipeline.py --metadata_path <metadata_folder>
   ```

2. **Extract Signals**  
   ```
   python executables/1_transformation_pipeline.py -m <metadata_path> -o <output_path> -d <data_path> 
   ```

3. **Extract Features**  
   ```
   python executables/2_feature_extraction.py -d <signals_path> -o <output_path>
   ```

4. **Train Models**  
   ```
   python executables/3_model_training.py -d <features_path> -o <output_path> -m <metadata_path> -e <epochs>
   ```

5. **Generate Artifacts for a Single Track**  
   ```
   python executables/4_artifacts_gen_single_track.py --server-data <server_path> --metadata <metadata_path> --song-id <id> [--generate-video] [--generate-features] --hop-size <hop> --signal-length <length>
   ```

---

## Dependencies

- Python 3.10+
- numpy, pandas, librosa, torch, scikit-learn, soundfile, fastparquet, etc.
- See `requirements` for a full list.

---

## Citation

If you use this codebase, please cite the repository or contact the author.

```

---

Feel free to ask for notebook-specific summaries or details on the `utils` submodules!

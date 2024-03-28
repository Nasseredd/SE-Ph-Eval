Speech Enhancement Metrics Computation

This folder comprises two Python scripts, `metrics_all_samples.py` and `metrics.py`, designed to compute multichannel speech enhancement metrics for a specific dataset of audio samples. The metrics include Signal-to-Distortion Ratio (SDR), Signal-to-Artifacts Ratio (SAR), and Signal-to-Interference Ratio (SIR) for both mixture (input) and estimated speech (output) stages.

## `metrics.py`

### Overview

`metrics.py` is a core script that calculates SDR, SAR, and SIR metrics for a single audio sample. It utilizes the `mir_eval` library for the computation, focusing on the input and output stages of speech enhancement.

### Functionality

- Loads audio files for clean speech, noise, mixture, and estimated speech.
- Computes SDR, SAR, and SIR at both input and output stages and their deltas.
- Assumes stereo input and processes channels 0 and 2.

### Usage

The script is intended to be imported and utilized by other scripts for metric computation. It requires a dictionary of parameters specifying paths to the required audio files and metadata.

## `metrics_all_samples.py`

### Overview

`metrics_all_samples.py` extends the functionality of `metrics.py` to process multiple audio samples across specified conditions including scenario, gender, model, and microphones.

### Functionality

- Extracts filenames for processing from a specified dataset directory, filtering by gender.
- Iterates over a list of samples to compute metrics using `metrics.py`.
- Aggregates and displays metrics summaries for all processed samples.

### Usage

Designed to be run as a standalone script, it requires specification of the dataset path, gender, noise type, scenario, and model. The script outputs a comprehensive metrics summary for each sample processed.

## Getting Started

To use these scripts, you will need the following Python libraries: `os`, `glob`, `pandas`, `librosa`, `numpy`, and `mir_eval`.

1. Install the required libraries using pip:
   ```bash
   pip install pandas librosa numpy mir_eval
2. Adjust the dataset_path and other parameters in metrics_all_samples.py to match your dataset structure.
3. Run metrics_all_samples.py to compute metrics for all samples:
    ```python
    python metrics_all_samples.py

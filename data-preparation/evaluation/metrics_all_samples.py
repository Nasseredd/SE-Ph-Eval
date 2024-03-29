#!/usr/bin/env python
"""
Calculate multichannel speech enhancement metrics for:
    * 1 scenario (single snr level, speech angle and noise angle)
    * 1 model 
    * front-left and front-right microphones
    * all samples

Author: Nasser-eddine Monir
Update: 28.03.2024
"""
# import standard libraries
import os
import argparse
from typing import List
from datetime import datetime

# import third-party libraries
import glob 
import pandas as pd
from tqdm import tqdm

# import local libraries
from metrics import compute_utterance_level_metrics, compute_phoneme_level_metrics_for_speaker  


now = datetime.now() # Get the current date and time

def extract_filenames(dataset_path: str) -> List[str]:
    """
    Extracts and returns a list of filenames without their directory paths and file extensions.

    Parameters
    ----------
    dataset_path : str
        The base path to the dataset containing audio files organized by noise type and scenario.

    Returns
    -------
    List[str]
        A list of filenames extracted from the provided path pattern, with each filename stripped
        of its directory path and file extension. Only the base names of the files are returned.

    Examples
    --------
    >>> filenames = extract_filenames(clean_path="clean")
    >>> print(filenames)
    ["filename1", "filename2", "filename3"]
    """
    # Use glob to match the file pattern and extract base filenames without extension
    clean_files = os.path.join(dataset_path, 'clean', '*.wav')
    filenames = [file.split('/')[-1].replace('.wav', '') for file in glob.glob(clean_files)]
    return filenames

# Utterance Level
def process_all_utterance_level_cases(dataset_path: str, noise_type: str, scenario: str, model: str, samples: list) -> List[dict]:
    """
    Iterates through a collection of samples to compute and display metrics for each case. 
    Metrics are calculated based on speech enhancement performance, including SDR, SAR, and SIR values.

    Parameters
    ----------
    dataset_path : str
        The base path to the dataset containing audio files organized by noise type and scenario.
    noise_type : str
        The type of noise present in the audio samples (e.g., 'ssn', 'babble').
    scenario : str
        The specific scenario or condition under which the audio samples were processed or recorded.
    model : str
        The name of the model used for generating the estimated speech audio files.
    samples : list of str
        A list of sample identifiers. Each identifier corresponds to a unique case within the dataset.

    Outputs
    -------
    list
        A list containing dictionaries for each sample, with each dictionary summarizing the computed metrics.
        Each dictionary includes keys for filename, model, scenario, microphone channel, SDR, SAR, SIR metrics at input
        and output stages, and their deltas (changes from input to output stage).


    Notes
    -----
    - The function assumes a specific directory structure based on `dataset_path`, `noise_type`, `scenario`,
      and `sample` to locate the audio files (clean speech, noise, mixture, and estimated speech).
    - The audio file names are expected to follow a predefined naming convention.
    """
    metrics_summary = []
    for sample in samples:
        # Construct file paths for audio components based on input parameters
        paths = {
            'speech_path'              : os.path.join(dataset_path, "mix", noise_type, scenario, sample, 'reverberated-speech.wav'),
            'noise_path'               : os.path.join(dataset_path, "mix", noise_type, scenario, sample, 'reverberated-scaled-noise.wav'),
            'mixture_path'             : os.path.join(dataset_path, "mix", noise_type, scenario, sample, 'mixture.wav'),
            'estimated_speech_ch0_path': os.path.join(dataset_path, "estimated_speech", model, noise_type, scenario, sample, 'estimated_speech_ch0.wav'),
            'estimated_speech_ch2_path': os.path.join(dataset_path, "estimated_speech", model, noise_type, scenario, sample, 'estimated_speech_ch2.wav'),
        }
        
        # Merge other parameters with file paths to create the params dict for compute_metrics
        params = {**paths, 'model': model, 'sample': sample, 'scenario': scenario}
        
        # Compute the metrics for each sample
        current_metrics_summary = compute_utterance_level_metrics(params)  # Assume compute_metrics is defined elsewhere
        metrics_summary.extend(current_metrics_summary)
    
    return metrics_summary

# Phoneme Level
def process_all_phoneme_level_cases(speaker_ids, model, noise_type, scenario, dataset_path):
    metrics_summary = []
    for speaker_id in tqdm(speaker_ids):
        params = {
            'speaker_id'               : speaker_id,
            'dataset_path'             : dataset_path, 
            'model'                    : model,
            'noise_type'               : noise_type,
            'scenario'                 : scenario,
        }
        current_metrics_summary = compute_phoneme_level_metrics_for_speaker(params)
        metrics_summary.extend(current_metrics_summary)
    return metrics_summary

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Process some integers.') # Create the parser
    parser.add_argument('-l', '--level', type=str, help='Evaluation level (utterance/phoneme)')
    parser.add_argument('-m', '--model', type=str, help='Model')
    parser.add_argument('-n', '--noise_type', type=str, help='Noise type')
    parser.add_argument('-s', '--scenario', type=str, help='Scenario')
    parser.add_argument('-g', '--gender', type=str, help='Gender')
    args = parser.parse_args()

    # Evaluation 
    dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/se-ph-eval-plus'
    results_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/se-ph-eval-plus/result-records'
    if args.level == 'utterance':
        filenames = extract_filenames(dataset_path)
        metrics_summary = process_all_utterance_level_cases(
                            noise_type=args.noise_type, 
                            scenario=args.scenario, 
                            model=args.model, 
                            samples=filenames,
                            dataset_path=dataset_path
                            )
        time = now.strftime("%d_%m_%Y_%H_%M") # Format the date and time as day_month_year_hour_min
        experiment_info = f"{args.model}_{args.noise_type}_{args.gender}_{args.scenario}"
        metrics_path = f"results_utt_{experiment_info}_{time}.wav"
        df = pd.DataFrame(metrics_summary)
        df.to_csv(metrics_path, sep=',', index=False)
    elif args.level == 'phoneme':
        filenames = extract_filenames(dataset_path)
        speaker_ids = set(filename.split('-')[0] for filename in filenames)
        metrics_summary = process_all_phoneme_level_cases(
                            speaker_ids,
                            model=args.model,
                            noise_type=args.noise_type,
                            scenario=args.scenario,
                            dataset_path=dataset_path
                            )
        time = now.strftime("%d_%m_%Y_%H_%M") # Format the date and time as day_month_year_hour_min
        experiment_info = f"{args.model}_{args.noise_type}_{args.gender}_{args.scenario}"
        metrics_path = f"results_ph_{experiment_info}_{time}.wav"
        df = pd.DataFrame(metrics_summary)
        df.to_csv(metrics_path, sep=',', index=False)
    else:
        print('[ERROR] Evaluation level')
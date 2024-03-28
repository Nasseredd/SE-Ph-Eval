#!/usr/bin/env python
"""
Calculate multichannel speech enhancement metrics for:
    * 1 scenario (single snr level, speech angle and noise angle)
    * 1 gender 
    * 1 model 
    * front-left and front-right microphones
    * all samples

Author: Nasser-eddine Monir
Update: 28.03.2024
"""
# import standard libraries
import os
import glob 
from typing import List

# import third-party libraries
import pandas as pd

# import local libraries
from metrics import compute_metrics  


def extract_filenames(dataset_path: str, gender: str) -> List[str]:
    """
    Extracts and returns a list of filenames without their directory paths and file extensions.

    Parameters
    ----------
    dataset_path : str
        The base path to the dataset containing audio files organized by gender, noise type, and scenario.
    gender : str
        The gender of the speaker in the audio samples ('male' or 'female').

    Returns
    -------
    List[str]
        A list of filenames extracted from the provided path pattern, with each filename stripped
        of its directory path and file extension. Only the base names of the files are returned.

    Examples
    --------
    >>> filenames = extract_filenames(clean_path="clean", gender="female")
    >>> print(filenames)
    ["filename1", "filename2", "filename3"]
    """
    # Use glob to match the file pattern and extract base filenames without extension
    clean_files = os.path.join(dataset_path, 'clean', gender, '*.wav')
    filenames = [file.split('/')[-1].replace('.wav', '') for file in glob.glob(clean_files)]
    return filenames


def process_all_cases(dataset_path: str, gender: str, noise_type: str, scenario: str, model: str, samples: list) -> List[dict]:
    """
    Iterates through a collection of samples to compute and display metrics for each case. 
    Metrics are calculated based on speech enhancement performance, including SDR, SAR, and SIR values.

    Parameters
    ----------
    dataset_path : str
        The base path to the dataset containing audio files organized by gender, noise type, and scenario.
    gender : str
        The gender of the speaker in the audio samples ('male' or 'female').
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
    - The function assumes a specific directory structure based on `dataset_path`, `gender`, `noise_type`, `scenario`,
      and `sample` to locate the audio files (clean speech, noise, mixture, and estimated speech).
    - The audio file names are expected to follow a predefined naming convention.
    """
    metrics_summary = []
    for sample in samples:
        # Construct file paths for audio components based on input parameters
        paths = {
            'speech_path'              : os.path.join(dataset_path, "mix", gender, noise_type, scenario, sample, 'reverberated-speech.wav'),
            'noise_path'               : os.path.join(dataset_path, "mix", gender, noise_type, scenario, sample, 'reverberated-scaled-noise.wav'),
            'mixture_path'             : os.path.join(dataset_path, "mix", gender, noise_type, scenario, sample, 'mixture.wav'),
            'estimated_speech_path_ch0': os.path.join(dataset_path, "estimated_speech", model, gender, noise_type, scenario, sample, 'estimated_speech_ch0.wav'),
            'estimated_speech_path_ch2': os.path.join(dataset_path, "estimated_speech", model, gender, noise_type, scenario, sample, 'estimated_speech_ch2.wav'),
        }
        
        # Merge other parameters with file paths to create the params dict for compute_metrics
        params = {**paths, 'model': model, 'sample': sample, 'scenario': scenario, 'gender': gender}
        
        # Compute the metrics for each sample
        current_metrics_summary = compute_metrics(params)  # Assume compute_metrics is defined elsewhere
        metrics_summary.extend(current_metrics_summary)
    
    return metrics_summary


if __name__ == "__main__":
    
    dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/se-ph-eval-plus'
    filenames = extract_filenames(dataset_path, gender='female')
    
    metrics_summary = process_all_cases(
                        gender='female', 
                        noise_type='ssn-mf', 
                        scenario='0dBS0N45', 
                        model='tango', 
                        samples=filenames,
                        dataset_path=dataset_path, 
                        )
    print(pd.DataFrame(metrics_summary))

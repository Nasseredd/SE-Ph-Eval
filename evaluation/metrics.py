#!/usr/bin/env python
"""
Calculate multichannel speech enhancement metrics at the input stage, the output stage and the delta between them: SDR, SAR, SIR.

Author: Nasser-eddine Monir
Update: 28.03.2024
"""
# import standard libraries
import os
import sys
import json
from typing import List, Dict, Tuple

# import third-party libraries
import glob
import librosa
import numpy as np
import pandas as pd
from mir_eval.separation import bss_eval_images

def extract_speaker_filenames(dataset_path: str, speaker_id: str) -> List[str]:
    """
    Extracts and returns a list of filenames without their directory paths and file extensions.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------
    >>> filenames = extract_filenames(clean_path="clean",)
    >>> print(filenames)
    ["filename1", "filename2", "filename3"]
    """
    # Use glob to match the file pattern and extract base filenames without extension
    clean_files = os.path.join(dataset_path, 'clean', f'{speaker_id}-*.wav')
    filenames = [file.split('/')[-1].replace('.wav', '') for file in glob.glob(clean_files)]
    return filenames

# Utterance Level  
def compute_utterance_level_metrics(params: dict)-> List[dict]:
    """
    Calculate multichannel speech enhancement metrics at an utterance level at the input stage, the output stage and the delta between them: SDR, SAR, SIR.
    
    Parameters
    ----------
    params : dict
        A dictionary containing paths to audio files and metadata.
        Expected keys are:
        - 'speech_path': str, path to the clean speech audio file.
        - 'noise_path': str, path to the noise audio file.
        - 'mixture_path': str, path to the mixture of speech and noise audio file.
        - 'estimated_speech_path': str, path to the estimated speech audio file.
        - 'sample': str, the name of the audio file being evaluated.
        - 'model': str, the name of the model used for speech enhancement.
        - 'scenario': str, a description of the scenario or conditions under which the audio was processed.
        
    Returns
    -------
    metrics_summary : list of dicts
        A list where each element is a dictionary containing the metrics for each microphone channel ('Front-Left', 'Front-Right').
        Each dictionary includes:
        - 'sample': str, the name of the audio file.
        - 'model': str, the model used for speech enhancement.
        - 'scenario': str, the scenario description.
        - 'microphone': str, the microphone channel.
        - 'sdr_in': float, Signal to Distortion Ratio before enhancement.
        - 'sdr_out': float, Signal to Distortion Ratio after enhancement.
        - 'sar_in': float, Signal to Artifacts Ratio before enhancement.
        - 'sar_out': float, Signal to Artifacts Ratio after enhancement.
        - 'sir_in': float, Signal to Interference Ratio before enhancement.
        - 'sir_out': float, Signal to Interference Ratio after enhancement.
        - 'delta_sdr': float, change in SDR from before to after enhancement.
        - 'delta_sar': float, change in SAR from before to after enhancement.
        - 'delta_sir': float, change in SIR from before to after enhancement.
        
    Notes
    -----
    This function assumes the input and estimated speech signals are stereo and selects channels 0 and 2 for processing.
    The speech enhancement metrics are computed using the mir_eval library's bss_eval_images function, which is designed for
    evaluating the quality of speech enhancement (source separation) algorithms.
    """
    
    # Load audios
    speech               = librosa.load(params['speech_path'], sr=None, mono=False)[0][[0, 2]]
    noise                = librosa.load(params['noise_path'], sr=None, mono=False)[0][[0, 2]]
    mixture              = librosa.load(params['mixture_path'], sr=None, mono=False)[0][[0, 2]]
    estimated_speech_ch0 = librosa.load(params['estimated_speech_ch0_path'], sr=None, mono=False)[0]
    estimated_speech_ch2 = librosa.load(params['estimated_speech_ch2_path'], sr=None, mono=False)[0]
    estimated_speech     = np.vstack((estimated_speech_ch0, estimated_speech_ch2))

    # Prepare reference and estimated source matrics for metric computation
    reference      = np.vstack([speech, noise])
    residual_noise = mixture - estimated_speech
    estimation_in  = np.vstack([mixture, mixture])
    estimation_out = np.vstack([estimated_speech, residual_noise])

    # Compute speech enhancement metrics using mir_eval
    sdr_in, _, sir_in, sar_in, _    = bss_eval_images(reference_sources=reference, estimated_sources=estimation_in, compute_permutation=False)
    sdr_out, _, sir_out, sar_out, _ = bss_eval_images(reference_sources=reference, estimated_sources=estimation_out, compute_permutation=False)

    # Prepare and return summary of metrics for each microphone channel 
    metrics_summary = []
    for i, microphone in enumerate(['Front-Left','Front-Right']):
        metrics_summary.append({
            'sample'          : params['sample'],
            'model'           : params['model'],
            'gender'          : params['gender'],
            'scenario'        : params['scenario'],
            'microphone'      : microphone,
            'phoneme_category': 'utterance',
            'sdr_in'          : round(sdr_in[i], 4),
            'sdr_out'         : round(sdr_out[i], 4),
            'sar_in'          : round(sar_in[i], 4),
            'sar_out'         : round(sar_out[i], 4),
            'sir_in'          : round(sir_in[i], 4),
            'sir_out'         : round(sir_out[i], 4),
            'delta_sdr'       : round(sdr_out[i] - sdr_in[i], 4),
            'delta_sar'       : round(sar_out[i] - sar_in[i], 4),
            'delta_sir'       : round(sir_out[i] - sir_in[i], 4),
        })
    
    return metrics_summary

# Phoneme Level
def get_paths(params: dict, sample: str) -> dict:
    """
    Generates a dictionary of paths for various audio and data files related to a given dataset sample.
    
    Parameters
    ----------
    params : dict
        A dictionary containing parameters for the dataset. Expected keys are 'dataset_path', 'gender',
        'noise_type', and 'scenario'.
    sample : str
        The sample identifier used to generate paths to specific files within the dataset.
    
    Returns
    -------
    dict
        A dictionary with keys corresponding to different types of data (e.g., 'speech_path', 'noise_path', etc.)
        and values as strings representing the full paths to these files.
    
    """
    dataset_path, noise_type, scenario = (params['dataset_path'], params['gender'], params['noise_type'], params['scenario'])
    paths = {
        'speech_path'               : os.path.join(dataset_path, f'mix/{gender}/{noise_type}/{scenario}/{sample}/reverberated-speech.wav'),
        'noise_path'                : os.path.join(dataset_path, f'mix/{gender}/{noise_type}/{scenario}/{sample}/reverberated-scaled-noise.wav'),
        'mixture_path'              : os.path.join(dataset_path, f'mix/{gender}/{noise_type}/{scenario}/{sample}/mixture.wav'),
        'phoneme_segmentations_path': os.path.join(dataset_path, f'clean/{gender}/{sample}.json'),
        'estimated_speech_ch0_path' : os.path.join(dataset_path, f'estimated_speech/tango/{gender}/{noise_type}/{scenario}/{sample}/estimated_speech_ch0.wav'),
        'estimated_speech_ch2_path' : os.path.join(dataset_path, f'estimated_speech/tango/{gender}/{noise_type}/{scenario}/{sample}/estimated_speech_ch2.wav'),
    }
    return paths

def load_phoneme_categories(phoneme_categories_file: str) -> List[str]:
    """
    Reads a JSON file containing phoneme categories and returns a list of the phoneme category names.

    Parameters
    ----------
    phoneme_categories_file : str
        The path to the JSON file containing phoneme categories.

    Returns
    -------
    List[str]
        A list of strings, each representing a phoneme category name extracted from the JSON file's keys.
    """
    with open(phoneme_categories_file, 'r') as f:
        phoneme_categories = list(json.load(f).keys())
    return phoneme_categories


def load_audios(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads audio data from the specified paths for speech, noise, mixture, and estimated speech channels.

    Parameters
    ----------
    paths : Dict[str, str]
        A dictionary with keys 'speech_path', 'noise_path', 'mixture_path', 
        'estimated_speech_ch0_path', and 'estimated_speech_ch2_path', where each key is associated
        with the path to the corresponding audio file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing four NumPy arrays:
        - The first array contains the stereo audio data for the speech.
        - The second array contains the stereo audio data for the noise.
        - The third array contains the stereo audio data for the mixture of speech and noise.
        - The fourth array contains the combined estimated speech channels.
    """
    speech               = librosa.load(paths['speech_path'], sr=None, mono=False)[0][[0, 2]]
    noise                = librosa.load(paths['noise_path'], sr=None, mono=False)[0][[0, 2]]
    mixture              = librosa.load(paths['mixture_path'], sr=None, mono=False)[0][[0, 2]]
    estimated_speech_ch0 = librosa.load(paths['estimated_speech_ch0_path'], sr=None, mono=False)[0]
    estimated_speech_ch2 = librosa.load(paths['estimated_speech_ch2_path'], sr=None, mono=False)[0]
    estimated_speech     = np.vstack((estimated_speech_ch0, estimated_speech_ch2))

    return speech, noise, mixture, estimated_speech

def retrieve_phonemes(phoneme_seg_file: str) -> List[Tuple[float, float, str, str]]:
    """
    Loads phoneme segmentations from a given JSON file and returns a list of phoneme segmentations.

    Each phoneme segmentation includes the start time, end time, phoneme symbol, and an additional identifier or label.

    Parameters
    ----------
    phoneme_seg_file : str
        The path to the JSON file containing phoneme segmentations for a speech audio.

    Returns
    -------
    List[Tuple[float, float, str, str]]
        A list of tuples, where each tuple contains:
        - Start time of the phoneme (float),
        - End time of the phoneme (float),
        - The phoneme symbol (str),
        - An additional identifier or label (str).
    """
    with open(phoneme_seg_file, 'r') as f:
        phoneme_segs = json.load(f)['segmentation']
    
    phoneme_segs = [(float(s[0]), float(s[1]), s[2], s[3]) for s in phoneme_segs]
    return phoneme_segs

def speech_enhancement_metrics(
        speech: np.ndarray, 
        noise: np.ndarray, 
        mixture: np.ndarray, 
        estimated_speech: np.ndarray
    ) -> Tuple[float, float, float, float, float, float]:
    """
    Computes speech enhancement metrics, including Signal-to-Distortion Ratio (SDR), 
    Signal-to-Artifact Ratio (SAR), and Signal-to-Interference Ratio (SIR), 
    before and after speech enhancement.

    Parameters
    ----------
    speech : np.ndarray
        The clean speech signal.
    noise : np.ndarray
        The noise signal.
    mixture : np.ndarray
        The mixture of speech and noise (i.e., the corrupted speech signal).
    estimated_speech : np.ndarray
        The estimated clean speech signal obtained after speech enhancement.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        A tuple containing the following metrics:
        - sdr_in: The Signal-to-Distortion Ratio before enhancement.
        - sdr_out: The Signal-to-Distortion Ratio after enhancement.
        - sar_in: The Signal-to-Artifact Ratio before enhancement.
        - sar_out: The Signal-to-Artifact Ratio after enhancement.
        - sir_in: The Signal-to-Interference Ratio before enhancement.
        - sir_out: The Signal-to-Interference Ratio after enhancement.
    """
    
    # Prepare reference and estimated source matrices for metric computation
    reference = np.vstack([speech, noise])
    residual_noise = mixture - estimated_speech
    estimation_in = np.vstack([mixture, mixture])
    estimation_out = np.vstack([estimated_speech, residual_noise])

    # Compute speech enhancement metrics using mir_eval
    sdr_in, _, sir_in, sar_in, _ = bss_eval_images(reference_sources=reference, estimated_sources=estimation_in, compute_permutation=False)
    sdr_out, _, sir_out, sar_out, _ = bss_eval_images(reference_sources=reference, estimated_sources=estimation_out, compute_permutation=False)

    return sdr_in, sdr_out, sar_in, sar_out, sir_in, sir_out

def compute_phoneme_level_metrics_for_speaker(params: Dict) -> List[Dict]:
    """
    Calculate multichannel speech enhancement metrics per speaker at a phoneme level, including at the input and 
    output stages, and the deltas between them (SDR, SAR, SIR) for different phoneme categories.

    Parameters
    ----------
    params : Dict
        A dictionary containing parameters for calculating the metrics, with keys:
        - 'dataset_path': The path to the dataset.
        - 'speaker_id': The ID of the speaker.
        - 'model': The name of the model used for speech enhancement.

    Returns
    -------
    List[Dict]
        A list of dictionaries, each containing the metrics for a specific phoneme category and microphone channel:
        - 'model': The name of the model.
        - 'scenario': The scenario under which the metrics were calculated.
        - 'microphone': The name of the microphone channel ('Front-Left' or 'Front-Right').
        - 'phoneme_category': The category of the phoneme.
        - 'sdr_in': The Signal-to-Distortion Ratio before enhancement.
        - 'sdr_out': The Signal-to-Distortion Ratio after enhancement.
        - 'sar_in': The Signal-to-Artifact Ratio before enhancement.
        - 'sar_out': The Signal-to-Artifact Ratio after enhancement.
        - 'sir_in': The Signal-to-Interference Ratio before enhancement.
        - 'sir_out': The Signal-to-Interference Ratio after enhancement.
        - 'delta_sdr': The change in SDR due to enhancement.
        - 'delta_sar': The change in SAR due to enhancement.
        - 'delta_sir': The change in SIR due to enhancement.

    Notes
    -----
    This function relies on external functions to load phoneme categories, extract filenames, get paths, load audios,
    retrieve phoneme segmentations, and compute speech enhancement metrics, which must be correctly implemented for 
    it to work.
    """
    # Load phoneme categories
    phoneme_categories = load_phoneme_categories(phoneme_categories_file='/home/nasmonir/PhD/se-up-evaluations/utils/phoneme-segmentation/phoneme-classes.json')

    # Loop across phoneme categories
    metrics_summary = []
    for phoneme_category in phoneme_categories:
        print(f"[INFO] Phoneme Category: {phoneme_category}")

        # Initialize
        speech_phCat, noise_phCat, mixture_phCat, estimated_speech_phCat = (np.empty((2,1)), np.empty((2,1)), np.empty((2,1)), np.empty((2,1)))

        # Loop over files and get sample of the phoneme category for speech, mixture and estimated speech 
        speaker_filenames = extract_speaker_filenames(dataset_path=params['dataset_path'], speaker_id=params['speaker_id'])
        for sample in speaker_filenames:
            
            # Load sample's audios and phoneme segmentation
            paths = get_paths(params, sample) # Get file paths
            speech, noise, mixture, estimated_speech = load_audios(paths) # Load audios
            phonemes_timeframes = retrieve_phonemes(paths['phoneme_segmentations_path']) # Load phoneme segmentation

            # Loop across phonemes
            for start, end, phoneme, current_phoneme_category in phonemes_timeframes:
                if current_phoneme_category == phoneme_category:
                    
                    # Get all samples of the phoneme category through all speeches
                    start_sample, end_sample = int(start*16000), int(end*16000)
                    speech_phCat           = np.concatenate([speech_phCat, speech[:, start_sample:end_sample+1]], axis=1) 
                    noise_phCat            = np.concatenate([noise_phCat, noise[:, start_sample:end_sample+1]], axis=1) 
                    mixture_phCat          = np.concatenate([mixture_phCat, mixture[:, start_sample:end_sample+1]], axis=1) 
                    estimated_speech_phCat = np.concatenate([estimated_speech_phCat, estimated_speech[:, start_sample:end_sample+1]], axis=1)

        # Compute speech enhancement metrics using mir_eval
        sdr_in, sdr_out, sar_in, sar_out, sir_in, sir_out = speech_enhancement_metrics(
            speech=speech_phCat, noise=noise_phCat, 
            mixture=mixture_phCat, estimated_speech=estimated_speech_phCat
            )
        
        # Prepare and return summary of metrics for each microphone channel 
        for i, microphone in enumerate(['Front-Left','Front-Right']):
            metrics_summary.append({
                'model'           : params['model'],
                'gender'          : params['gender'],
                'scenario'        : params['scenario'],
                'microphone'      : microphone,
                'phoneme_category': phoneme_category,
                'sdr_in'          : round(sdr_in[i], 4),
                'sdr_out'         : round(sdr_out[i], 4),
                'sar_in'          : round(sar_in[i], 4),
                'sar_out'         : round(sar_out[i], 4),
                'sir_in'          : round(sir_in[i], 4),
                'sir_out'         : round(sir_out[i], 4),
                'delta_sdr'       : round(sdr_out[i] - sdr_in[i], 4),
                'delta_sar'       : round(sar_out[i] - sar_in[i], 4),
                'delta_sir'       : round(sir_out[i] - sir_in[i], 4),
            })
    return metrics_summary

# Main execution block 
if __name__ == '__main__':
    # Utterance level evaluation for a single sample
    """dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/se-ph-eval-plus/'
    params = {
        'speech_path'              : os.path.join(dataset_path, 'mix/female/ssn-m/0dBS0N45/121-121726-0000/reverberated_speech.wav'),
        'noise_path'               : os.path.join(dataset_path, 'mix/female/ssn-m/0dBS0N45/121-121726-0000/reverberated_scaled_noise.wav'),
        'mixture_path'             : os.path.join(dataset_path, 'mix/female/ssn-m/0dBS0N45/121-121726-0000/mixture.wav'),
        'estimated_speech_ch0_path': os.path.join(dataset_path, 'estimated_speech/tango/female/ssn-m/0dBS0N45/121-121726-0000/estimated_speech_ch0.wav'),
        'estimated_speech_ch2_path': os.path.join(dataset_path, 'estimated_speech/tango/female/ssn-m/0dBS0N45/121-121726-0000/estimated_speech_ch2.wav'),
        'sample'                   : '121-121726-0000',
        'model'                    : 'tango',
        'gender'                   : 'female',
        'scenario'                 : '0dBS0N45',
    }

    metrics_summary = compute_metrics(params)
    print(metrics_summary)"""

    # Phoneme Level evaluation for a single speaker 
    dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/se-ph-eval-plus/'
    """params = {
        'speaker_id'               : '61',
        'dataset_path'             : dataset_path, 
        'model'                    : 'tango',
        'noise_type'               : 'ssn-mf',
        'gender'                   : 'male',
        'scenario'                 : '0dBS0N45',
    }
    metrics_summary = compute_phoneme_level_metrics_for_speaker(params)
    print(pd.DataFrame(metrics_summary))"""
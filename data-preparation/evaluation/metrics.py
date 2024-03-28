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
from typing import List

# import third-party libraries
import librosa
import numpy as np
from mir_eval.separation import bss_eval_images

# Compute metrics 
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
        - 'gender': str, the gender of the speaker in the audio samples ('male' or 'female').
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
    estimated_speech_ch0 = librosa.load(params['estimated_speech_path_ch0'], sr=None, mono=False)[0]
    estimated_speech_ch2 = librosa.load(params['estimated_speech_path_ch2'], sr=None, mono=False)[0]
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

def compute_phoneme_level_metrics_per_speaker(params: dict)-> List[dict]:
    """
    Calculate multichannel speech enhancement metrics per speaker at a phoneme level at the input stage, the output stage and the delta between them: SDR, SAR, SIR.
    
    Parameters
    ----------
    
        
    Returns
    -------
    
        
    Notes
    -----
    
    """ 
    pass

# Main execution block 
if __name__ == '__main__':
    # Utterance level evaluation for a single sample
    """dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/se-ph-eval-plus/'
    params = {
        'speech_path'              : os.path.join(dataset_path, 'mix/female/ssn-m/0dBS0N45/121-121726-0000/reverberated_speech.wav'),
        'noise_path'               : os.path.join(dataset_path, 'mix/female/ssn-m/0dBS0N45/121-121726-0000/reverberated_scaled_noise.wav'),
        'mixture_path'             : os.path.join(dataset_path, 'mix/female/ssn-m/0dBS0N45/121-121726-0000/mixture.wav'),
        'estimated_speech_path_ch0': os.path.join(dataset_path, 'estimated_speech/tango/female/ssn-m/0dBS0N45/121-121726-0000/estimated_speech_ch0.wav'),
        'estimated_speech_path_ch2': os.path.join(dataset_path, 'estimated_speech/tango/female/ssn-m/0dBS0N45/121-121726-0000/estimated_speech_ch2.wav'),
        'sample'                   : '121-121726-0000',
        'model'                    : 'tango',
        'gender'                   : 'female',
        'scenario'                 : '0dBS0N45',
    }

    metrics_summary = compute_metrics(params)
    print(metrics_summary)"""

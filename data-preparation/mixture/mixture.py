#!/usr/bin/env python
"""

"""
# import standard libraries
import os 
import sys 
import json

# import third-party libraries 
import glob
import librosa 
import numpy as np 
import soundfile as sf 
from tqdm import tqdm
from argparse import ArgumentParser

# Colors 
BLUE = "\033[34m"
RED = "\033[31m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
GREY = "\033[90m" 
RESET = "\033[0m"

def output_names(mix_dir, speech_id):
    """

    """
    # create the folder if it does not exist
    output_dir = os.path.join(mix_dir, speech_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # output names
    speech_rev_file       = os.path.join(output_dir, "reverberated-speech.wav")
    noise_rev_scaled_file = os.path.join(output_dir, "reverberated-scaled-noise.wav")
    noise_scaled_file     = os.path.join(output_dir, "scaled-noise.wav")
    mixture_file          = os.path.join(output_dir, "mixture.wav")

    return speech_rev_file, noise_rev_scaled_file, noise_scaled_file, mixture_file

def desired_snr_tag(desired_snr):
    if desired_snr == 0: return "0dB"
    elif desired_snr > 0: return f"p{desired_snr}dB"
    else: return f"m{abs(desired_snr)}dB"

def scenario_to_snr(scenario):
    if '0dB' in scenario: return 0.0
    elif 'p5dB' in scenario: return 5.0
    elif 'm5dB' in scenario: return -5.0
    else: return 'ERROR'

def load_configs_metadata(configs_file):
    with open(configs_file, "r") as f:
        configs = json.load(f)
    
    desired_snr = configs['desired-snr']
    speech_angle = "S" + str(configs['speech-angle'])
    noise_angle = "N" + str(configs['noise-angle'])
    scenario = configs['scenario']
    silence_threshold = configs['silence-threshold']

    return desired_snr, speech_angle, noise_angle, scenario, silence_threshold

def load_configs_paths(configs_file):
    with open(configs_file, "r") as f:
        configs = json.load(f)
    
    speech_rir_file = configs['rir']['speech']
    noise_rir_file = configs['rir']['noise']
    scenario_folder = configs['scenario-path']
    speech_files = configs['speech-files']
    dataset_folder = configs['dataset-folder']

    return dataset_folder, speech_files, scenario_folder, speech_rir_file, noise_rir_file

def adjust_noise_to_speech(speech, noise):
    if speech.shape[0] > noise.shape[0]: 
        noise = np.tile(noise, int(speech.shape[0]/noise.shape[0])+1)[:speech.shape[0]]
    else:
        noise = noise[:speech.shape[0]]
    
    return noise 

def compute_gain(speech_samples: np.ndarray, 
                 noise_samples: np.ndarray, 
                 snr_dB: float,
                 silence_threshold: float,
                 normalization_type: str = 'sqrt-power'):
    """
    """
    # remove silence from speech signal 
    speech_wo_silence = speech_samples[np.abs(speech_samples)>silence_threshold]

    # calculate powers of the signals
    speech_power = np.sum(np.square(speech_wo_silence)) / len(speech_wo_silence)
    noise_power  = np.sum(np.square(noise_samples)) / len(noise_samples)
    
    # calculate the gain to achieve the desired SNR
    gain_factor = np.sqrt((speech_power / noise_power) * 10**(-snr_dB / 10))

    return gain_factor, speech_power, noise_power

# Loading Audios 
def loading_audios(speech_file, speech_rir_file, noise_file, noise_rir_file):
    """

    """
    # Speech 
    speech, speech_sr = librosa.load(speech_file, sr=None, mono=False)

    # Noise
    noise, noise_sr = librosa.load(noise_file, sr=None, mono=False)

    # Sample Rate 
    if speech_sr != noise_sr:
        print("[ERROR] Speech Sample Rate ({}) ≠ Noise Sample Rate ({})")
        sys.exit()
    else: sample_rate = speech_sr

    return speech, noise, sample_rate

def loading_rirs(speech_rir_file, noise_rir_file):
    """

    """
    # Speech RIR 
    speech_rir_npz = np.load(speech_rir_file)
    speech_rir = np.stack((
                        speech_rir_npz['phl_left_front'], 
                        speech_rir_npz['phl_left_rear'], 
                        speech_rir_npz['phl_right_front'], 
                        speech_rir_npz['phl_right_rear']), axis=-1).T
    # Noise RIR
    noise_rir_npz = np.load(noise_rir_file)
    noise_rir = np.stack((
                        noise_rir_npz['phl_left_front'], 
                        noise_rir_npz['phl_left_rear'], 
                        noise_rir_npz['phl_right_front'], 
                        noise_rir_npz['phl_right_rear']), axis=-1).T
    return speech_rir, noise_rir

# Mixture
def paths(dataset_dir, noise, scenario, speech_angle, noise_angle):
    speech_files          = os.path.join(dataset_dir, f'clean/anechoic/*.wav') # Construct the path to the directory of clean speech files
    noise_file            = os.path.join(dataset_dir, f'noise/{noise}.wav') # Define the path to the noise file to be used in mixtures
    mix_dir               = os.path.join(dataset_dir, f'mix/{noise}/{scenario}/') # Create the path for saving output mixtures with specified noise type and scenario
    speech_rir_file       = os.path.join(dataset_dir, f'impulse-response/rir_{speech_angle}.npz') # Set the path to the speech room impulse response file
    noise_rir_file        = os.path.join(dataset_dir, f'impulse-response/rir_p{noise_angle}.npz') # Set the path to the noise room impulse response file
    return speech_files, noise_file, mix_dir, speech_rir_file, noise_rir_file

def generate_mixture(speech, noise, sample_rate, speech_rir, noise_rir, desired_snr, silence_threshold=0.01):
    """

    """

    # 0. normalization of speech and noise (see compute_gain) ?? 

    # 1. adjust the noise length to the speech length
    noise = adjust_noise_to_speech(speech, noise)

    # 2. compute the gain 
    gain_factor, speech_power, noise_power = compute_gain(speech, noise, snr_dB=desired_snr, silence_threshold=silence_threshold)
    
    # 3. scaled the noise given the gain
    noise_scaled = gain_factor*noise

    # 4. convolution(speech, speech_rir) and convolution(noise, noise_rir)
    speech_channels = [np.convolve(speech, ir_channel, mode="full") for ir_channel in speech_rir]
    speech_reverberated = np.vstack(speech_channels).T

    noise_channels = [np.convolve(noise_scaled, ir_channel, mode="full") for ir_channel in noise_rir]
    noise_reverberated_scaled = np.vstack(noise_channels).T

    # 5. compute the mixture = speech_convolved + noise_convolved
    mixture = speech_reverberated + noise_reverberated_scaled

    return mixture, speech_reverberated, noise_reverberated_scaled, noise_scaled

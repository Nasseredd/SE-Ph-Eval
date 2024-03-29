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

# import local libraries
from mixture import *

BLUE = "\033[34m"
RED = "\033[31m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
GREY = "\033[90m" 
RESET = "\033[0m"

def main(
        noise: str, 
        scenario: str, 
        sample_rate: int, 
        silence_threshold: float,
        dataset_dir: str,
        ) -> None:
    """
    
    Parameters
    ----------
    
    Returns
    -------
    

    Notes
    -----
    """

    desired_snr = scenario_to_snr(scenario)
    
    # ---------------------- RETRIEVE PATHS -----------------------
    speech_angle = scenario[-4]
    noise_angle  = scenario[-2:]
    speech_files, noise_file, mix_dir, speech_rir_file, noise_rir_file = paths(dataset_dir, noise, scenario, speech_angle, noise_angle)

    # -------------------------- LOADING -------------------------- 
    # Noise signal, and RIRs (speech and noise)
    noise, _              = librosa.load(noise_file, sr=sample_rate, mono=True) # Load the noise file with librosa, resampling to the specified sample rate and ensuring mono channel
    speech_rir, noise_rir = loading_rirs(speech_rir_file, noise_rir_file) # Load the room impulse response files for speech and noise

    # --------------------- MIXTURE GENERATION --------------------
    print(RED + "[START] MIXTURES GENERATION ... " + RESET)

    for speech_file in tqdm(glob.glob(speech_files)):
        
        # Speech and Speech ID
        speech, _ = librosa.load(speech_file, sr=sample_rate, mono=False) # load the speech signal 
        speech_id = os.path.basename(speech_file).replace(".wav", "") # get the speech id from the speech path
        
        # Compute Mixture (return mixture, reverb speech, reverb noise, scaled noise)
        mixture_channels, speech_reverberated, noise_reverberated_scaled, noise_scaled = generate_mixture(
            speech, noise, sample_rate, speech_rir, noise_rir, desired_snr=desired_snr, silence_threshold=silence_threshold
        )
        
        # Get output names for mixture, reverb speech, reverb noise and scaled noise
        speech_rev_file, noise_rev_scaled_file, noise_scaled_file, mixture_file = output_names(
            mix_dir, speech_id
        )

        # Export mixture, reverb speech, reverb noise, scaled noise
        sf.write(speech_rev_file, speech_reverberated, sample_rate)
        sf.write(noise_rev_scaled_file, noise_reverberated_scaled, sample_rate)
        sf.write(noise_scaled_file, noise_scaled, sample_rate)
        sf.write(mixture_file, mixture_channels, sample_rate)
    
    print(RED + "[END] MIXTURES GENERATION" + RESET)

# Main 
if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser(description="Generate mixtures from speech files.")
    parser.add_argument("--noise-type", type=str, choices=["white-noise", "speech-shaped-noise", "babble-noise"], help="Type of noise.")
    parser.add_argument("--scenario", type=str, choices=["p5dBS0N45", "0dBS0N45", "m5dBS0N45", "p5dBS0N90", "0dBS0N90", "m5dBS0N90"], help="Scenario identifier.")  
    args = parser.parse_args()

    # Generate mixtures
    main(
        noise=args.noise_type,
        scenario=args.scenario,
        sample_rate=16000,
        silence_threshold= 0.01,
        dataset_dir='/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/nmonir/corpus/nmansd/', 
        )
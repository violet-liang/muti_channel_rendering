import os
import numpy as np
import soundfile as sf
import pysofaconventions
import spaudiopy as spa
from scipy.signal import fftconvolve
import pyloudnorm as pyln
from spaudiopy import decoder


def load_hrtf(sofa_path):
    # Load HRTF data from SOFA file
    sofa = pysofaconventions.SOFAFile(sofa_path, 'r')
    az = sofa.getVariableValue('SourcePosition')[:, 0]
    el = sofa.getVariableValue('SourcePosition')[:, 1]
    hrir = sofa.getDataIR()  # (n_directions, 2, n_samples)
    return az, el, hrir


def spherical_to_cartesian(az_deg, el_deg, r=1.0):
 
    az_rad = np.deg2rad(az_deg)
    zen_rad = np.deg2rad(90 - el_deg)
    x = r * np.cos(az_rad) * np.sin(zen_rad)
    y = r * np.sin(az_rad) * np.sin(zen_rad)
    z = r * np.cos(zen_rad)
    return x, y, z



def binaural_convolve(audio_714, hrtf_l_list, hrtf_r_list):
    max_hrtf_len = max(len(h) for h in hrtf_l_list)
    n_samples = audio_714.shape[0] + max_hrtf_len - 1
    out = np.zeros((n_samples, 2))  # stereo

    for ch in range(audio_714.shape[1]):
        ch_audio = audio_714[:, ch]
        if ch == 3:  # LFE channel
            lfe = ch_audio * 0.5  # down-mix by 6 dB
            out[:len(lfe), 0] += lfe
            out[:len(lfe), 1] += lfe
        else:
            out[:, 0] += fftconvolve(ch_audio, hrtf_l_list[ch])
            out[:, 1] += fftconvolve(ch_audio, hrtf_r_list[ch])

    out /= np.max(np.abs(out) + 1e-6)
    return out


def main():
    input_path = "./test.wav"
    output_path = "./output_binaural.wav"
    sofa_path = "./HRIR_L2702.sofa"

    # 1. Define 7.1.4 loudspeaker angles
    ls_deg = [
        (-30, 0), (30, 0), (0, 0), (0, 0),
        (-110, 0), (110, 0), (-135, 0), (135, 0),
        (-45, 45), (45, 45), (-135, 45), (135, 45),
    ]
    # Convert negative azimuth angles to 0-360 degrees
    ls_deg = [(az+360 if az<0 else az, el) for az, el in ls_deg]

    # 2. sph2cart
    ls_cart_coords = np.array([spherical_to_cartesian(az, el, 1.0) for az, el in ls_deg])

    # 3. Load HRTF data and convert to cartesian coordinates
    hrtf_az, hrtf_el, hrtf = load_hrtf(sofa_path)
    hrtf_cart_coords = np.array([spherical_to_cartesian(az, el, 1.0) for az, el in zip(hrtf_az, hrtf_el)])

    # 4. Creates a 'hull' object
    hrtf_setup = decoder.LoudspeakerSetup(hrtf_cart_coords[:,0], hrtf_cart_coords[:,1], hrtf_cart_coords[:,2])

    # 5. VBAP 
    vbap_gains = decoder.vbap(ls_cart_coords, hrtf_setup)  # (12, N_hrtf)

    # 6. Interpolate HRTF using weights
    hrtf_l_list = []
    hrtf_r_list = []
    for ch in range(12):
        weights = vbap_gains[ch]
        nonzero_idx = np.where(weights != 0)[0]
        az, el = ls_deg[ch]
        print(f"Channel {ch}: angle (azimuth={az}, elevation={el})")
        print(f" 3 HRTF indices: {nonzero_idx}")
        print("  Corresponding angles and weights:")
        for idx in nonzero_idx:
            cur_hrtf_az = hrtf_az[idx]
            cur_hrtf_el = hrtf_el[idx]
            w = weights[idx]
            print(f"    HRTF idx {idx}: azimuth={cur_hrtf_az:.3f}, elevation={cur_hrtf_el:.3f}, weight={w:.3f}")
        # Weighted sum for left and right ear
        h_l = np.sum(hrtf[:,0,:] * weights[:,None], axis=0)
        h_r = np.sum(hrtf[:,1,:] * weights[:,None], axis=0)
        hrtf_l_list.append(h_l)
        hrtf_r_list.append(h_r)

    # 7. Convolution
    audio_714, sr = sf.read(input_path)
    binaural_audio = binaural_convolve(audio_714, hrtf_l_list, hrtf_r_list)

    # 8. LUFS normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(binaural_audio)
    binaural_audio = pyln.normalize.loudness(binaural_audio, loudness, -24.0)

    sf.write(output_path, binaural_audio, sr)

if __name__ == "__main__":
    main()

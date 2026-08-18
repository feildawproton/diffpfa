import torch
from diffpfa.io import CPHDReader
from diffpfa.algo.channel.geometry_channel import compute_look_vectors, get_image_plane_vectors
from diffpfa.algo.channel.patch.geometry_patch import compute_look_components
from diffpfa.algo.channel.collection_channel import compute_fasttime_frequencies
from diffpfa.constants import SPEED_OF_LIGHT

def inspect_dataset(path, name):
    print(f"\n--- {name} ---")
    reader = CPHDReader(path)
    meta = reader.get_metadata()
    ch_names = reader.get_channel_names()
    ch_data = reader.read_channel(ch_names[0])
    
    pvp = ch_data.pvp
    N_samples = ch_data.signal.shape[1]
    
    F_hz = compute_fasttime_frequencies(pvp, N_samples, ch_data.domain_type, device="cpu")
    print(f"F_hz start: {F_hz[0,0].item():.2f}, end: {F_hz[0,-1].item():.2f}")
    
    P_vecs = compute_look_vectors(pvp, device="cpu")
    uIAX, uIAY = get_image_plane_vectors(meta, "Ground", "cpu")
    cos_theta, sin_theta = compute_look_components(P_vecs, uIAX, uIAY)
    
    F_cpm = 2.0 * F_hz / SPEED_OF_LIGHT
    Kr = F_cpm * sin_theta.unsqueeze(1)
    
    k_start = Kr[:, 0]
    k_end = Kr[:, -1]
    k_step = (k_end - k_start) / (N_samples - 1)
    
    print(f"Kr[0] start: {k_start[0].item():.4f}, end: {k_end[0].item():.4f}, step: {k_step[0].item():.6f}")
    print(f"sin_theta[0]: {sin_theta[0].item():.4f}")

if __name__ == '__main__':
    good = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_CPHD.cphd"
    bad1 = "/home/feildaw/data/2023-09-13-21-18-21_UMBRA-06_CPHD.cphd"
    bad2 = "/home/feildaw/data/2023-10-04-02-03-26_UMBRA-04_CPHD.cphd"
    
    inspect_dataset(good, "GOOD (UMBRA-04)")
    inspect_dataset(bad1, "BAD (UMBRA-06)")
    inspect_dataset(bad2, "BAD (UMBRA-04-10-04)")

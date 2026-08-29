# Simulation Tools for diffpfa

This directory contains standalone, physically realistic SAR waveform and geometry simulators designed to evaluate `diffpfa` image formation performance under scenarios not easily accessible with standard single-channel public datasets.

## Available Simulations

### 1. Stepped-Chirp Multi-Channel Radar Simulation (`stepped_chirp_simulation.py`)
Simulates an ultra-wideband stepped-chirp radar synthesizing a high-bandwidth aperture across multiple discrete frequency channels transmitted with sequential inter-pulse delays ($\Delta \tau$) from an orbital platform ($v_{sat} = 7.5\text{ km/s}$).

**Physical Effects Modeled:**
- Inter-step time delays ($\Delta \tau = 150\,\mu\text{s}$) and platform displacement ($1.125\text{ m/step}$).
- Independent baseband downconversion at each subband local oscillator carrier ($f_{c,m}$).
- Motion compensation to SRP and differential carrier phase compensation ($\phi_{corr} = -2\pi (f_{c,0} - f_{c,m}) \tau$).
- Multi-channel coherent K-space accumulation through CZT-NUFFT gridding.
- Comparison against an ideal monolithic full-band reference pulse (evaluating complex coherence $|\rho|$, target localization, 3dB mainlobe resolution, and peak sidelobe ratios).

**Usage:**
```bash
python simulation/stepped_chirp_simulation.py
```

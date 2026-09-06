# Simulation Tools for diffpfa

This directory contains standalone, physically realistic SAR waveform and geometry simulators designed to evaluate `diffpfa` image formation performance under scenarios not easily accessible with standard single-channel public datasets.

## Available Simulations

### 1. Stepped-Chirp Multi-Channel Radar Simulation (`stepped_chirp_simulation.py`)
Simulates an ultra-wideband stepped-chirp radar synthesizing a high-bandwidth aperture across multiple discrete frequency channels transmitted with sequential inter-pulse delays ($\Delta \tau$) from an orbital platform ($v_{sat} = 7.5\text{ km/s}$).

**Physical Effects Modeled:**
- Inter-step time delays ($\Delta \tau = 137.3217\,\mu\text{s}$) and platform displacement ($1.03\text{ m/step}$).
- Independent baseband downconversion at each subband carrier ($f_{c,m}$).
- Motion compensation and phase referencing to Scene Reference Point (SRP) per CPHD DIDD §1.4.
- Multi-channel coherent K-space accumulation through CZT-NUFFT gridding.
- Comparison against an ideal monolithic full-band reference pulse (evaluating complex coherence $|\rho|$, target localization, 3dB mainlobe resolution, and peak sidelobe ratios).

**Usage:**
```bash
python simulation/stepped_chirp_simulation.py
```

### 2. Stepped-Chirp Receiver & CPHD Compensation Simulation (`stepped_chirp_receiver_simulation.py`)
Simulates the internal receiver hardware and CPHD producer compensation chain explicitly:
- Models transmitter carrier stepping, pulse-to-pulse local oscillator (LO) phase reset, and baseband mixing.
- Models the CPHD producer's phase compensation ($+j 2\pi f_x (2 R_{SRP}/c) + j \theta$), demonstrating that compliant CPHD data has identically zero phase at the SRP across all pulses.
- Evaluates subband coherence across arbitrary fractional delay offsets ($\Delta \tau$) and PRI jitter.
- Implements the single-subband inter-band phase diagnostic to inspect real multi-step CPHD data.

**Usage:**
```bash
python simulation/stepped_chirp_receiver_simulation.py
```


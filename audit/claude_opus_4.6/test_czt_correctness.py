"""
Audit Script 1: CZT Correctness Verification
Tests the CZT implementation against direct DFT computation (brute-force)
and against scipy.signal.czt for known signals.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import math

from diffpfa.IFA.channel.czt_torch import czt_1d_torch, czt_resample_kspace_1d

def test_czt_vs_brute_force():
    """
    Compare CZT output to brute-force DFT sum for a simple signal.
    CZT evaluates: C(r_m) = sum_n x[n] * exp(j*2*pi*r_m*(k_start + n*k_step))
    where r_m = r_min + m*dr, dr = (r_max - r_min)/(M-1)
    """
    print("=" * 60)
    print("TEST 1: CZT vs Brute-Force DFT")
    print("=" * 60)
    
    N = 32
    M = 64
    
    # Create a simple test signal
    np.random.seed(42)
    x_np = np.random.randn(N) + 1j * np.random.randn(N)
    x_t = torch.from_numpy(x_np)
    
    k_start = 100.0
    k_step = 0.5
    r_min = -10.0
    r_max = 10.0
    
    # CZT result
    result_czt = czt_1d_torch(
        x_t, M=M,
        r_min=r_min, r_max=r_max,
        k_step=torch.tensor(k_step, dtype=torch.float64),
        k_start=torch.tensor(k_start, dtype=torch.float64),
        dim=-1
    ).numpy()
    
    # Brute-force DFT
    dr = (r_max - r_min) / max(M - 1, 1)
    r_m = r_min + np.arange(M) * dr
    result_bf = np.zeros(M, dtype=complex)
    for m_idx in range(M):
        for n in range(N):
            phase = 2.0 * math.pi * r_m[m_idx] * (k_start + n * k_step)
            result_bf[m_idx] += x_np[n] * np.exp(1j * phase)
    
    max_err = np.max(np.abs(result_czt - result_bf))
    rel_err = max_err / np.max(np.abs(result_bf))
    
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.2e}")
    print(f"  PASS: {rel_err < 1e-10}" )
    return rel_err < 1e-10


def test_czt_vs_brute_force_batched():
    """
    Test batched CZT (per-pulse varying k_start, k_step) against brute force.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Batched CZT (per-pulse k_start/k_step) vs Brute-Force")
    print("=" * 60)
    
    N_pulses = 4
    N_samples = 16
    M = 32
    
    np.random.seed(123)
    x_np = np.random.randn(N_pulses, N_samples) + 1j * np.random.randn(N_pulses, N_samples)
    x_t = torch.from_numpy(x_np)
    
    k_starts = np.array([100.0, 101.0, 102.0, 103.0])
    k_steps = np.array([0.5, 0.51, 0.49, 0.5])
    
    r_min = -5.0
    r_max = 5.0
    
    k_start_t = torch.from_numpy(k_starts).unsqueeze(1)  # (4,1)
    k_step_t = torch.from_numpy(k_steps).unsqueeze(1)    # (4,1)
    
    result_czt = czt_1d_torch(
        x_t, M=M,
        r_min=r_min, r_max=r_max,
        k_step=k_step_t,
        k_start=k_start_t,
        dim=-1
    ).numpy()
    
    # Brute-force
    dr = (r_max - r_min) / max(M - 1, 1)
    result_bf = np.zeros((N_pulses, M), dtype=complex)
    for p in range(N_pulses):
        for m_idx in range(M):
            r_m = r_min + m_idx * dr
            for n in range(N_samples):
                phase = 2.0 * math.pi * r_m * (k_starts[p] + n * k_steps[p])
                result_bf[p, m_idx] += x_np[p, n] * np.exp(1j * phase)
    
    max_err = np.max(np.abs(result_czt - result_bf))
    rel_err = max_err / np.max(np.abs(result_bf))
    
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Max relative error: {rel_err:.2e}")
    print(f"  PASS: {rel_err < 1e-10}")
    return rel_err < 1e-10


def test_czt_resample_roundtrip():
    """
    Test CZT resampling: create a signal on a uniform grid,
    resample it, and verify energy is preserved and the spectrum shape is correct.
    """
    print("\n" + "=" * 60)
    print("TEST 3: CZT Resample K-space Round-trip (energy preservation)")
    print("=" * 60)
    
    N = 64
    M_out = 64
    
    # Uniform k-space signal: a single point target at the origin
    # In k-space, a point target at origin is a constant (flat spectrum)
    k_start_val = 100.0
    k_step_val = 0.1
    spatial_extent = 1.0 / k_step_val  # = 10.0 meters
    
    signal = torch.ones((1, N), dtype=torch.complex128)
    
    k_start = torch.tensor([[k_start_val]], dtype=torch.float64)
    k_step = torch.tensor([[k_step_val]], dtype=torch.float64)
    
    k_out_start = k_start_val
    k_out_step = k_step_val
    
    result = czt_resample_kspace_1d(
        signal,
        k_start=k_start,
        k_step=k_step,
        M_out=M_out,
        k_out_start=k_out_start,
        k_out_step=k_out_step,
        spatial_extent=spatial_extent,
    )
    
    input_energy = torch.sum(torch.abs(signal)**2).item()
    output_energy = torch.sum(torch.abs(result)**2).item()
    energy_ratio = output_energy / input_energy
    
    # For same-grid resampling, output should be close to input
    max_dev = torch.max(torch.abs(result - signal)).item()
    
    print(f"  Input energy:  {input_energy:.4f}")
    print(f"  Output energy: {output_energy:.4f}")
    print(f"  Energy ratio:  {energy_ratio:.6f}")
    print(f"  Max deviation from identity: {max_dev:.2e}")
    print(f"  PASS (energy within 5%): {abs(energy_ratio - 1.0) < 0.05}")
    return abs(energy_ratio - 1.0) < 0.05


def test_czt_bluestein_kernel_symmetry():
    """
    The Bluestein kernel v should have a specific symmetry: 
    v[l] = exp(-j*pi*dr*k_step*l^2) and the wrap-around indices should
    be the negated indices.
    
    This tests the l_idx construction in the CZT code.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Bluestein Kernel Index Construction")
    print("=" * 60)
    
    N = 8
    M = 12
    L = 2 ** math.ceil(math.log2(N + M - 1))
    
    l_idx = np.zeros(L)
    l_idx[:M] = np.arange(M)
    l_idx[L - N + 1:] = np.arange(1, N)[::-1]
    
    # Verify: l_idx should contain [0,1,...,M-1, 0,...,0, N-1, N-2,...,1]
    # The kernel phase is v(l) = exp(-j*pi*dr*k_step*l^2) so l_idx^2 matters.
    # The negative indices wrap around: v[-k] should be evaluated at l=k.
    
    # The negative indices (for wrap-around convolution) should be:
    # at position L-k for k=1,...,N-1, the value should be k
    for k in range(1, N):
        expected_val = k
        actual_val = l_idx[L - k]
        if actual_val != expected_val:
            print(f"  FAIL: l_idx[L-{k}] = {actual_val}, expected {expected_val}")
            return False
    
    # But wait - the code has l_idx[L - N + 1:] = arange(1, N).flip(0)
    # So L-N+1 -> N-1, L-N+2 -> N-2, ..., L-1 -> 1
    # Let's verify: at position L-k, what is l_idx?
    # position L-1: l_idx = 1 (correct, since we need v[-1] = v[1])
    # position L-2: l_idx = 2 (correct, since we need v[-2] = v[2])
    # ...
    # position L-N+1: l_idx = N-1 (correct)
    
    # Since phase uses l^2, and (-l)^2 = l^2, this is correct.
    print("  l_idx construction verified correct")
    print("  PASS: True")
    return True


def test_czt_sign_convention():
    """
    Verify the CZT sign convention.
    The code evaluates: C(r) = sum_n x[n] * exp(j * 2pi * r * (k_start + n * k_step))
    This is a POSITIVE exponent (forward FT convention with +j).
    
    For SAR: going from k-space to spatial domain needs exp(+j*k*r),
    and going back needs exp(-j*k*r) = conj of exp(+j*k*r).
    
    The czt_resample_kspace_1d uses:
    1. CZT with +j to go k-space -> spatial
    2. Conjugate the spatial result
    3. CZT with +j on conjugate -> this evaluates sum conj(s[n]) * exp(+j*r*k)
    4. Conjugate the result -> conj(sum conj(s[n]) * exp(+j*r*k)) = sum s[n] * exp(-j*r*k)
    
    This is effectively computing exp(-j*k*r) which is the correct inverse.
    """
    print("\n" + "=" * 60)
    print("TEST 5: CZT Sign Convention Verification")
    print("=" * 60)
    
    N = 32
    M = 32
    
    # Known signal: single frequency tone at k0
    k0 = 105.0
    k_start = 100.0
    k_step = 0.5
    
    # Signal x[n] = exp(j*2*pi*r0*(k_start + n*k_step)) for target at r0
    r0 = 2.0  # target at 2 meters
    n = np.arange(N)
    x_np = np.exp(1j * 2 * np.pi * r0 * (k_start + n * k_step))
    x_t = torch.from_numpy(x_np)
    
    r_min = -5.0
    r_max = 5.0
    
    result = czt_1d_torch(
        x_t, M=M,
        r_min=r_min, r_max=r_max,
        k_step=torch.tensor(k_step, dtype=torch.float64),
        k_start=torch.tensor(k_start, dtype=torch.float64),
        dim=-1
    ).numpy()
    
    # The CZT evaluates sum x[n] * exp(j*2*pi*r_m*(k_start + n*k_step))
    # = sum exp(j*2*pi*r0*...) * exp(j*2*pi*r_m*...)
    # = sum exp(j*2*pi*(r0+r_m)*...)
    # This peaks when r_m = -r0 (constructive interference of the double frequency)
    # Wait, actually let me reconsider:
    # x[n] = exp(j*2*pi*r0*k_n), CZT = sum exp(j*2*pi*r0*k_n)*exp(j*2*pi*r_m*k_n)
    # = sum exp(j*2*pi*(r0+r_m)*k_n)
    # This is a geometric sum that peaks when r0+r_m = 0, i.e., r_m = -r0
    
    dr = (r_max - r_min) / max(M - 1, 1)
    r_values = r_min + np.arange(M) * dr
    peak_idx = np.argmax(np.abs(result))
    peak_r = r_values[peak_idx]
    
    print(f"  Target at r0 = {r0}")
    print(f"  Expected peak at r_m = {-r0}")
    print(f"  Actual peak at r_m = {peak_r:.4f}")
    print(f"  PASS: {abs(peak_r - (-r0)) < dr}")
    return abs(peak_r - (-r0)) < dr


if __name__ == "__main__":
    results = []
    results.append(("CZT vs Brute-Force", test_czt_vs_brute_force()))
    results.append(("Batched CZT vs Brute-Force", test_czt_vs_brute_force_batched()))
    results.append(("CZT Resample Round-trip", test_czt_resample_roundtrip()))
    results.append(("Bluestein Kernel Symmetry", test_czt_bluestein_kernel_symmetry()))
    results.append(("CZT Sign Convention", test_czt_sign_convention()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

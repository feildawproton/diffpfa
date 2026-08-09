# Single-Channel GPU Optimization Records

This document tracks empirical performance metrics for intra-channel GPU optimization strategies on real Umbra CPHD datasets.

## 1. Intra-Channel PCIe Transfer & Compute Overlap (August 2026)

**Goal:** Hide the massive Host-to-Device (H2D) transfer cost of huge single-channel real datasets by streaming data to the GPU in chunks, overlapping the transfer of Chunk `N+1` with the cuFFT compute of Chunk `N`.

**Context:** The engine previously performed a synchronous, blocking memory copy of the entire multi-gigabyte channel payload to the GPU at the very beginning of the IFP (Image Formation Process) before any compute began. 

### Baseline (Full-Channel Synchronous Copy)
- **Dataset:** Real Umbra CPHD (`2023-11-14-03-38-20_UMBRA-04_CPHD.cphd`)
- **Mode:** `cztnufft` (Single Channel, sequential processing)
- **H2D Copy Time (`aten::to`):** ~25.6 seconds
- **GPU Idle/Sync Time (`cudaStreamSynchronize`):** ~24.4 seconds
- **cuFFT Kernel Compute (`cudaLaunchKernel`):** ~1.7 seconds
- **Total GPU Pipeline Duration:** ~33.7 seconds

*Finding:* For a massive real dataset, nearly 90% of the single-channel processing time was spent blocking the thread for the PCIe bus to transfer the entire payload. 

### Optimization Strategy
*(Pending Implementation and re-profiling)*
- Remove the initial full-tensor `.to(device)` call.
- Pin the CPU memory of the channel.
- Implement a dual-stream architecture (Compute Stream + Copy Stream) that asynchronously transfers the channel in sub-aperture batches matching the algorithm's CZT processing batches.
- **Expected Benefit:** Hide the 25.6 seconds of copy time behind the cuFFT execution.

### Post-Optimization Results (Chunked Streaming)
- **Dataset:** Real Umbra CPHD (`2023-11-14-03-38-20_UMBRA-04_CPHD.cphd`)
- **Mode:** `cztnufft` (Single Channel)
- **Total GPU Pipeline Duration:** **~14.6 seconds** (Down from ~33.7s)
- **H2D Copy Time (`aten::to`):** ~5.2 seconds (Down from ~25.6s)
- **GPU Idle/Sync Time (`cudaStreamSynchronize`):** ~4.4 seconds (Down from ~24.4s)

*Conclusion:* By streaming the multi-gigabyte channel payload over the PCIe bus in sub-aperture batches (overlapped with CZT cuFFT execution), we **cut the total algorithmic runtime by more than half (56% reduction)**. The GPU is no longer completely starved while waiting for a single monolithic transfer.

## 2. K-Space Sub-Channel Combination (August 2026)

**Goal:** Eliminate redundant 2D IFFT computations and their associated domain transitions when processing multiple sub-channels of the same polarization.

**Context:** Previously, every sub-channel was individually transformed all the way into the spatial domain (via a 2D IFFT and deconvolution) inside `pfa_engine.py`. Then, in `channel_combine.py`, these individual spatial images were phase-aligned via cross-correlation and summed. Because all sub-channels share the exact same spatial bounds and grid resolution, this resulted in N massive 2D IFFTs when only 1 was mathematically required.

### Baseline (Spatial Domain Combination)
- **Dataset:** Synthetic 4-channel dataset (4x 3000x3000 complex matrices)
- **Mode:** `cztnufft` (Multi-Channel)
- **Total Pipeline Wall Clock Time:** ~9.267 seconds

### Optimization Strategy
1. **Combine in K-Space:** Refactored `process_channel_czt_nufft` to optionally return the fully populated 2D K-space grid *before* applying the 2D IFFT. 
2. **In-place Alignment:** Sub-channels are now phase-aligned (via Parseval's theorem, doing cross-correlation directly on the K-space grids) and accumulated in-place in K-space.
3. **Single IFFT:** A single, massive 2D IFFT (and spatial deconvolution) is applied to the final combined K-space grid.

### Post-Optimization Results (K-Space Combination)
- **Dataset:** Synthetic 4-channel dataset (4x 3000x3000 complex matrices)
- **Mode:** `cztnufft` (Multi-Channel)
- **Total Pipeline Wall Clock Time:** **~6.807 seconds** (Down from ~9.267s)

*Conclusion:* Combining the channels in the frequency domain **reduced the multi-channel processing wall-clock time by ~27%**. By accumulating the K-space grids in-place prior to the final transform, we avoided the high algorithmic cost of redundant $O(N \log N)$ 2D IFFTs and sidestepped large intermediate GPU memory allocations.

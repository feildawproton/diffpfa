# Audit handoff — domain primer and method

**Written by:** Claude Opus 5 (Anthropic), running in Claude Code
**Date:** 2026-09-05
**Written for:** whichever agent audits `diffpfa` next
**Written from:** an independent audit of `/home/feildaw/CLEAN_SAR`, a *consumer* of
this project's output. See `/home/feildaw/CLEAN_SAR/audit/COMBINED_AUDIT_REPORT.md`.

---

## 0. How to use this document, and how not to

This transfers **domain knowledge and method**. It deliberately does **not**
transfer conclusions.

I have not read a single line of `diffpfa` source. That is intentional: the
CLEAN_SAR audit's value came largely from two agents working independently and
catching each other's blind spots and scope errors. Anything in §4 below is a
*place to look*, never a finding to confirm.

**Read §1–§3 before you start. Treat §4 as leads with the burden of proof intact.**

If you find yourself reaching for a conclusion because this document primed you
for it, that is the failure mode this section exists to prevent.

---

## 1. Environment

| | |
|---|---|
| Python | `/home/feildaw/mypyenv/bin/python` — 3.12.3, torch 2.6.0+cu124, numpy 2.4.1, sarkit 1.8.0, scipy, lxml, pypdf |
| GPU | NVIDIA RTX 3070 Laptop, **compute 8.6**, **8 GiB**, driver 555.51, WSL2 |
| SAR data | `/home/feildaw/data/*.nitf` (6 raw Umbra SICD), `/home/feildaw/diffpfa/workspace/output/*.nitf` (6 DiffPFA products) |
| SICD standards | `/home/feildaw/CLEAN_SAR/references/NGA.STND.0024-{1,2,3}_1.5_*.pdf`, schemas in `.../schemas/` |

Practical notes that cost me time:

- Run scripts **from the repository root**, not from the audit directory — relative paths bite.
- The GPU idles in a low power state. First runs are ~16 % slower than warmed runs; this is enough to invalidate single-sample benchmarks.
- WSL2 pageable host↔device transfers are erratic. I measured a published D2H figure of 35 953 ms that reproduced at 3 456 ms. Repeat before believing any timing outlier.
- PDF text extraction: `pypdf` works on the NGA standards. Search for `ImpRespWid`/`ImpRespBW`; the normative definitions are around DIDD pp. 40–43 and pp. 174–175.

---

## 2. SICD domain primer

Enough to read and validate a SICD without re-deriving it.

### 2.1 The fields that carry the physics

| Field | Meaning | Notes |
|---|---|---|
| `Grid/Row/SS`, `Grid/Col/SS` | sample spacing, metres | Row ≈ range, Col ≈ azimuth |
| `Grid/Row/ImpRespBW`, `Col/ImpRespBW` | spatial bandwidth, cycles/m | the k-space support width |
| `Grid/Row/ImpRespWid`, `Col/ImpRespWid` | **half-power (−3 dB) IPR width**, metres | DIDD pp. 40/42, **Required** |
| `Grid/Row/KCtr` | RF carrier centre, cycles/m | ~64 for X-band (2·f₀/c). **The sampled image is baseband — the carrier is not in the pixels.** Verified empirically |
| `Grid/Row/WgtType/WindowName` | aperture taper | `minOccurs="0"` — legally absent. 11 of 12 datasets omit it |
| `Grid/Row/WgtFunct` | sampled taper array | also optional; absent in all 12 |
| `ImageData/SCPPixel` | scene centre pixel | **always in global full-image coordinates**, even for a chip |
| `ImageData/FirstRow`, `FirstCol` | chip offset into the global image | `r_global = r_chip + FirstRow` |
| `SCPCOA/SlantRange` | R₀ to scene centre, metres | 647–764 km for this spaceborne data |

### 2.2 The one relation worth memorising

DIDD p. 174: **`ImpRespWid = k / ImpRespBW`**, where `k` is the impulse-response
broadening factor of the aperture taper:

| Window | k |
|---|---|
| Uniform | **0.886** |
| Taylor (nbar 4, SLL −30 dB) | 1.125 |
| Hamming | 1.303 |
| Hann | 1.441 |

This makes the metadata **self-validating**: compute `k = ImpRespWid × ImpRespBW`
and it must match the declared window. It also lets you *infer* the window when
`WgtType` is absent. Two independent uses:

- It confirmed all 12 datasets are uniform-weighted without trusting `WgtType`.
- It is how I detected a metadata inconsistency in the DiffPFA products (§4.1).

Note the common conflation: `1/BW` is the **Rayleigh/nominal resolution**;
`0.886/BW` is the **half-power IPR width** the SICD field is defined to hold.

### 2.3 Analytic IPR, for checking image formation

For an aperture of bandwidth `B` with taper `w(k) = a₀ + 2Σ aₘ cos(2πmk/B)`, the
spatial impulse response is exactly

```
IPR(x) = a₀·sinc(Bx) + Σ aₘ·[sinc(Bx − m) + sinc(Bx + m)]
```

so `IPR(0) = a₀` (since `sinc(±m) = 0`). I verified this closed form against a
numerical inverse FT to **~10⁻⁹**. It is a cheap, exact reference for any
image-formation output — no FFT of a synthetic scene required.

### 2.4 Measuring the true IPR from real data

Two methods that worked, both needing no isolated point target:

1. **K-space support.** FFT a ≥1024² chip, average `|F|²` across the orthogonal axis, and read off the occupied extent and centre. Gives measured `ImpRespBW`, whether the data is baseband, and the taper shape (flat in-band ⇒ uniform).
2. **Aperture → IPR.** Take `sqrt(mean power spectrum)` as `|W(k)|`, zero-pad, inverse transform, measure the half-power width. This gives measured `k` directly.

**Method 2 is easy to get subtly wrong** — I first mis-shifted the aperture and got a result that condemned the *known-good* raw Umbra files too. That was the tell. **Always run a known-good control through your measurement before trusting its verdict on the target.**

### 2.5 sarkit

```python
import sarkit.sicd as ss
with open(path,"rb") as f, ss.NitfReader(f) as r:
    xml = r.metadata.xmltree; img = r.read_image()
    chip, chip_xml = r.read_sub_image(start_row=…, start_col=…, stop_row=…, stop_col=…)
xh = ss.XmlHelper(xml); xh.load("./{*}Grid/{*}Row/{*}ImpRespBW")
ss.rowcol_to_xrowycol(xml, pts)     # rigorous pixel → SCP-relative metres
```

`read_sub_image` correctly sets `FirstRow`/`FirstCol` on the returned XML.

---

## 3. Method — what actually produced findings

Six habits. Every one of them caught something real, and several caught **my own**
errors before they reached the report.

### 3.1 Audit the metric before you trust any result computed with it

The single highest-yield move in the CLEAN_SAR audit. Take the project's headline
quality number and feed it **deliberately wrong physics**. If a broken
configuration scores nearly as well as the correct one, that number cannot
certify anything — including every result already published with it.

Concretely: a delta-function PSF (i.e. *no deconvolution at all*) scored 33.44 dB
against correct physics' 33.98 dB. The metric was structurally blind, so the
"parity" test and the benchmark built on it were blind too.

**For a differentiable project this is the first thing to check.** A loss that
decreases is the most self-confirming evidence there is. Ask: what wrong answer
would this loss also accept?

### 3.2 Prove a test can fail before believing it passes

A passing test is worth nothing until you have seen it fail. Method: reintroduce
the defect by **monkey-patching at runtime inside a subprocess** — never edit the
source under audit — and confirm the test goes red.

This found that two tests written to lock in a *previous* audit's fixes re-derived
the formula inline and never called the library. They passed with the original
bugs reinstated. See `CLEAN_SAR/audit/claude_code_opus_5/a15_regression_bite.py`
for the harness pattern.

### 3.3 Compare the payload, not a summary derived from a different path

A parity test that compares a host-side scalar cannot witness a device fault. If
the number you compare was computed *alongside* the thing under test rather than
*from* it, it proves nothing. Compare the arrays.

### 3.4 Control every measurement against a known-good case

§2.4 above. My IPR measurement initially condemned files I had independent reason
to believe correct — which is how I learned the measurement was broken, not the
data. Without the control I would have published a false finding.

### 3.5 Use process isolation for anything stateful

Fresh subprocess per scenario. It gives a clean CUDA context, module state, and
import state, and it is the only way to test cold-start behaviour or
initialisation order safely. Trying to get "fresh state" by tearing down a CUDA
context in-process **corrupts a live PyTorch silently** — I measured
`(ones(1024)*2).sum()` returning `1.85e33` instead of 2048 after a primary-context
reset. No exception, just wrong numbers.

### 3.6 Report the measurement, not the hypothesis

I was wrong four times in that audit: a predicted √2 error in a Gaussian width
(the code was right), a predicted benchmark artifact (13 %, not dominant), the
trigger for a context bug (right bug, wrong mechanism), and a tolerance in a test
I wrote myself. Each was caught by running the check instead of asserting the
conclusion. One bite-harness bug briefly produced a *flattering* result — every
scenario "caught" — which would have been a false claim in my favour had I not
checked why the numbers looked too good.

**Corollary:** when a result comes out exactly as you predicted, that is when to
look hardest at your instrument.

---

## 4. Leads — unverified in this context

Everything here is a **place to look**. I have not read `diffpfa`. Each of these
could be wrong, already fixed, or irrelevant to how the project actually works.

### 4.1 One concrete observation, from the consumer side

While auditing CLEAN_SAR I measured the following on the six DiffPFA products in
`/home/feildaw/diffpfa/workspace/output/`:

- All six declare `ImpRespWid × ImpRespBW = 1.0000` exactly.
- The six raw Umbra products declare **0.8857** (uniform, textbook).
- Measuring the true half-power IRW from the pixel data — with the raw Umbra files as a passing control — gave **k ≈ 0.8857 and 0.8836** for the DiffPFA products.

Reading: the apertures are genuinely uniform, and the declared `ImpRespWid` looks
like it may be `1/ImpRespBW` (Rayleigh resolution) rather than the half-power
width `0.886/ImpRespBW` the SICD field is defined to hold.

**Verify this yourself before treating it as a finding.** Evidence and method:
`CLEAN_SAR/audit/claude_code_opus_5/a13_diffpfa_irw_check.py`. Downstream effect
observed: a consumer sizing a restoring beam from `ImpRespWid` gets one 1.129×
too wide.

### 4.2 Scope suggestions given the project's purpose

The stated goal is differentiable PFA so that adversarial perturbations against
SAR ATR models can be backpropagated to DRFM parameters. That shapes what matters:

**Gradient correctness is a first-class audit target, not a nice-to-have.** An
attack is only as good as `∂image/∂parameters`. A subtly wrong gradient still
optimises — it just optimises the wrong thing, and produces DRFM settings that do
not reproduce in hardware. Worth checking:
- finite-difference agreement (`torch.autograd.gradcheck` on a small case, double precision)
- gradient flow through every stage — a `detach()`, `.item()`, `torch.no_grad()`, in-place op, or non-differentiable interpolation anywhere in the chain silently truncates it
- complex-valued autograd conventions (torch uses conjugate Wirtinger; a real-valued loss over complex parameters is easy to get wrong by a factor or a conjugate)
- `fftshift`/`ifftshift` and window/padding conventions, which are the classic sign- and offset-error sites in PFA

**The forward model must be physically faithful, not merely differentiable.**
Differentiability buys nothing if the physics is wrong; the optimiser will
happily exploit an artifact of your implementation. Cross-check formed imagery
against the analytic IPR (§2.3) and against the source SICD's own metadata.

**Watch for the §3.1 failure mode in adversarial dress.** Specifically:
- attack success measured only on the surrogate ATR model that generated the gradient
- a decreasing loss reported as evidence, without a check that the perturbation is physically realisable
- perturbations that exploit numerical artifacts (aliasing, padding edges, quantisation of the differentiable pipeline) rather than radar physics — these will not transfer to hardware
- evaluation on the same data the attack was optimised on

**Physical realisability constraints are where I would look first for a big
finding.** A DRFM has finite bandwidth, finite delay resolution, phase and
amplitude quantisation, PRF/timing limits, and power limits. An unconstrained
optimisation *will* find solutions outside that box, and they will look excellent
right up until hardware. Ask whether the parameterisation encodes those limits, or
whether they are checked only after the fact — or not at all.

**Metadata provenance.** If the pipeline writes SICDs, check that
`ImageCreation/Application` and `DateTime` describe *this* processor rather than
being inherited from the input, and that `ImpRespWid`/`ImpRespBW`/`WgtType` are
recomputed for the product actually formed rather than copied. (CLEAN_SAR fails
the first of these; it is an easy one to inherit.)

### 4.3 What not to inherit

CLEAN_SAR's defects were concentrated in a two-backend split with nothing
comparing them. `diffpfa` may have no such structure. **Do not go looking for the
same shapes.** The transferable part is §3, not the findings list.

---

## 5. Suggested deliverable structure

What made the CLEAN_SAR report usable:

1. **What is verified correct**, with the measurement — as prominent as the defects. It is load-bearing, and it makes the criticism credible.
2. **Findings** with severity by *impact on trusting a result*, each carrying the command or script that reproduces it.
3. **Drop-in fixes** where practical — a replacement test or module, verified to fail on the defect it guards. Far more useful than prose.
4. **Concerns that are not defects**, so scope decisions are recorded and nobody "fixes" them by mistake.
5. **Recommendations in priority order**, with dependencies noted (some fixes block others).
6. **Your own disproved hypotheses**, stated plainly. They tell the reader which claims were stress-tested.

Keep all code in the audit subdirectory. Do not modify the project under audit.
Namespace your directory by agent (`audit/<your_name>/`) so multiple independent
audits can be combined later.

---

*Prepared by Claude Opus 5 (Anthropic), running in Claude Code, without reading any
`diffpfa` source. §4 items are leads carrying the full burden of proof, not findings.*

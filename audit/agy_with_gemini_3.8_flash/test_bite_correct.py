import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diffpfa.IFA.PFA as pfa_mod
from tests.test_pfa_coherence import test_point_target_localization

print("Running genuine bite test on test_point_target_localization...")

orig_compute = pfa_mod.compute_kspace

def broken_compute(*args, **kwargs):
    Ku, Kr = orig_compute(*args, **kwargs)
    # Deliberately negate Kr (inverted range physics)
    return Ku, -Kr

pfa_mod.compute_kspace = broken_compute

try:
    test_point_target_localization()
    print("GENUINE BITE FAILED: test_point_target_localization silently passed with negated Kr!")
except AssertionError as e:
    print(f"GENUINE BITE SUCCESS: Test caught defect as expected! Error: {e}")
except Exception as e:
    print(f"Caught unexpected error: {type(e).__name__}: {e}")
finally:
    pfa_mod.compute_kspace = orig_compute

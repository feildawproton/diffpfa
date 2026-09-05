import pytest
import subprocess
import sys

# Test whether the test suite fails when intentional defects are introduced (Test bite check)
print("Testing bite harness on test_point_target_localization...")

code_corrupt_phase = """
import numpy as np
import diffpfa.IFA.kspace as kmod

# Monkey patch kspace to corrupt look vectors
orig_compute = kmod.compute_kspace
def broken_compute(*args, **kwargs):
    Ku, Kr = orig_compute(*args, **kwargs)
    # Negate Kr (wrong range direction)
    return Ku, -Kr
kmod.compute_kspace = broken_compute

from tests.test_pfa_coherence import test_point_target_localization
try:
    test_point_target_localization()
    print("BITE TEST FAILED: Test passed with inverted Kr!")
except Exception as e:
    print(f"BITE TEST SUCCESS: Test caught defect: {type(e).__name__}: {e}")
"""

res = subprocess.run([sys.executable, "-c", code_corrupt_phase], capture_output=True, text=True)
print(res.stdout)
print(res.stderr)

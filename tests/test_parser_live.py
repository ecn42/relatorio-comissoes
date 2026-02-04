import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Mock streamlit to allow importing the page file
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit'].session_state = {}

# Add path to allow import
sys.path.append(os.getcwd())

# Import using importlib because filename has spaces/numbers (though importlib isn't strictly needed if we add to path and use valid name, but here filename is tricky)
import importlib.util
file_path = "pages/22_Rating_To_Fitch.py"
spec = importlib.util.spec_from_file_location("rating_to_fitch", file_path)
module = importlib.util.module_from_spec(spec)
sys.modules["rating_to_fitch"] = module
spec.loader.exec_module(module)

translate_to_fitch = module.translate_to_fitch

test_cases = [
    "brAAA-sf",
    "brAAA (sf)",
    "brAAA (sf)",
    "brAA+ (sf)",
    "AAsf(bra)",
    "AAsf(bra)",
    "AAAsf(bra)",
    "AAAsf(bra)",
    "AAA(exp)sf(bra)",
    "AAA(exp)sf(bra)",
    "AAA.br (sf)",
    "AAA.br (sf)",
    "AAA",
    "AAA",
    "AAA",
    "AAA",
    "AA+.br (sf)",
    "AA-sf(bra)",
    "AA-sf(bra)",
    "AA-sf(bra)",
    "AA-sf (bra)",
    "AA-sf (bra)"
]

print(f"{'Input':<20} | {'Result':<10} | {'Status'}")
print("-" * 45)

passed = 0
for case in test_cases:
    result = translate_to_fitch(case)
    # Check if result is not None (meaning it matched something)
    status = "OK" if result else "FAIL"
    if result: passed += 1
    print(f"{case:<20} | {str(result):<10} | {status}")

print(f"\nPassed: {passed}/{len(test_cases)}")

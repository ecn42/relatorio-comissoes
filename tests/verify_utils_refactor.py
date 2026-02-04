import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Mock streamlit to allow importing the page file
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit'].session_state = {}

# Make sure we can import from utils
sys.path.append(os.getcwd())

# 1. Verify utils import works directly
try:
    from utils.rating_utils import translate_to_fitch
    print("Successfully imported translate_to_fitch from utils.")
except ImportError as e:
    print(f"Failed to import from utils: {e}")
    sys.exit(1)

# 2. Verify pages/22 imports it correctly
import importlib.util
file_path = "pages/22_Rating_To_Fitch.py"
spec = importlib.util.spec_from_file_location("rating_to_fitch", file_path)
module = importlib.util.module_from_spec(spec)
sys.modules["rating_to_fitch"] = module
spec.loader.exec_module(module)

# Check if the module has translate_to_fitch (imported)
if hasattr(module, 'translate_to_fitch'):
    print("pages/22_Rating_To_Fitch.py has translate_to_fitch available.")
else:
    print("pages/22_Rating_To_Fitch.py missing translate_to_fitch.")

# 3. Test logic again with complex case
case = "brAAA-sf"
res = translate_to_fitch(case)
print(f"Test case '{case}' -> '{res}' (Expected: AAA)")

if res == "AAA":
    print("Verification PASSED")
else:
    print("Verification FAILED")

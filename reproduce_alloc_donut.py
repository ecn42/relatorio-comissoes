import pandas as pd
import numpy as np
import sys
import os
import importlib.util
from unittest.mock import MagicMock

# Import the file
file_path = '/home/eduardo/relatorio-comissoes/pages/7_factsheet.py'
spec = importlib.util.spec_from_file_location("factsheet", file_path)
factsheet = importlib.util.module_from_spec(spec)
sys.modules["factsheet"] = factsheet
sys.modules['streamlit'] = MagicMock()
spec.loader.exec_module(factsheet)

categorize_tp_ativo = factsheet.categorize_tp_ativo
get_alloc_donut = factsheet.get_alloc_donut

print("--- Testing categorize_tp_ativo (Expect Pass-Through) ---")
# Since user disabled categorization, output should equal input
test_cases = [
    "CDB do banco X",
    "LETRA FINANCEIRA Y",
    "TESOURO IPCA",
    "FII ABC",
    "OUTRA COISA"
]

for inp in test_cases:
    res = categorize_tp_ativo(inp)
    print(f"'{inp}' -> '{res}'")
    assert res == inp, f"Failed: Expected '{inp}' but got '{res}'. Logic should be pass-through."

print("\n--- Testing get_alloc_donut ---")
# We need > 6 distinct items to trigger "Outros" logic
data_many = {
    "TP_ATIVO": [
        "ITEM 1",
        "ITEM 2",
        "ITEM 3",
        "ITEM 4",
        "ITEM 5",
        "ITEM 6",
        "ITEM 7"  # This will be properly kept separate until donut logic forces "Outros"
    ],
    "PCT_CARTEIRA": [10, 10, 10, 10, 10, 10, 40]
}
df_many = pd.DataFrame(data_many)

res_many = get_alloc_donut(df_many)
print("\nResult with 7 items:")
print(res_many)
print("Columns:", res_many.columns.tolist())

if "CATEGORIA" in res_many.columns:
    print("BUG: 'CATEGORIA' column found!")
    sys.exit(1)
if "TP_APLIC" not in res_many.columns:
    print("BUG: 'TP_APLIC' column missing!")
    sys.exit(1)

# Check for NaNs
if res_many.isna().any().any():
    print("BUG: NaNs found in result!")
    print(res_many[res_many.isna().any(axis=1)])
    sys.exit(1)

# Check that we have valid results
assert len(res_many) > 0

print("\nSUCCESS: All tests passed!")

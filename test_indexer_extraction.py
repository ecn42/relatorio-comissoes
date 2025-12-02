#!/usr/bin/env python3
"""Test the indexer extraction from raw JSON."""

import json
import sys

# Test data - CDB with FIXED_INCOME_PREFIXED
cdb_json = {
    "referenceDate": "2025-11-07",
    "broker": {"id": "59281253000123", "name": "BTG PACTUAL DTVM"},
    "security": {
        "id": 80484322,
        "name": "CDB_PRE_16.12_10664513000150_2025_01_06_2027_01_06",
        "type": "CORPORATE_BONDS_CDB",
        "assetClass": "FIXED_INCOME",
        "description": "CDB 16.12% a.a. BANCO AGIBANK 06/01/2027",
        "issuer": "10664513000150",
        "classifications": [
            {"id": "FIXED_INCOME", "name": "Renda Fixa", "level": "CLASSIFICATION_L0", "parentId": None},
            {"id": "FIXED_INCOME_PREFIXED", "name": "Prefixado", "level": "CLASSIFICATION_L1", "parentId": "FIXED_INCOME"},
            {"id": "FIXED_INCOME_PREFIXED_FINANCIAL_INSTITUTION", "name": "Instituição Financeira", "level": "CLASSIFICATION_L2", "parentId": "FIXED_INCOME_PREFIXED"}
        ]
    },
    "marketValue": 113464.9925074752,
    "currency": "BRL"
}

# Test data - CRA with FIXED_INCOME_INTEREST_INDEXED
cra_json = {
    "referenceDate": "2025-11-26",
    "broker": {"id": "59281253000123", "name": "BTG PACTUAL DTVM"},
    "security": {
        "id": 74923522,
        "isin": "BRRBRACRA470",
        "name": "CRA BCO BTG PACTUAL NOV/2033 CRA02300NRL",
        "type": "CORPORATE_BONDS_CRA",
        "assetClass": "FIXED_INCOME",
        "issuer": "02773542000122",
        "classifications": [
            {"id": "FIXED_INCOME", "name": "Renda Fixa", "level": "CLASSIFICATION_L0", "parentId": None},
            {"id": "FIXED_INCOME_INTEREST_INDEXED", "name": "Indexado a Juros", "level": "CLASSIFICATION_L1", "parentId": "FIXED_INCOME"},
            {"id": "FIXED_INCOME_INTEREST_INDEXED_PRIVATE_CREDIT", "name": "Crédito Privado", "level": "CLASSIFICATION_L2", "parentId": "FIXED_INCOME_INTEREST_INDEXED"}
        ]
    },
    "marketValue": 79808.70255,
    "currency": "BRL"
}

# Test data - CRI with FIXED_INCOME_INTEREST_INDEXED
cri_json = {
    "referenceDate": "2025-11-26",
    "broker": {"id": "59281253000123", "name": "BTG PACTUAL DTVM"},
    "security": {
        "id": 63686765,
        "name": "CRI ALLPARK MAR/2029 22L1723201",
        "type": "CORPORATE_BONDS_CRI",
        "assetClass": "FIXED_INCOME",
        "issuer": "02773542000122",
        "classifications": [
            {"id": "FIXED_INCOME", "name": "Renda Fixa", "level": "CLASSIFICATION_L0", "parentId": None},
            {"id": "FIXED_INCOME_INTEREST_INDEXED", "name": "Indexado a Juros", "level": "CLASSIFICATION_L1", "parentId": "FIXED_INCOME"},
            {"id": "FIXED_INCOME_INTEREST_INDEXED_PRIVATE_CREDIT", "name": "Crédito Privado", "level": "CLASSIFICATION_L2", "parentId": "FIXED_INCOME_INTEREST_INDEXED"}
        ]
    },
    "marketValue": 1027.723427,
    "currency": "BRL"
}

# Test data - DEBENTURE with FIXED_INCOME_INFLATION_INDEXED
debenture_json = {
    "referenceDate": "2025-11-26",
    "broker": {"id": "59281253000123", "name": "BTG PACTUAL DTVM"},
    "security": {
        "id": 62289925,
        "isin": "BRENEVDBS0K8",
        "name": "ENEV19",
        "type": "CORPORATE_BONDS_DEBENTURE",
        "assetClass": "FIXED_INCOME",
        "classifications": [
            {"id": "FIXED_INCOME", "name": "Renda Fixa", "level": "CLASSIFICATION_L0", "parentId": None},
            {"id": "FIXED_INCOME_INFLATION_INDEXED", "name": "Indexado a Inflação", "level": "CLASSIFICATION_L1", "parentId": "FIXED_INCOME"},
            {"id": "FIXED_INCOME_INFLATION_INDEXED_PRIVATE_CREDIT", "name": "Crédito Privado", "level": "CLASSIFICATION_L2", "parentId": "FIXED_INCOME_INFLATION_INDEXED"}
        ]
    },
    "marketValue": 1157.254239,
    "currency": "BRL"
}

def _extract_indexer_from_raw_json(raw_json_str):
    """
    Extract indexer from raw JSON's CLASSIFICATION_L2 parentId field.
    Maps:
      - FIXED_INCOME_PREFIXED -> PRE
      - FIXED_INCOME_INFLATION_INDEXED -> POS IPCA
      - FIXED_INCOME_INTEREST_INDEXED -> POS CDI
    """
    if not raw_json_str:
        return None
    
    try:
        data = json.loads(raw_json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    
    # Navigate to classifications list
    sec = data.get("security", {})
    if not isinstance(sec, dict):
        return None
    
    classifications = sec.get("classifications", [])
    if not isinstance(classifications, list):
        return None
    
    # Find CLASSIFICATION_L2 entry
    for c in classifications:
        if not isinstance(c, dict):
            continue
        if c.get("level") == "CLASSIFICATION_L2":
            parent_id = c.get("parentId")
            if parent_id == "FIXED_INCOME_PREFIXED":
                return "PRE"
            elif parent_id == "FIXED_INCOME_INFLATION_INDEXED":
                return "POS IPCA"
            elif parent_id == "FIXED_INCOME_INTEREST_INDEXED":
                return "POS CDI"
    
    return None

# Test cases
test_cases = [
    ("CDB with FIXED_INCOME_PREFIXED", cdb_json, "PRE"),
    ("CRA with FIXED_INCOME_INTEREST_INDEXED", cra_json, "POS CDI"),
    ("CRI with FIXED_INCOME_INTEREST_INDEXED", cri_json, "POS CDI"),
    ("DEBENTURE with FIXED_INCOME_INFLATION_INDEXED", debenture_json, "POS IPCA"),
]

print("Testing indexer extraction from raw JSON\n")
print("=" * 60)

for test_name, test_data, expected in test_cases:
    result = _extract_indexer_from_raw_json(json.dumps(test_data, ensure_ascii=False))
    status = "✓ PASS" if result == expected else "✗ FAIL"
    print(f"\nTest: {test_name}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")
    print(f"  Status:   {status}")

print("\n" + "=" * 60)
print("All tests completed!")


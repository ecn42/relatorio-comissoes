import re
import pandas as pd
from typing import Optional

def translate_to_fitch(rating: str) -> Optional[str]:
    """
    Translate Moody's or S&P rating to Fitch format, or normalize existing Fitch rating.
    Handles complex strings like 'brAAA-sf', 'AAA(exp)sf(bra)'.
    
    Args:
        rating: The rating string (case-insensitive)
        
    Returns:
        Fitch rating string (uppercase) if matched, None if unmatched
    """
    if not rating or pd.isna(rating):
        return None
    
    rating_str = str(rating).strip()
    if not rating_str or rating_str == "-":
        return None
    
    # Pre-cleaning
    cleaned = rating_str.upper()
    
    # Remove "BR" prefix if it starts the string
    cleaned = re.sub(r'^BR', '', cleaned)
    
    # Remove noise patterns
    # (sf), (exp), (bra), .br and variations without parens if strict checking allows
    # We use explicit list to avoid over-cleaning
    noise_patterns = [
        r'\(SF\)', r'SF', 
        r'\(EXP\)', r'EXP',
        r'\(BRA\)', r'BRA',
        r'\.BR', 
    ]
    
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned)
        
    cleaned = cleaned.strip()
    
    # Define mappings: Moody's and S&P to Fitch (case-insensitive)
    # Note: All inputs are converted to uppercase before matching
    # Moody's mappings (Moody's uses Aaa, Aa1-Aa3, A1-A3, Baa1-Baa3, Ba1-Ba3, B1-B3, Caa1-Caa3, Ca, C)
    moodys_to_fitch = {
        "AAA": "AAA",  # Aaa uppercased
        "AA1": "AA+",  # Aa1 uppercased
        "AA2": "AA",   # Aa2 uppercased
        "AA3": "AA-",  # Aa3 uppercased
        "A1": "A+",
        "A2": "A",
        "A3": "A-",
        "BAA1": "BBB+",  # Baa1 uppercased
        "BAA2": "BBB",   # Baa2 uppercased
        "BAA3": "BBB-",  # Baa3 uppercased
        "BA1": "BB+",    # Ba1 uppercased
        "BA2": "BB",     # Ba2 uppercased
        "BA3": "BB-",    # Ba3 uppercased
        "B1": "B+",
        "B2": "B",
        "B3": "B-",
        "CAA1": "CCC",   # Caa1 uppercased
        "CAA2": "CCC",   # Caa2 uppercased
        "CAA3": "CCC",   # Caa3 uppercased
        "CA": "CCC",     # Ca uppercased
        "C": "DDD",      # Moody's C maps to DDD (default) for Fitch
    }
    
    # S&P mappings
    sp_to_fitch = {
        "AAA": "AAA",
        "AA+": "AA+",
        "AA": "AA",
        "AA-": "AA-",
        "A+": "A+",
        "A": "A",
        "A-": "A-",
        "BBB+": "BBB+",
        "BBB": "BBB",
        "BBB-": "BBB-",
        "BB+": "BB+",
        "BB": "BB",
        "BB-": "BB-",
        "B+": "B+",
        "B": "B",
        "B-": "B-",
        "CCC+": "CCC",
        "CCC": "CCC",
        "CCC-": "CCC",
        "CC": "CCC",
        "C": "DDD",  # S&P C maps to DDD
        "D": "DDD",  # S&P D maps to DDD
    }
    
    # Valid Fitch ratings (for normalization only)
    valid_fitch = {
        "AAA", "AA+", "AA", "AA-",
        "A+", "A", "A-",
        "BBB+", "BBB", "BBB-",
        "BB+", "BB", "BB-",
        "B+", "B", "B-",
        "CCC", "DDD", "DD", "D"
    }
    
    def lookup(r):
        if r in moodys_to_fitch:
            return moodys_to_fitch[r]
        if r in sp_to_fitch:
            return sp_to_fitch[r]
        if r in valid_fitch:
            return r
        return None

    # 1. Try exact match of cleaned string
    res = lookup(cleaned)
    if res:
        return res
        
    # 2. If no match and ends with hyphen (e.g. from brAAA-sf -> AAA-), try stripping it
    # But only if the stripped version is valid. 
    # (Note: AA- is valid and matched in step 1. AAA- is invalid and fails step 1. So stripping - leads to AAA, which is valid)
    if cleaned.endswith('-'):
        cleaned_no_dash = cleaned.rstrip('-')
        res = lookup(cleaned_no_dash)
        if res:
            return res
            
    # 3. Try original raw uppercase just in case (fallback)
    rating_upper = rating_str.upper()
    res = lookup(rating_upper)
    if res:
        return res
    
    # No match found
    return None

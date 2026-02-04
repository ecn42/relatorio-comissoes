import re

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

def clean_rating(rating_str):
    if not rating_str or rating_str == "-":
        return None
    
    # Uppercase
    cleaned = rating_str.upper()
    
    # Remove common prefixes/suffixes/noise
    # Order matters: more specific first
    noise_patterns = [
        r'\(SF\)', r'SF', 
        r'\(EXP\)', r'EXP',
        r'\(BRA\)', r'BRA', # risky? "BRA" might be part of something? 'br' prefix handled separately?
        r'\.BR', 
    ]
    
    # 1. Strip 'br' prefix specifically if it starts the string
    cleaned = re.sub(r'^BR', '', cleaned)
    
    # 2. Remove noise patterns
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned)
        
    # 3. Cleanup loose hyphens or parens that might be left: e.g. "AAA-" from "AAA-sf" if sf removed
    # But "AA-" needs the hyphen. 
    # "brAAA-sf" -> "AAA-" (if sf removed). 
    # Valid ratings usually end with + or -, or nothing.
    # If we have a hyphen at the end that is NOT part of a standard rating like AA-, we should check?
    # Fitch ratings: AAA, AA+, AA, AA-, etc.
    
    cleaned = cleaned.strip()
    
    # Some extra cleanup for "-sf" where the hyphen was a separator
    # If the remaining string ends with "-", check if it's a valid rating like "AA-".
    # If it is "AAA-", that's invalid (AAA doesn't take minus? Actually maybe not, Fitch Aaa is AAA).
    # S&P/Fitch: AAA, AA+, AA, AA-, A+, A, A-, BBB+...
    # AAA- is not standard. AA- is.
    
    # Logic: if matches a known rating exactly, good.
    # If not, try stripping trailing punctuation?
    
    return cleaned

print(f"{'Input':<20} | {'Cleaned':<10}")
print("-" * 35)
for case in test_cases:
    cleaned = clean_rating(case)
    print(f"{case:<20} | {cleaned:<10}")


import pandas as pd
import unicodedata

# Copied helpers
def strip_accents(text: str) -> str:
    if text is None:
        return ""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(c)
    )

def is_missing(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if s == "":
        return True
    s_up = strip_accents(s).upper()
    return s_up in {"#N/D", "N/D", "NA", "NAN", "NONE"}

def parse_br_number(val):
    if is_missing(val):
        return None
    if isinstance(val, (int, float)):
        try:
            if pd.isna(val):
                return None
        except:
            pass
        return float(val)
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return None
    s = s.replace("R$", "").replace("%", "").replace(" ", "").replace("\u00a0","").strip()
    if not s: return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return None

# The function under test
def generate_performance_text(perf_df: pd.DataFrame, benchmark_name: str) -> str:
    if perf_df.empty:
        return ""
    
    df = perf_df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    
    serie_col = df.columns[0]
    
    def get_val(row_idx, col_name):
        if row_idx < 0 or row_idx >= len(df):
            return None
        if col_name not in df.columns:
            return None
        val = df.iloc[row_idx][col_name]
        return parse_br_number(val)

    fund_idx = 0
    cdi_idx = -1
    ibov_idx = -1
    vol_idx = -1
    sharpe_idx = -1
    
    for idx, row in df.iterrows():
        s = str(row[serie_col]).upper()
        if "VOLATILIDADE" in s or "VOL" in s:
            vol_idx = idx
        elif "SHARPE" in s:
            sharpe_idx = idx
        elif "CDI" in s:
            cdi_idx = idx
        elif "IBOV" in s:
            ibov_idx = idx
            
    bench_idx = -1
    is_cdi = True
    
    if benchmark_name:
        bname = benchmark_name.upper()
        if "IBOV" in bname:
            bench_idx = ibov_idx
            is_cdi = False
        elif "CDI" in bname:
            bench_idx = cdi_idx
            is_cdi = True
    
    if bench_idx == -1:
        if cdi_idx != -1:
            bench_idx = cdi_idx
            is_cdi = True
        elif ibov_idx != -1:
            bench_idx = ibov_idx
            is_cdi = False

    ytd_val = get_val(fund_idx, "YTD")
    r36_val = get_val(fund_idx, "36M")
    vol36_val = get_val(vol_idx, "36M") if vol_idx != -1 else None
    sharpe36_val = get_val(sharpe_idx, "36M") if sharpe_idx != -1 else None

    def fmt_pct(v):
        if v is None: return "-"
        return f"{v*100:.1f}%".replace(".", ",")

    def fmt_num(v):
        if v is None: return "-"
        return f"{v:.2f}".replace(".", ",")
        
    parts = []
    
    if ytd_val is not None:
        p1 = f"O fundo rendeu {fmt_pct(ytd_val)} neste ano"
        if bench_idx != -1:
            bench_ytd = get_val(bench_idx, "YTD")
            if bench_ytd is not None:
                if is_cdi:
                    if abs(bench_ytd) > 1e-9:
                        pct_of_cdi = (ytd_val / bench_ytd)
                        p1 += f", {pct_of_cdi*100:.0f}% do CDI"
                else:
                    diff = ytd_val - bench_ytd
                    direction = "acima" if diff >= 0 else "abaixo"
                    p1 += f", {fmt_pct(abs(diff))} {direction} do IBOV"
        p1 += "."
        parts.append(p1)
    
    if r36_val is not None:
        p2 = f"Nos últimos 36 meses, a estratégia acumulou retorno de {fmt_pct(r36_val)}"
        stats = []
        if vol36_val is not None:
            stats.append(f"com volatilidade anualizada de {fmt_pct(vol36_val)}")
        if sharpe36_val is not None:
            stats.append(f"índice Sharpe de {fmt_num(sharpe36_val)}")
        if stats:
            p2 += ", " + " e ".join(stats)
        p2 += "."
        parts.append(p2)
        
    fluff = (
        "O portfólio segue posicionado de forma consistente com a tese de investimentos de longo prazo, "
        "buscando capturar oportunidades assimétricas e preservação de capital em diferentes cenários de mercado."
    )
    parts.append(fluff)
    
    return " ".join(parts)

# Test Cases
print("Test 1: Normal CDI")
df1 = pd.DataFrame({
    'SERIE': ['Fundo', 'CDI', 'Volatilidade', 'Sharpe'],
    'YTD': [0.05, 0.045, None, None],
    '36M': [0.30, 0.25, 0.05, 1.2],
})
print(generate_performance_text(df1, 'CDI'))

print("\nTest 2: Normal IBOV")
df2 = pd.DataFrame({
    'SERIE': ['Fundo', 'IBOV', 'Vol', 'Sharpe'],
    'YTD': [0.10, 0.08, None, None],
    '36M': [0.40, 0.35, 0.15, 0.8],
})
print(generate_performance_text(df2, 'IBOV'))

print("\nTest 3: Missing 36M")
df3 = pd.DataFrame({
    'SERIE': ['Fundo', 'CDI'],
    'YTD': [0.02, 0.02],
    # No 36M col
})
print(generate_performance_text(df3, 'CDI'))


import re
import unicodedata
from datetime import datetime

import pandas as pd
import streamlit as st

# -------------------- Streamlit setup --------------------
st.set_page_config(page_title="Stock Guide Extractor", layout="wide")
st.title("📊 Stock Guide Data Extractor")

# -------------------- helpers for extra sheets --------------------


def _norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\n", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _clean_colname(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("..", ".")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(":")
    s = s.replace("Cap.", "Cap")
    s = s.replace("Market Cap.", "Market Cap")
    return s


def parse_br_number(x: object) -> float | object:
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if s in ("", "nan", "na", "n/a", "n.a", "n.a.", "-", "–", "—"):
        return pd.NA

    # Check for parentheses indicating negative
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]

    s = s.replace("r$", "")
    s = s.replace("mn", "")
    s = s.replace("mi", "")
    s = s.replace("x", "")
    s = s.replace("%", "")
    s = re.sub(r"[^0-9\-.,]", "", s)
    if s == "" or s in ("-", "–", "—"):
        return pd.NA
    s = s.replace(".", "").replace(",", ".")
    try:
        val = float(s)
        if is_negative:
            val = -val
        return val
    except Exception:
        return pd.NA


def _build_combined_headers(row1: list, row2: list) -> list[str]:
    # Forward-fill parent headers and combine with child headers.
    parents = []
    cur_parent = ""
    for v in row1:
        p = "" if pd.isna(v) else str(v).strip()
        if p and p.lower() != "nan":
            cur_parent = p
        parents.append(cur_parent)

    combined = []
    for i, (p, c) in enumerate(zip(parents, row2)):
        p = _clean_colname(str(p).strip() if pd.notna(p) else "")
        c = _clean_colname(str(c).strip() if pd.notna(c) else "")
        if c in ("Empresa", "Ticker"):
            combined.append(c)
        elif c:
            if p and p not in ("", "nan"):
                combined.append(_clean_colname(f"{p} {c}"))
            else:
                combined.append(_clean_colname(c))
        elif p:
            combined.append(_clean_colname(p))
        else:
            combined.append(f"Col_{i}")
    if len(combined) > 0 and combined[0].lower().startswith("empresa"):
        combined[0] = "Empresa"
    if len(combined) > 1 and combined[1].lower().startswith("ticker"):
        combined[1] = "Ticker"
    return combined


def _find_section_row_indices(
    df_raw: pd.DataFrame, keyword_patterns: list[str]
) -> list[int]:
    # Returns list of row indices whose concatenated text matches any pattern.
    hits = []
    pats = [re.compile(p, re.IGNORECASE) for p in keyword_patterns]
    for idx, row in df_raw.iterrows():
        row_text = " ".join(
            [str(v) for v in row.values if pd.notna(v) and str(v).strip() != ""]
        )
        nt = _norm_text(row_text)
        if any(p.search(nt) for p in pats):
            hits.append(idx)
    return hits


def _slice_section_table(
    df_raw: pd.DataFrame, header1_idx: int, header2_idx: int
) -> pd.DataFrame:
    # Determine where data ends: blank row or next section header-like row.
    start = header2_idx + 1
    end = start
    n = len(df_raw)
    stop_patterns = re.compile(
        r"\b(2\.\s*volume|3\.\s*multipl|4\.\s*dados|5\.\s*destaques|"
        r"empresa\s*\|\s*ticker)\b",
        re.IGNORECASE,
    )
    while end < n:
        row = df_raw.iloc[end]
        all_na = all(pd.isna(v) or str(v).strip() == "" for v in row.values)
        row_text = " ".join(
            [str(v) for v in row.values if pd.notna(v) and str(v).strip() != ""]
        )
        nt = _norm_text(row_text)
        if all_na:
            break
        if stop_patterns.search(nt):
            break
        end += 1

    header_row_1 = df_raw.iloc[header1_idx].tolist()
    header_row_2 = df_raw.iloc[header2_idx].tolist()
    cols = _build_combined_headers(header_row_1, header_row_2)

    df_section = df_raw.iloc[start:end].copy()
    df_section = df_section.iloc[:, : len(cols)]
    df_section.columns = cols
    df_section = df_section.dropna(axis=1, how="all")
    keep_mask = []
    for _, r in df_section.iterrows():
        e = str(r.get("Empresa", "")).strip().lower()
        t = str(r.get("Ticker", "")).strip().lower()
        keep_mask.append(not ((e in ("", "nan")) and (t in ("", "nan"))))
    df_section = df_section.loc[keep_mask]
    return df_section


def _numericify_columns(
    df: pd.DataFrame, percent_as_number: bool = True
) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ("Empresa", "Ticker", "Setor"):
            continue
        out[c] = out[c].apply(parse_br_number)
    return out


def _parse_section_by_keywords(
    df_raw: pd.DataFrame, keywords: list[str]
) -> tuple[pd.DataFrame, int, int] | tuple[None, None, None]:
    idxs = _find_section_row_indices(df_raw, keywords)
    if not idxs:
        return None, None, None
    header1_idx = idxs[0]
    header2_idx = header1_idx + 1
    if header2_idx >= len(df_raw):
        return None, None, None
    maybe_hdr2 = df_raw.iloc[header2_idx].tolist()
    if not any(_norm_text(x) == "empresa" for x in maybe_hdr2) or not any(
        _norm_text(x) == "ticker" for x in maybe_hdr2
    ):
        header2_idx = header1_idx + 2
        if header2_idx >= len(df_raw):
            return None, None, None
    try:
        section_df = _slice_section_table(df_raw, header1_idx, header2_idx)
    except Exception:
        return None, None, None
    return section_df, header1_idx, header2_idx


def process_sector_sheet(
    df_raw: pd.DataFrame, setor: str
) -> dict[str, pd.DataFrame]:
    out = {
        "fundamentals": pd.DataFrame(),
        "highlights": pd.DataFrame(),
    }

    # 4. Dados fundamentalistas
    df_fund, _, _ = _parse_section_by_keywords(
        df_raw, [r"\b4\.\s*dados\s*fundament", r"\bfundamental"]
    )
    if df_fund is not None and not df_fund.empty:
        df_fund["Setor"] = setor
        cols_keep = ["Setor", "Empresa", "Ticker"] + [
            c for c in df_fund.columns if c not in ("Setor", "Empresa", "Ticker")
        ]
        out["fundamentals"] = _numericify_columns(df_fund[cols_keep])

    # 5. Destaques financeiros
    df_high, _, _ = _parse_section_by_keywords(
        df_raw, [r"\b5\.\s*destaques\s*financeiros", r"\bdestaques\s*financeiros"]
    )
    if df_high is not None and not df_high.empty:
        df_high["Setor"] = setor
        cols_keep = ["Setor", "Empresa", "Ticker"] + [
            c for c in df_high.columns if c not in ("Setor", "Empresa", "Ticker")
        ]
        out["highlights"] = _numericify_columns(df_high[cols_keep])

    return out


def parse_extra_sheets(
    file_like, sheet_names: list[str]
) -> dict[str, pd.DataFrame]:
    # Read only available sheets once via ExcelFile for efficiency.
    try:
        xl = pd.ExcelFile(file_like)
        available = set(xl.sheet_names)
    except Exception:
        return {
            "fundamentals": pd.DataFrame(),
            "highlights": pd.DataFrame(),
        }

    funds = []
    highs = []

    for name in sheet_names:
        if name not in available:
            continue
        try:
            df_raw = xl.parse(sheet_name=name, header=None)
        except Exception:
            continue
        sect = process_sector_sheet(df_raw, setor=name)
        if not sect["fundamentals"].empty:
            funds.append(sect["fundamentals"])
        if not sect["highlights"].empty:
            highs.append(sect["highlights"])

    out = {
        "fundamentals": pd.concat(funds, ignore_index=True)
        if funds
        else pd.DataFrame(),
        "highlights": pd.concat(highs, ignore_index=True)
        if highs
        else pd.DataFrame(),
    }

    for k in out:
        if not out[k].empty:
            out[k] = out[k].drop_duplicates()

    return out


def safe_merge_by_ticker(
    base: pd.DataFrame, extra: pd.DataFrame, how: str = "left"
) -> pd.DataFrame:
    # Merge extra columns by Ticker without overwriting existing columns.
    if extra.empty or "Ticker" not in extra.columns:
        return base
    extra_cols = [
        c for c in extra.columns if c not in ("Setor", "Empresa", "Ticker")
    ]
    if not extra_cols:
        return base
    add = extra[["Ticker"] + extra_cols].drop_duplicates("Ticker")
    rename_map = {}
    for c in extra_cols:
        if c in base.columns:
            rename_map[c] = f"{c} (extra)"
    if rename_map:
        add = add.rename(columns=rename_map)
    return base.merge(add, on="Ticker", how=how)


# -------------------- Comparator Utilities (Theme + Helpers + HTML) --------------------

# Theme constants (aligned with your factsheet styling)
BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
LIGHT_GRAY = "#F5F5F5"
BLOCK_BG = "#F8F8F8"
TABLE_BORDER = "#E0D5CA"
TEXT_DARK = "#333333"

import html as _html  # noqa: E402


def sanitize_filename(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    s = re.sub(r"[\s_]+", "_", s).strip("_ ")
    return s or "comparador"


def _format_ptbr_number(value: float, decimals: int = 2, trim: bool = True) -> str:
    try:
        s = f"{float(value):,.{decimals}f}"
    except Exception:
        return "-"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    if trim and "," in s:
        s = s.rstrip("0").rstrip(",")
    return s


def _is_percent_col(colname: str) -> bool:
    n = str(colname).strip().lower()
    tokens = [
        "%",
        "yld",
        "yield",
        "margem",
        "margin",
        "roe",
        "roa",
        "roic",
        "growth",
        "cagr",
        "var.",
        "variação",
    ]
    return any(t in n for t in tokens)


def _is_multiple_col(colname: str) -> bool:
    n = str(colname).strip().lower()
    tokens = [
        "p/l",
        "p\\l",
        "p/e",
        "p\\e",
        "p/b",
        "p\\b",
        "p/vp",
        "ev/ebitda",
        "ev ebitda",
        "net debt/ebitda",
        "dívida líquida/ebitda",
        "divida liquida/ebitda",
    ]
    return any(t in n for t in tokens)


def _best_is_higher(colname: str) -> bool:
    # Múltiplos menores são melhores; retornos/margens/market cap maiores são melhores
    if _is_multiple_col(colname):
        return False
    n = str(colname).strip().lower()
    higher_good_tokens = [
        "market cap",
        "volume",
        "vol.",
        "vol 3m",
        "roe",
        "roa",
        "roic",
        "yld",
        "yield",
        "dy",
        "margem",
        "margin",
        "crescimento",
        "growth",
        "cagr",
    ]
    return any(t in n for t in higher_good_tokens) or _is_percent_col(colname)


def _to_numeric_ptbr_series(s: pd.Series) -> pd.Series:
    return s.apply(parse_br_number)


def _collect_metric_candidates(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    exclude = {"empresa", "nome", "ticker", "setor", "stock guide", "rating"}
    candidates = []
    for c in df.columns:
        if str(c).strip().lower() in exclude:
            continue
        ser = _to_numeric_ptbr_series(df[c])
        non_na = ser.notna().sum()
        if non_na >= max(3, int(0.15 * len(ser))):
            candidates.append(c)

    priority = [
        "Market Cap (R$ mn)",
        "Volume",
        "Volume 3M",
        "Dividend",
        "Dividend Yield",
        "Yld",
        "Yield",
        "P/L",
        "P/E",
        "EV/EBITDA",
        "P/BV",
        "P/VP",
        "ROE",
        "ROIC",
        "ROA",
        "Margem",
        "Margin",
        "Net Debt/EBITDA",
    ]

    def score(col: str) -> tuple[int, str]:
        n = str(col)
        p = 100
        for i, key in enumerate(priority):
            if key.lower() in n.lower():
                p = i
                break
        return (p, n.lower())

    candidates = sorted(candidates, key=score)
    return candidates


def _html_escape(x: object) -> str:
    return _html.escape("" if pd.isna(x) else str(x))


def generate_comparator_html(
    df_source: pd.DataFrame,
    tickers: list[str],
    metrics: list[str],
    title: str = "Stock Comparator",
    subtitle: str = "",
) -> str:
    if df_source.empty or not tickers or not metrics:
        return "<p>No data to render.</p>"

    base_cols = []
    for candidate in ["Nome", "Empresa"]:
        if candidate in df_source.columns:
            base_cols.append(candidate)
            break
    for candidate in ["Ticker"]:
        if candidate in df_source.columns:
            base_cols.append(candidate)
            break
    if "Setor" in df_source.columns:
        base_cols.append("Setor")

    cols = base_cols + metrics
    df = df_source.copy()
    df = df[df["Ticker"].astype(str).isin(tickers)].copy()
    df = df.sort_values(by=["Ticker"])
    df = df.drop_duplicates(subset=["Ticker"], keep="first")

    num_map: dict[str, pd.Series] = {}
    rank_map: dict[str, pd.Series] = {}
    for m in metrics:
        ser = _to_numeric_ptbr_series(df[m])
        num_map[m] = ser
        if ser.notna().sum() >= 2:
            if _best_is_higher(m):
                rank = ser.rank(ascending=False, method="min")
            else:
                rank = ser.rank(ascending=True, method="min")
        else:
            rank = pd.Series([pd.NA] * len(ser), index=ser.index)
        rank_map[m] = rank

    def cell_bg_css(metric: str, idx) -> str:
        r = rank_map[metric].get(idx, pd.NA)
        if pd.isna(r):
            return ""
        total = rank_map[metric].dropna().max()
        if not total or pd.isna(total):
            return ""
        val = (r - 1) / max(total - 1, 1)
        alpha = 0.22 * (1.0 - float(val))
        return (
            f"background: rgba(130,81,32,{alpha:.2f}); "
            f"color: {TEXT_DARK};"
        )

    rows_html = []
    for i, row in df.iterrows():
        tds = []
        for c in base_cols:
            tds.append(f"<td>{_html_escape(row.get(c, ''))}</td>")
        for m in metrics:
            raw_v = num_map[m].get(i, pd.NA)
            if not pd.isna(raw_v) and _is_percent_col(m):
                txt = _format_ptbr_number(raw_v * 100.0, 2, trim=True) + "%"
            elif not pd.isna(raw_v):
                decimals = 2
                if (
                    "market cap" in str(m).lower()
                    or "volume" in str(m).lower()
                ):
                    decimals = 0
                txt = _format_ptbr_number(raw_v, decimals=decimals, trim=True)
            else:
                raw_txt = row.get(m, "")
                txt = (
                    "-"
                    if (pd.isna(raw_txt) or str(raw_txt).strip() == "")
                    else _html_escape(raw_txt)
                )
            style = cell_bg_css(m, i)
            tds.append(f'<td style="{style}">{txt}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    ths = []
    for j, c in enumerate(base_cols):
        align = "left"
        ths.append(
            f'<th class="sticky" data-col="{_html_escape(c)}" '
            f'data-type="text" style="text-align:{align}">'
            f"{_html_escape(c)}</th>"
        )
    for m in metrics:
        dtype = "num"
        ths.append(
            f'<th class="sticky sortable" data-col="{_html_escape(m)}" '
            f'data-type="{dtype}">{_html_escape(m)}</th>'
        )

    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
      :root {{
        --brand: {BRAND_BROWN};
        --brand-dark: {BRAND_BROWN_DARK};
        --bg: {BLOCK_BG};
        --text: {TEXT_DARK};
        --tbl-border: {TABLE_BORDER};
        --light: {LIGHT_GRAY};
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
        background: #f0f0f0;
        color: var(--text);
        margin: 0;
        padding: 0;
      }}
      .page {{
        max-width: 1200px;
        margin: 12px auto;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-radius: 6px;
        overflow: hidden;
      }}
      .header {{
        background: var(--brand);
        color: #fff;
        padding: 10px 14px;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
      }}
      .title {{ font-size: 16px; font-weight: 700; letter-spacing: 0.3px; }}
      .subtitle {{ font-size: 11px; opacity: 0.9; }}
      .controls {{
        background: var(--bg);
        padding: 8px 14px;
        display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
        border-bottom: 1px solid var(--tbl-border);
      }}
      .controls .label {{ font-size: 12px; font-weight: 600; color: #555; }}
      .controls input[type="text"] {{
        padding: 6px 8px; border: 1px solid var(--tbl-border);
        border-radius: 4px; min-width: 220px; font-size: 12px;
      }}
      .table-wrap {{ width: 100%; overflow-x: auto; }}
      table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 12px;
      }}
      thead th {{
        position: sticky; top: 0; z-index: 2;
        background: var(--brand); color: #fff;
        text-align: center; padding: 8px 6px; font-weight: 700;
        border-bottom: 1px solid var(--tbl-border);
        white-space: nowrap;
      }}
      thead th.sticky:first-child {{ left: 0; }}
      tbody td {{
        padding: 8px 6px; border-bottom: 1px solid var(--tbl-border);
        text-align: center; white-space: nowrap;
      }}
      tbody tr:nth-child(even) td {{ background: var(--light); }}
      tbody td:first-child, thead th:first-child {{ text-align: left; }}
      th.sortable {{ cursor: pointer; }}
      .footer {{
        padding: 8px 14px; color: #666; font-size: 11px;
        border-top: 1px solid var(--tbl-border);
        background: #fff;
      }}
    </style>
    """

    js = """
    <script>
      (function() {
        const tbl = document.getElementById('cmp-table');
        const q = document.getElementById('cmp-search');
        let sortState = { col: -1, asc: true };

        function getCellNum(td) {
          const txt = td.textContent.trim()
                         .replace(/\\./g, '')
                         .replace(',', '.')
                         .replace('%', '');
          const v = parseFloat(txt);
          return isNaN(v) ? NaN : v;
        }

        function sortBy(thIdx, isNum) {
          const tbody = tbl.tBodies[0];
          const rows = Array.from(tbody.rows);
          const asc = (sortState.col === thIdx) ? !sortState.asc : true;
          rows.sort((a, b) => {
            const A = a.cells[thIdx];
            const B = b.cells[thIdx];
            if (isNum) {
              const va = getCellNum(A);
              const vb = getCellNum(B);
              if (isNaN(va) && isNaN(vb)) return 0;
              if (isNaN(va)) return 1;
              if (isNaN(vb)) return -1;
              return asc ? va - vb : vb - va;
            } else {
              const sa = A.textContent.trim().toLowerCase();
              const sb = B.textContent.trim().toLowerCase();
              if (sa < sb) return asc ? -1 : 1;
              if (sa > sb) return asc ? 1 : -1;
              return 0;
            }
          });
          rows.forEach(r => tbody.appendChild(r));
          sortState = { col: thIdx, asc };
        }

        const ths = Array.from(tbl.tHead.rows[0].cells);
        ths.forEach((th, idx) => {
          th.addEventListener('click', () => {
            const type = th.getAttribute('data-type') || 'text';
            sortBy(idx, type === 'num');
          });
        });

        q.addEventListener('input', () => {
          const term = q.value.trim().toLowerCase();
          const tbody = tbl.tBodies[0];
          Array.from(tbody.rows).forEach(row => {
            const txt = row.textContent.toLowerCase();
            row.style.display = txt.indexOf(term) >= 0 ? '' : 'none';
          });
        });
      })();
    </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>{_html_escape(title)}</title>
      {css}
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div class="title">{_html_escape(title)}</div>
          <div class="subtitle">{_html_escape(subtitle)}</div>
        </div>
        <div class="controls">
          <div class="label">Buscar</div>
          <input id="cmp-search" type="text" placeholder="Empresa, Ticker, Setor..."/>
        </div>
        <div class="table-wrap">
          <table id="cmp-table">
            <thead>
              <tr>
                {''.join(ths)}
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
        <div class="footer">
          Gerado automaticamente — comparador offline em HTML.
        </div>
      </div>
      {js}
    </body>
    </html>
    """
    return html


def generate_simple_table_html(
    df: pd.DataFrame, title: str = "Tabela de Dados", subtitle: str = ""
) -> str:
    import html as _html

    if df.empty:
        return "<p>Sem dados para exibir.</p>"

    cols = [str(c) for c in df.columns]
    ths = "".join(
        f'<th class="sticky sortable" data-type="text">{_html.escape(c)}</th>'
        for c in cols
    )

    rows_html = []
    for _, row in df.iterrows():
        tds = []
        for c in cols:
            v = row.get(c, "")
            txt = "" if pd.isna(v) else str(v)
            tds.append(f"<td>{_html.escape(txt)}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
      :root {{
        --brand: {BRAND_BROWN};
        --brand-dark: {BRAND_BROWN_DARK};
        --bg: {BLOCK_BG};
        --text: {TEXT_DARK};
        --tbl-border: {TABLE_BORDER};
        --light: {LIGHT_GRAY};
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
        background: #f0f0f0;
        color: var(--text);
        margin: 0;
        padding: 0;
      }}
      .page {{
        max-width: 1200px;
        margin: 12px auto;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-radius: 6px;
        overflow: hidden;
      }}
      .header {{
        background: var(--brand);
        color: #fff;
        padding: 10px 14px;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
      }}
      .title {{ font-size: 16px; font-weight: 700; letter-spacing: 0.3px; }}
      .subtitle {{ font-size: 11px; opacity: 0.9; }}
      .controls {{
        background: var(--bg);
        padding: 8px 14px;
        display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
        border-bottom: 1px solid var(--tbl-border);
      }}
      .controls .label {{ font-size: 12px; font-weight: 600; color: #555; }}
      .controls input[type="text"] {{
        padding: 6px 8px; border: 1px solid var(--tbl-border);
        border-radius: 4px; min-width: 220px; font-size: 12px;
      }}
      .table-wrap {{ width: 100%; overflow-x: auto; }}
      table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 12px;
      }}
      thead th {{
        position: sticky; top: 0; z-index: 2;
        background: var(--brand); color: #fff;
        text-align: center; padding: 8px 6px; font-weight: 700;
        border-bottom: 1px solid var(--tbl-border);
        white-space: nowrap;
      }}
      tbody td {{
        padding: 8px 6px; border-bottom: 1px solid var(--tbl-border);
        text-align: left; white-space: nowrap;
      }}
      tbody tr:nth-child(even) td {{ background: var(--light); }}
      th.sortable {{ cursor: pointer; }}
      .footer {{
        padding: 8px 14px; color: #666; font-size: 11px;
        border-top: 1px solid var(--tbl-border);
        background: #fff;
      }}
    </style>
    """

    js = """
    <script>
      (function() {
        const tbl = document.getElementById('simple-table');
        const q = document.getElementById('simple-search');
        let sortState = { col: -1, asc: true };

        function getCellNum(td) {
          const txt = td.textContent.trim()
                         .replace(/\\./g, '')
                         .replace(',', '.')
                         .replace('%', '');
          const v = parseFloat(txt);
          return isNaN(v) ? NaN : v;
        }

        function sortBy(thIdx) {
          const tbody = tbl.tBodies[0];
          const rows = Array.from(tbody.rows);
          const asc = (sortState.col === thIdx) ? !sortState.asc : true;
          rows.sort((a, b) => {
            const A = a.cells[thIdx];
            const B = b.cells[thIdx];
            const va = getCellNum(A);
            const vb = getCellNum(B);
            const aNum = !isNaN(va);
            const bNum = !isNaN(vb);
            if (aNum && bNum) return asc ? va - vb : vb - va;
            const sa = A.textContent.trim().toLowerCase();
            const sb = B.textContent.trim().toLowerCase();
            if (sa < sb) return asc ? -1 : 1;
            if (sa > sb) return asc ? 1 : -1;
            return 0;
          });
          rows.forEach(r => tbody.appendChild(r));
          sortState = { col: thIdx, asc };
        }

        const ths = Array.from(tbl.tHead.rows[0].cells);
        ths.forEach((th, idx) => {
          th.addEventListener('click', () => sortBy(idx));
        });

        q.addEventListener('input', () => {
          const term = q.value.trim().toLowerCase();
          const tbody = tbl.tBodies[0];
          Array.from(tbody.rows).forEach(row => {
            const txt = row.textContent.toLowerCase();
            row.style.display = txt.indexOf(term) >= 0 ? '' : 'none';
          });
        });
      })();
    </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>{_html.escape(title)}</title>
      {css}
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div class="title">{_html.escape(title)}</div>
          <div class="subtitle">{_html.escape(subtitle)}</div>
        </div>
        <div class="controls">
          <div class="label">Buscar</div>
          <input id="simple-search" type="text" placeholder="Buscar em toda a tabela..."/>
        </div>
        <div class="table-wrap">
          <table id="simple-table">
            <thead>
              <tr>
                {ths}
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
        <div class="footer">
          Tabela HTML gerada automaticamente (offline).
        </div>
      </div>
      {js}
    </body>
    </html>
    """
    return html


# -------------------- NEW: Two-Company Comparison Graphs HTML --------------------


def generate_two_company_comparison_html(
    df_source: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    metrics: list[str],
    title: str = "Comparação de Empresas",
    subtitle: str = "",
) -> str:
    """
    Generate an HTML page with bar charts and a radar chart comparing two companies.
    Uses Chart.js via CDN for rendering.
    """
    import json

    if df_source.empty or not ticker1 or not ticker2 or not metrics:
        return "<p>Dados insuficientes para gerar comparação.</p>"

    df = df_source.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"].isin([ticker1, ticker2])].drop_duplicates(
        subset=["Ticker"], keep="first"
    )

    if len(df) < 2:
        return "<p>Uma ou ambas as empresas não foram encontradas.</p>"

    row1 = df[df["Ticker"] == ticker1].iloc[0]
    row2 = df[df["Ticker"] == ticker2].iloc[0]

    # Get company names
    name_col = (
        "Nome"
        if "Nome" in df.columns
        else ("Empresa" if "Empresa" in df.columns else None)
    )
    name1 = str(row1.get(name_col, ticker1)) if name_col else ticker1
    name2 = str(row2.get(name_col, ticker2)) if name_col else ticker2
    if pd.isna(name1) or name1.lower() == "nan":
        name1 = ticker1
    if pd.isna(name2) or name2.lower() == "nan":
        name2 = ticker2

    # Extract numeric values for each metric
    values1 = []
    values2 = []
    labels = []
    for m in metrics:
        v1 = parse_br_number(row1.get(m, pd.NA))
        v2 = parse_br_number(row2.get(m, pd.NA))
        # Convert percent columns
        if _is_percent_col(m):
            if not pd.isna(v1):
                v1 = v1 * 100
            if not pd.isna(v2):
                v2 = v2 * 100
        values1.append(None if pd.isna(v1) else round(float(v1), 2))
        values2.append(None if pd.isna(v2) else round(float(v2), 2))
        labels.append(m)

    # For radar chart, we need normalized values (0-100 scale)
    radar_values1 = []
    radar_values2 = []
    for i, m in enumerate(metrics):
        v1 = values1[i]
        v2 = values2[i]
        if v1 is None and v2 is None:
            radar_values1.append(0)
            radar_values2.append(0)
        elif v1 is None:
            radar_values1.append(0)
            radar_values2.append(100)
        elif v2 is None:
            radar_values1.append(100)
            radar_values2.append(0)
        else:
            max_val = max(abs(v1), abs(v2), 0.001)
            # For multiples (lower is better), invert the scale
            if _is_multiple_col(m):
                # Invert: smaller value gets higher score
                radar_values1.append(
                    round(100 * (1 - abs(v1) / (abs(v1) + abs(v2) + 0.001)), 1)
                )
                radar_values2.append(
                    round(100 * (1 - abs(v2) / (abs(v1) + abs(v2) + 0.001)), 1)
                )
            else:
                radar_values1.append(round(100 * abs(v1) / max_val, 1))
                radar_values2.append(round(100 * abs(v2) / max_val, 1))

    # Build summary table data
    table_rows_html = ""
    for i, m in enumerate(labels):
        v1_str = _format_ptbr_number(values1[i], 2) if values1[i] is not None else "-"
        v2_str = _format_ptbr_number(values2[i], 2) if values2[i] is not None else "-"
        if _is_percent_col(m):
            if values1[i] is not None:
                v1_str += "%"
            if values2[i] is not None:
                v2_str += "%"
        # Highlight winner
        highlight1 = ""
        highlight2 = ""
        if values1[i] is not None and values2[i] is not None:
            if _is_multiple_col(m):
                # Lower is better
                if values1[i] < values2[i]:
                    highlight1 = "font-weight: 700; color: #2E7D32;"
                elif values2[i] < values1[i]:
                    highlight2 = "font-weight: 700; color: #2E7D32;"
            else:
                # Higher is better
                if values1[i] > values2[i]:
                    highlight1 = "font-weight: 700; color: #2E7D32;"
                elif values2[i] > values1[i]:
                    highlight2 = "font-weight: 700; color: #2E7D32;"
        table_rows_html += f"""
        <tr>
          <td>{_html_escape(m)}</td>
          <td style="{highlight1}">{v1_str}</td>
          <td style="{highlight2}">{v2_str}</td>
        </tr>
        """

    # JSON data for charts
    labels_json = json.dumps(labels)
    values1_json = json.dumps(values1)
    values2_json = json.dumps(values2)
    radar1_json = json.dumps(radar_values1)
    radar2_json = json.dumps(radar_values2)
    name1_json = json.dumps(name1)
    name2_json = json.dumps(name2)

    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
      :root {{
        --brand: {BRAND_BROWN};
        --brand-dark: {BRAND_BROWN_DARK};
        --bg: {BLOCK_BG};
        --text: {TEXT_DARK};
        --tbl-border: {TABLE_BORDER};
        --light: {LIGHT_GRAY};
        --company1: #825120;
        --company2: #7F8C8D;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
        background: #f0f0f0;
        color: var(--text);
        margin: 0;
        padding: 0;
      }}
      .page {{
        max-width: 1200px;
        margin: 12px auto;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-radius: 6px;
        overflow: hidden;
      }}
      .header {{
        background: var(--brand);
        color: #fff;
        padding: 14px 20px;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
      }}
      .title {{ font-size: 18px; font-weight: 700; letter-spacing: 0.3px; }}
      .subtitle {{ font-size: 12px; opacity: 0.9; }}
      .content {{
        padding: 20px;
      }}
      .companies-header {{
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-bottom: 20px;
        flex-wrap: wrap;
      }}
      .company-badge {{
        padding: 10px 24px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        color: #fff;
      }}
      .company-badge.c1 {{ background: var(--company1); }}
      .company-badge.c2 {{ background: var(--company2); }}
      .charts-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 30px;
      }}
      @media (max-width: 800px) {{
        .charts-grid {{ grid-template-columns: 1fr; }}
      }}
      .chart-container {{
        background: var(--light);
        border-radius: 8px;
        padding: 16px;
        position: relative;
        height: 350px;
      }}
      .chart-container h3 {{
        margin: 0 0 12px 0;
        font-size: 14px;
        color: var(--brand);
        text-align: center;
      }}
      .chart-wrapper {{
        position: relative;
        width: 100%;
        height: calc(100% - 30px);
      }}
      .radar-section {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
      }}
      .radar-container {{
        background: var(--light);
        border-radius: 8px;
        padding: 20px;
        max-width: 500px;
        width: 100%;
      }}
      .radar-container h3 {{
        margin: 0 0 12px 0;
        font-size: 14px;
        color: var(--brand);
        text-align: center;
      }}
      .radar-wrapper {{
        position: relative;
        width: 100%;
        height: 350px;
      }}
      .summary-section {{
        margin-top: 20px;
      }}
      .summary-section h3 {{
        font-size: 14px;
        color: var(--brand);
        margin-bottom: 10px;
      }}
      table.summary {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      table.summary th, table.summary td {{
        padding: 10px 12px;
        border: 1px solid var(--tbl-border);
        text-align: center;
      }}
      table.summary th {{
        background: var(--brand);
        color: #fff;
        font-weight: 600;
      }}
      table.summary td:first-child {{
        text-align: left;
        font-weight: 500;
      }}
      table.summary tr:nth-child(even) td {{
        background: var(--light);
      }}
      .footer {{
        padding: 12px 20px;
        color: #666;
        font-size: 11px;
        border-top: 1px solid var(--tbl-border);
        background: #fff;
        text-align: center;
      }}
    </style>
    """

    js = f"""
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', function() {{
        const labels = {labels_json};
        const values1 = {values1_json};
        const values2 = {values2_json};
        const radar1 = {radar1_json};
        const radar2 = {radar2_json};
        const name1 = {name1_json};
        const name2 = {name2_json};

        const color1 = 'rgba(130, 81, 32, 0.85)';
        const color1Light = 'rgba(130, 81, 32, 0.3)';
        const color2 = 'rgba(127, 140, 141, 0.85)';
        const color2Light = 'rgba(127, 140, 141, 0.3)';

        const commonOptions = {{
          responsive: true,
          maintainAspectRatio: false,
          animation: {{ duration: 500 }},
          resizeDelay: 100
        }};

        // Bar Chart (NORMALIZED)
        const barCtx = document.getElementById('barChart').getContext('2d');
        new Chart(barCtx, {{
          type: 'bar',
          data: {{
            labels: labels,
            datasets: [
              {{
                label: name1,
                data: radar1,  // Use normalized values
                backgroundColor: color1,
                borderColor: color1,
                borderWidth: 1
              }},
              {{
                label: name2,
                data: radar2,  // Use normalized values
                backgroundColor: color2,
                borderColor: color2,
                borderWidth: 1
              }}
            ]
          }},
          options: {{
            ...commonOptions,
            plugins: {{
              legend: {{ position: 'top' }},
              tooltip: {{
                callbacks: {{
                  label: function(ctx) {{
                    const idx = ctx.dataIndex;
                    const raw = ctx.datasetIndex === 0 ? values1[idx] : values2[idx];
                    const rawStr = raw !== null ? raw.toLocaleString('pt-BR') : 'N/A';
                    return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + ' (real: ' + rawStr + ')';
                  }}
                }}
              }}
            }},
            scales: {{
              x: {{ grid: {{ display: false }} }},
              y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Score (0-100)' }} }}
            }}
          }}
        }});

        // Horizontal Bar Chart (NORMALIZED)
        const hbarCtx = document.getElementById('hbarChart').getContext('2d');
        new Chart(hbarCtx, {{
          type: 'bar',
          data: {{
            labels: labels,
            datasets: [
              {{
                label: name1,
                data: radar1,  // Use normalized values
                backgroundColor: color1,
                borderColor: color1,
                borderWidth: 1
              }},
              {{
                label: name2,
                data: radar2,  // Use normalized values
                backgroundColor: color2,
                borderColor: color2,
                borderWidth: 1
              }}
            ]
          }},
          options: {{
            ...commonOptions,
            indexAxis: 'y',
            plugins: {{
              legend: {{ position: 'top' }},
              tooltip: {{
                callbacks: {{
                  label: function(ctx) {{
                    const idx = ctx.dataIndex;
                    const raw = ctx.datasetIndex === 0 ? values1[idx] : values2[idx];
                    const rawStr = raw !== null ? raw.toLocaleString('pt-BR') : 'N/A';
                    return ctx.dataset.label + ': ' + ctx.parsed.x.toFixed(1) + ' (real: ' + rawStr + ')';
                  }}
                }}
              }}
            }},
            scales: {{
              y: {{ grid: {{ display: false }} }},
              x: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Score (0-100)' }} }}
            }}
          }}
        }});

        // Radar Chart (already normalized)
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
          type: 'radar',
          data: {{
            labels: labels,
            datasets: [
              {{
                label: name1,
                data: radar1,
                fill: true,
                backgroundColor: color1Light,
                borderColor: color1,
                pointBackgroundColor: color1,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: color1
              }},
              {{
                label: name2,
                data: radar2,
                fill: true,
                backgroundColor: color2Light,
                borderColor: color2,
                pointBackgroundColor: color2,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: color2
              }}
            ]
          }},
          options: {{
            ...commonOptions,
            plugins: {{
              legend: {{ position: 'top' }}
            }},
            scales: {{
              r: {{
                angleLines: {{ display: true }},
                suggestedMin: 0,
                suggestedMax: 100
              }}
            }}
          }}
        }});
      }});
    </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>{_html_escape(title)}</title>
      {css}
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div class="title">{_html_escape(title)}</div>
          <div class="subtitle">{_html_escape(subtitle)}</div>
        </div>
        <div class="content">
          <div class="companies-header">
            <div class="company-badge c1">{_html_escape(name1)} ({_html_escape(ticker1)})</div>
            <div class="company-badge c2">{_html_escape(name2)} ({_html_escape(ticker2)})</div>
          </div>

          <div class="charts-grid">
            <div class="chart-container">
              <h3>Comparação por Métrica (Barras Verticais)</h3>
              <div class="chart-wrapper">
                <canvas id="barChart"></canvas>
              </div>
            </div>
            <div class="chart-container">
              <h3>Comparação por Métrica (Barras Horizontais)</h3>
              <div class="chart-wrapper">
                <canvas id="hbarChart"></canvas>
              </div>
            </div>
          </div>

          <div class="radar-section">
            <div class="radar-container">
              <h3>Visão Geral (Radar Normalizado)</h3>
              <div class="radar-wrapper">
                <canvas id="radarChart"></canvas>
              </div>
              <p style="font-size:11px; color:#888; text-align:center; margin-top:8px;">
                * Valores normalizados (0-100). Para múltiplos (P/L, EV/EBITDA), menor é melhor.
              </p>
            </div>
          </div>

          <div class="summary-section">
            <h3>📊 Tabela Resumo</h3>
            <table class="summary">
              <thead>
                <tr>
                  <th>Métrica</th>
                  <th>{_html_escape(name1)}</th>
                  <th>{_html_escape(name2)}</th>
                </tr>
              </thead>
              <tbody>
                {table_rows_html}
              </tbody>
            </table>
          </div>
        </div>
        <div class="footer">
          Comparação gerada automaticamente — HTML offline com Chart.js
        </div>
      </div>
      {js}
    </body>
    </html>
    """
    return html


# -------------------- App file uploader --------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read the StockGuide sheet without headers
    df_raw = pd.read_excel(uploaded_file, sheet_name="StockGuide", header=None)

    # Find the two header rows (look for "Ticker" in second header row)
    header_row_1_idx = None
    header_row_2_idx = None

    for idx, row in df_raw.iterrows():
        row_values = [str(v).strip() if pd.notna(v) else "" for v in row.values]
        if "Ticker" in row_values:
            header_row_2_idx = idx
            header_row_1_idx = idx - 1
            break

    if header_row_2_idx is None:
        st.error("Could not find header row with 'Ticker'")
        st.stop()

    # Get both header rows
    header_row_1 = df_raw.iloc[header_row_1_idx].tolist()
    header_row_2 = df_raw.iloc[header_row_2_idx].tolist()

    # Build combined headers
    current_parent = ""
    combined_headers = []

    for i, (parent, child) in enumerate(zip(header_row_1, header_row_2)):
        parent_str = str(parent).strip() if pd.notna(parent) else ""
        child_str = str(child).strip() if pd.notna(child) else ""

        if parent_str and parent_str != "nan":
            current_parent = parent_str

        if child_str and child_str != "nan":
            if child_str in ["Ticker", "Rating"]:
                combined_headers.append(child_str)
            elif child_str == "(R$ mn)" and "Market Cap" in current_parent:
                combined_headers.append("Market Cap (R$ mn)")
            elif current_parent and current_parent not in [
                "Stock Guide",
                "Dados Gerais",
            ]:
                combined_headers.append(f"{current_parent} {child_str}")
            else:
                combined_headers.append(child_str)
        elif parent_str and parent_str != "nan":
            combined_headers.append(parent_str)
        else:
            combined_headers.append(f"Col_{i}")

    # First column is the stock name
    combined_headers[0] = "Nome"

    # Get data rows (after headers)
    df_data = df_raw.iloc[header_row_2_idx + 1 :].copy()
    df_data.columns = combined_headers

    # Build Setor by scanning "Stock Guide" blocks
    if "Ticker" not in df_data.columns or "Stock Guide" not in df_data.columns:
        st.error("Expected columns 'Ticker' and 'Stock Guide' not found.")
        st.stop()

    def _is_empty(x):
        return (
            pd.isna(x)
            or str(x).strip() == ""
            or str(x).strip().lower() == "nan"
        )

    df_data["Setor"] = pd.NA
    i = 0
    n = len(df_data)
    while i < n:
        sg_val = df_data.iloc[i]["Stock Guide"]
        ticker_val = df_data.iloc[i]["Ticker"]
        if (
            _is_empty(ticker_val)
            and not _is_empty(sg_val)
            and str(sg_val).strip().lower() != "mediana"
        ):
            current_sector = str(sg_val).strip()
            j = i + 1
            while j < n:
                sg_j = df_data.iloc[j]["Stock Guide"]
                if not _is_empty(sg_j) and str(sg_j).strip().lower() == "mediana":
                    break
                if not _is_empty(df_data.iloc[j]["Ticker"]):
                    df_data.at[df_data.index[j], "Setor"] = current_sector
                j += 1
            i = j + 1
        else:
            i += 1

    # Keep only stock rows (non-empty Ticker)
    mask_valid_ticker = (
        df_data["Ticker"].astype(str).str.strip().str.lower().ne("nan")
        & df_data["Ticker"].astype(str).str.strip().ne("")
    )
    df_final = df_data.loc[mask_valid_ticker].copy()

    # Reorder columns to put Setor after Nome and Ticker
    cols = df_final.columns.tolist()
    if "Setor" in cols:
        cols.remove("Setor")
        insert_at = (
            2
            if "Nome" in df_final.columns and "Ticker" in df_final.columns
            else len(cols)
        )
        cols.insert(insert_at, "Setor")
        df_final = df_final[cols]

    # Normalize empty strings to NA so we can filter/drop cleanly
    df_final = df_final.replace(r"^\s*$", pd.NA, regex=True)

    # Take out rows where "Market Cap (R$ mn)" is empty (if present)
    mc_col = "Market Cap (R$ mn)"
    if mc_col in df_final.columns:
        df_final = df_final.dropna(subset=[mc_col])
    else:
        st.warning(
            "Column 'Market Cap (R$ mn)' not found. "
            "Skipping market cap filter."
        )

    # Drop completely empty columns (all NA after normalization)
    df_final = df_final.dropna(axis=1, how="all")

    # Display results
    total_stocks = len(df_final)
    total_sectors = (
        df_final["Setor"].nunique(dropna=True) if "Setor" in df_final else 0
    )
    st.success(f"✅ Extracted {total_stocks} stocks from {total_sectors} sectors")

    # Show sector filter
    available_sectors = (
        df_final["Setor"].dropna().unique() if "Setor" in df_final else []
    )
    selected_sectors = st.multiselect(
        "Filter by Sector", options=available_sectors, default=available_sectors
    )

    if len(selected_sectors) > 0 and "Setor" in df_final.columns:
        df_filtered = df_final[df_final["Setor"].isin(selected_sectors)].copy()
    else:
        df_filtered = df_final.copy()

    # Show columns info
    with st.expander("📋 Column Names"):
        st.write(df_final.columns.tolist())

    # Display the dataframe
    st.subheader(f"Stock Guide Data ({len(df_filtered)} stocks)")
    st.dataframe(df_filtered, use_container_width=True, height=600)

    # Download button
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="stock_guide_processed.csv",
        mime="text/csv",
    )

    # ------------------ EXTRA SHEETS PARSING AND DISPLAY ------------------
    sector_sheets = [
        "Big Banks",
        "Planilha1",
        "Small Banks",
        "Financials",
        "Apparel and Footwear",
        "Discretionary",
        "Non Discretionary",
        "Housing",
        "Commercial Properties",
        "Distribution",
        "Generation",
        "Transmission",
        "Sewage & Water",
        "Metals and Mining",
        "Pulp and Paper",
        "Healthcare",
        "Education",
        "Oil",
        "Agribusiness",
        "Food and Beverage",
        "Car Rental & Logistics",
        "Aerospace",
        "Capital Goods",
        "Telecom",
        "Tech and Media",
        "Infrastructure",
    ]

    with st.spinner("Scanning sector worksheets for extra data..."):
        # Reset file pointer so we can read again
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        extras = parse_extra_sheets(uploaded_file, sector_sheets)

    st.subheader("🧩 Extra Data Found In Sector Worksheets")

    tabs = st.tabs(["Fundamentalistas", "Destaques"])

    # 1) Fundamentalistas (wide)
    with tabs[0]:
        df_fund = extras["fundamentals"]
        if df_fund.empty:
            st.info("No 'Dados fundamentalistas' tables found.")
        else:
            st.write(
                f"Tickers: {df_fund['Ticker'].nunique()} | "
                f"Rows: {len(df_fund)} | "
                f"Sectors: {df_fund['Setor'].nunique()}"
            )
            st.dataframe(df_fund, use_container_width=True, height=480)
            fund_csv = df_fund.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Fundamentals CSV",
                data=fund_csv,
                file_name="extra_fundamentals.csv",
                mime="text/csv",
            )

    # 2) Destaques financeiros (wide)
    with tabs[1]:
        df_high = extras["highlights"]
        if df_high.empty:
            st.info("No 'Destaques financeiros' tables found.")
        else:
            st.write(
                f"Tickers: {df_high['Ticker'].nunique()} | "
                f"Rows: {len(df_high)} | "
                f"Sectors: {df_high['Setor'].nunique()}"
            )
            st.dataframe(df_high, use_container_width=True, height=480)
            high_csv = df_high.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Highlights CSV",
                data=high_csv,
                file_name="extra_highlights.csv",
                mime="text/csv",
            )

    # Always create enriched dataframe with all extra metrics
    df_enriched = df_final.copy()
    for key in ("fundamentals", "highlights"):
        df_enriched = safe_merge_by_ticker(
            df_enriched, extras[key], how="left"
        )
    
    # Option to display the enriched dataframe
    st.subheader("🔗 Merge extra info into main StockGuide dataframe")
    want_merge = st.checkbox(
        "Show 'Stock Guide + Extra Metrics' table",
        value=True,
    )
    if want_merge:
        st.subheader(f"Stock Guide + Extra Metrics ({len(df_enriched)} stocks)")
        st.dataframe(df_enriched, use_container_width=True, height=600)
        csv_enriched = df_enriched.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Enriched CSV",
            data=csv_enriched,
            file_name="stock_guide_enriched.csv",
            mime="text/csv",
        )

    # ------------------ STOCK COMPARATOR (HTML) ------------------
    st.subheader("🧪 Stock Comparator (HTML)")

    # Use the enriched dataframe for comparator (includes fundamentals + highlights)
    # This same df_comp_source will be used for both HTML comparator and graph comparisons
    df_comp_source = df_enriched


    cols_all = df_comp_source.columns.tolist()
    
    # Show info about available metrics
    st.info(
        f"📊 **{len(cols_all)} colunas** disponíveis para comparação "
        f"(incluindo {len(df_comp_source)} empresas com métricas de Fundamentals e Highlights)"
    )


    # Auto-detecção de colunas para facilitar
    def _auto_pick(aliases: list[str]) -> str | None:
        low = [c.lower() for c in cols_all]
        for i, c in enumerate(low):
            if any(a in c for a in aliases):
                return cols_all[i]
        return None

    ticker_guess = _auto_pick(
        ["ticker", "código", "codigo", "papel", "symbol", "ativo", "codneg"]
    ) or ("Ticker" if "Ticker" in df_comp_source.columns else None)
    name_guess = (
        "Nome"
        if "Nome" in df_comp_source.columns
        else _auto_pick(["nome", "empresa", "companhia", "company", "razão", "razao"])
    )
    sector_guess = (
        "Setor"
        if "Setor" in df_comp_source.columns
        else _auto_pick(["setor", "sector", "segmento"])
    )

    st.markdown(
        "Mapeie as colunas para o comparador (use fallback se Ticker não existir):"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        ticker_col = st.selectbox(
            "Coluna do Ticker",
            options=["<nenhuma>"] + cols_all,
            index=(
                (["<nenhuma>"] + cols_all).index(ticker_guess)
                if ticker_guess in cols_all
                else 0
            ),
            help="Obrigatória para filtrar as linhas por ticker",
        )
    with c2:
        name_col = st.selectbox(
            "Coluna do Nome/Empresa (opcional)",
            options=["<nenhuma>"] + cols_all,
            index=(
                (["<nenhuma>"] + cols_all).index(name_guess)
                if name_guess in cols_all
                else 0
            ),
        )
    with c3:
        sector_col = st.selectbox(
            "Coluna do Setor (opcional)",
            options=["<nenhuma>"] + cols_all,
            index=(
                (["<nenhuma>"] + cols_all).index(sector_guess)
                if sector_guess in cols_all
                else 0
            ),
        )

    # Renomeia para padronizar colunas esperadas pelo gerador
    df_comp_named = df_comp_source.copy()
    rename_map = {}
    if ticker_col != "<nenhuma>":
        rename_map[ticker_col] = "Ticker"
    if name_col != "<nenhuma>":
        rename_map[name_col] = "Nome"
    if sector_col != "<nenhuma>":
        rename_map[sector_col] = "Setor"
    if rename_map:
        df_comp_named = df_comp_named.rename(columns=rename_map)

    # Se não houver Ticker mesmo após mapeamento, gera fallback: tabela simples
    if "Ticker" not in df_comp_named.columns:
        st.info(
            "Ticker não mapeado. Gerarei uma Tabela HTML simples com as colunas disponíveis."
        )
        if st.button("🔧 Gerar Tabela HTML (fallback)", type="primary"):
            html_tbl = generate_simple_table_html(
                df=df_comp_named,
                title="Tabela de Dados",
                subtitle=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            st.components.v1.html(html_tbl, height=700, scrolling=True)
            st.download_button(
                "📥 Download Tabela (HTML)",
                data=html_tbl.encode("utf-8"),
                file_name="tabela_dados.html",
                mime="text/html",
                use_container_width=True,
            )
    else:
        # Fluxo normal do comparador
        base_name_col = (
            "Nome"
            if "Nome" in df_comp_named.columns
            else ("Empresa" if "Empresa" in df_comp_named.columns else None)
        )

        all_rows = df_comp_named.dropna(subset=["Ticker"]).copy()
        all_rows["Ticker"] = all_rows["Ticker"].astype(str).str.strip()
        tickers_all = sorted(all_rows["Ticker"].unique().tolist())

        # Default: Top 10 por Market Cap (se existir) ou primeiros 10
        default_tickers = tickers_all[:10]
        mc_col_cmp = next(
            (c for c in df_comp_named.columns if "market cap" in str(c).lower()),
            None,
        )
        if mc_col_cmp:
            tmp = all_rows.copy()
            tmp["_mc"] = _to_numeric_ptbr_series(tmp[mc_col_cmp])
            tmp = tmp.sort_values("_mc", ascending=False, na_position="last")
            default_tickers = tmp["Ticker"].head(10).tolist()

        selected_tickers = st.multiselect(
            "Selecione os Tickers",
            options=tickers_all,
            default=default_tickers,
            help="Escolha as ações para comparar",
        )

        metric_candidates = _collect_metric_candidates(df_comp_named)

        pref_keys = [
            "Market Cap (R$ mn)",
            "Volume",
            "Volume 3M",
            "Dividend Yield",
            "Yld",
            "Yield",
            "P/L",
            "P/E",
            "EV/EBITDA",
            "P/BV",
            "ROE",
            "Margem",
        ]
        preferred_defaults = []
        for k in pref_keys:
            hit = next(
                (c for c in metric_candidates if k.lower() in c.lower()), None
            )
            if hit and hit not in preferred_defaults:
                preferred_defaults.append(hit)
        default_metrics = preferred_defaults[:8] or metric_candidates[:8]

        selected_metrics = st.multiselect(
            "Selecione as Métricas",
            options=metric_candidates,
            default=default_metrics,
            help="Somente colunas com valores numéricos detectados",
        )

        col_cmp_a, col_cmp_b = st.columns([3, 1])
        with col_cmp_a:
            cmp_title = st.text_input(
                "Título do comparador", value="Stock Comparator"
            )
            cmp_subtitle = st.text_input(
                "Subtítulo (opcional)",
                value=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
        with col_cmp_b:
            gen_cmp = st.button(
                "🧩 Gerar HTML do Comparador",
                type="primary",
                use_container_width=True,
            )

        if gen_cmp:
            if not selected_tickers:
                st.warning("Selecione ao menos um ticker.")
            elif not selected_metrics:
                st.warning("Selecione ao menos uma métrica.")
            else:
                html_cmp = generate_comparator_html(
                    df_source=df_comp_named,
                    tickers=selected_tickers,
                    metrics=selected_metrics,
                    title=cmp_title,
                    subtitle=cmp_subtitle,
                )
                st.components.v1.html(html_cmp, height=700, scrolling=True)

                safe_base = sanitize_filename(cmp_title or "comparador")
                file_name = f"{safe_base}.html"
                st.download_button(
                    "📥 Download Comparador (HTML)",
                    data=html_cmp.encode("utf-8"),
                    file_name=file_name,
                    mime="text/html",
                    use_container_width=True,
                )

    # ------------------ NEW: TWO-COMPANY GRAPH COMPARISON ------------------
    st.markdown("---")
    st.subheader("📈 Comparação Gráfica de Duas Empresas")
    st.markdown(
        "Selecione **exatamente duas empresas** para gerar um relatório HTML "
        "com gráficos de barras e radar comparando as métricas selecionadas."
    )

    # Prepare data source (same as comparator)
    df_graph_source = df_comp_source.copy()
    if ticker_col != "<nenhuma>" and ticker_col in df_graph_source.columns:
        df_graph_source = df_graph_source.rename(columns={ticker_col: "Ticker"})
    if name_col != "<nenhuma>" and name_col in df_graph_source.columns:
        df_graph_source = df_graph_source.rename(columns={name_col: "Nome"})
    if sector_col != "<nenhuma>" and sector_col in df_graph_source.columns:
        df_graph_source = df_graph_source.rename(columns={sector_col: "Setor"})

    if "Ticker" not in df_graph_source.columns:
        st.warning("Coluna 'Ticker' não encontrada. Mapeie a coluna acima.")
    else:
        graph_rows = df_graph_source.dropna(subset=["Ticker"]).copy()
        graph_rows["Ticker"] = graph_rows["Ticker"].astype(str).str.strip()
        graph_tickers = sorted(graph_rows["Ticker"].unique().tolist())

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ticker_a = st.selectbox(
                "🏢 Empresa 1",
                options=[""] + graph_tickers,
                index=0,
                help="Selecione a primeira empresa",
            )
        with col_g2:
            ticker_b = st.selectbox(
                "🏢 Empresa 2",
                options=[""] + graph_tickers,
                index=0,
                help="Selecione a segunda empresa",
            )

        # Metrics for graph comparison
        graph_metric_candidates = _collect_metric_candidates(df_graph_source)

        # Default metrics for graphs (fewer for better visualization)
        graph_pref_keys = [
            "Market Cap",
            "P/L",
            "P/E",
            "EV/EBITDA",
            "ROE",
            "Dividend",
            "Yield",
            "Margem",
            "Volume",
        ]
        graph_preferred = []
        for k in graph_pref_keys:
            hit = next(
                (c for c in graph_metric_candidates if k.lower() in c.lower()),
                None,
            )
            if hit and hit not in graph_preferred:
                graph_preferred.append(hit)
        graph_default_metrics = graph_preferred[:6] or graph_metric_candidates[:6]

        selected_graph_metrics = st.multiselect(
            "Métricas para os gráficos",
            options=graph_metric_candidates,
            default=graph_default_metrics,
            help="Selecione as métricas que deseja comparar nos gráficos",
        )

        col_gt1, col_gt2 = st.columns([3, 1])
        with col_gt1:
            graph_title = st.text_input(
                "Título do relatório gráfico",
                value="Comparação de Empresas",
            )
            graph_subtitle = st.text_input(
                "Subtítulo do relatório gráfico",
                value=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
        with col_gt2:
            gen_graph = st.button(
                "📊 Gerar Comparação Gráfica",
                type="primary",
                use_container_width=True,
            )

        if gen_graph:
            if not ticker_a or not ticker_b:
                st.warning("Selecione duas empresas para comparar.")
            elif ticker_a == ticker_b:
                st.warning("Selecione duas empresas **diferentes**.")
            elif not selected_graph_metrics:
                st.warning("Selecione ao menos uma métrica para os gráficos.")
            else:
                html_graph = generate_two_company_comparison_html(
                    df_source=df_graph_source,
                    ticker1=ticker_a,
                    ticker2=ticker_b,
                    metrics=selected_graph_metrics,
                    title=graph_title,
                    subtitle=graph_subtitle,
                )

                st.components.v1.html(html_graph, height=900, scrolling=True)

                safe_graph_name = sanitize_filename(
                    f"{ticker_a}_vs_{ticker_b}"
                )
                st.download_button(
                    "📥 Download Comparação Gráfica (HTML)",
                    data=html_graph.encode("utf-8"),
                    file_name=f"{safe_graph_name}_comparison.html",
                    mime="text/html",
                    use_container_width=True,
                )

else:
    st.info("👆 Please upload an Excel file to get started")
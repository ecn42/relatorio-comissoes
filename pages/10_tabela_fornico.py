import io
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy

import pandas as pd
import streamlit as st
from pptx import Presentation
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

# Constants (keep from original)
TITLE_COLOR = "#8C6239"
FONT_NAME = "Montserrat"
SLIDE_WIDTH = Inches(8.27)
SLIDE_HEIGHT = Inches(11.69)
CERES_BRONZE = RGBColor(206, 170, 102)
HDR_FONT_SIZE = 8
BODY_FONT_SIZE = 7
MARGIN_IN = 0.5


@dataclass
class AllocationItem:
    """Represents a single allocation row."""
    name: str
    profile_data: Dict[str, Tuple[str, str]]  # {profile: (target, range)}
    parent: Optional[str] = None
    level: int = 0
    children: List['AllocationItem'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def parse_allocation_table(raw_text: str) -> Dict:
    """
    Parse multi-profile allocation table format.
    Detects profiles (Conservador, Moderado, Sofisticado) and hierarchical
    structure.
    """
    text = raw_text.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    if not lines:
        return {"profiles": [], "items": []}

    # Step 1: Detect profile names from first line with profile keywords
    profiles = []
    profile_keywords = {
        "conservador": "Conservador",
        "moderado": "Moderado",
        "sofisticado": "Sofisticado"
    }

    for line in lines[:20]:
        s_lower = line.lower()
        for keyword, name in profile_keywords.items():
            if keyword in s_lower and name not in profiles:
                profiles.append(name)

    if not profiles:
        profiles = ["Profile 1", "Profile 2", "Profile 3"]

    # Step 2: Parse rows using regex for tab/space separation
    items: List[AllocationItem] = []
    sep_pattern = re.compile(r"\t+| {2,}")

    for line in lines:
        if not line.strip() or "alvo" in line.lower() or "faixa" in line.lower():
            continue

        # Skip header-like lines
        if any(
            keyword in line.lower()
            for keyword in ["conservador", "moderado", "sofisticado",
                            "total", "asset allocation"]
        ):
            continue

        parts = [p.strip() for p in sep_pattern.split(line) if p.strip()]

        if len(parts) < 2:
            continue

        name = parts[0]
        profile_data = {}

        # Extract profile values
        idx = 1
        for profile in profiles:
            if idx < len(parts):
                target = parts[idx]
                pct_range = parts[idx + 1] if idx + 1 < len(parts) else ""
                profile_data[profile] = (target, pct_range)
                idx += 2
            else:
                profile_data[profile] = ("", "")

        item = AllocationItem(
            name=name,
            profile_data=profile_data,
            level=_detect_level(name)
        )
        items.append(item)

    return {"profiles": profiles, "items": items}


def _detect_level(name: str) -> int:
    """Detect hierarchy level by indentation or keywords."""
    if name and name[0].isspace():
        return len(name) - len(name.lstrip())
    return 0


def hex_to_rgb(hex_str: str) -> RGBColor:
    """Convert hex color to RGBColor."""
    h = hex_str.strip().lstrip("#")
    return RGBColor(
        int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    )


def set_cell_text_centered(
    cell, text: str, size: int, bold: bool = False, color: str = "#FFFFFF"
):
    """Set cell text centered."""
    cell.text_frame.clear()
    cell.text_frame.word_wrap = True
    cell.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = hex_to_rgb(color)


def set_cell_fill_with_alpha(
    cell, rgb_tuple: Tuple[int, int, int], alpha: float
):
    """Set cell fill with alpha transparency."""
    from pptx.oxml.xmlchemy import OxmlElement

    try:
        tcPr = cell._tc.get_or_add_tcPr()

        for fill in tcPr.findall(qn("a:solidFill")):
            tcPr.remove(fill)

        solidFill = OxmlElement("a:solidFill")
        srgbClr = OxmlElement("a:srgbClr")
        hex_val = "{:02X}{:02X}{:02X}".format(
            rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]
        )
        srgbClr.set("val", hex_val)

        alpha_elem = OxmlElement("a:alpha")
        alpha_elem.set("val", str(int(alpha * 100000)))
        srgbClr.append(alpha_elem)

        solidFill.append(srgbClr)
        tcPr.append(solidFill)
    except Exception:
        pass


def apply_borderless_theme_to_table(table) -> None:
    """Remove all borders from table."""
    from pptx.oxml.xmlchemy import OxmlElement

    try:
        tbl = table._tbl
        for tc in tbl.iter(qn("a:tc")):
            tcPr = tc.get_or_add_tcPr()

            for border_tag in [
                qn("a:lnL"), qn("a:lnR"), qn("a:lnT"), qn("a:lnB")
            ]:
                for border_elem in tcPr.findall(border_tag):
                    tcPr.remove(border_elem)

            for border_pos in ["lnL", "lnR", "lnT", "lnB"]:
                ln = OxmlElement(f"a:{border_pos}")
                noFill = OxmlElement("a:noFill")
                ln.append(noFill)
                tcPr.append(ln)
    except Exception:
        pass


def create_allocation_slide(
    prs: Presentation,
    title: str,
    parsed: Dict,
    bg_source: Optional[io.BytesIO] = None,
) -> None:
    """Create a slide with allocation table."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Add background if provided
    if bg_source:
        try:
            slide.shapes.add_picture(
                bg_source, Inches(0), Inches(0),
                prs.slide_width, prs.slide_height
            )
        except Exception:
            pass

    margin = Inches(MARGIN_IN)
    content_w = prs.slide_width - 2 * margin

    # Title
    title_top = Inches(0.3)
    tb_title = slide.shapes.add_textbox(
        margin, title_top, content_w, Inches(0.6)
    )
    p = tb_title.text_frame.paragraphs[0]
    p.text = title
    p.font.name = FONT_NAME
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb("#FFFFFF")

    profiles = parsed.get("profiles", [])
    items = parsed.get("items", [])

    if not profiles or not items:
        st.warning("No data to display")
        return

    # Calculate table dimensions
    num_rows = len(items) + 1
    num_cols = 1 + len(profiles) * 2

    table_top = Inches(1.2)
    table_left = margin

    table_shape = slide.shapes.add_table(
        num_rows, num_cols, table_left, table_top,
        content_w, Inches(9.0)
    )
    table = table_shape.table

    apply_borderless_theme_to_table(table)

    # Column widths
    col_widths = []
    name_w = int(content_w * 0.25)
    col_widths.append(name_w)

    per_profile_w = int((content_w - name_w) / len(profiles))
    for _ in range(len(profiles)):
        col_widths.append(int(per_profile_w * 0.5))
        col_widths.append(int(per_profile_w * 0.5))

    for j, w in enumerate(col_widths):
        table.columns[j].width = w

    # Set row heights
    hdr_h = Inches(0.35)
    row_h = Inches(0.3)

    table.rows[0].height = hdr_h
    for i in range(1, num_rows):
        table.rows[i].height = row_h

    # Build header row
    cell = table.cell(0, 0)
    set_cell_fill_with_alpha(cell, (0x8C, 0x62, 0x39), 0.6)
    set_cell_text_centered(cell, "Alocação", HDR_FONT_SIZE, True)

    col_idx = 1
    for profile in profiles:
        # Target
        cell = table.cell(0, col_idx)
        set_cell_fill_with_alpha(cell, (0x8C, 0x62, 0x39), 0.6)
        set_cell_text_centered(
            cell, f"{profile}\n(Alvo %)", HDR_FONT_SIZE, True
        )
        col_idx += 1

        # Range
        cell = table.cell(0, col_idx)
        set_cell_fill_with_alpha(cell, (0x8C, 0x62, 0x39), 0.6)
        set_cell_text_centered(
            cell, f"{profile}\n(Faixa %)", HDR_FONT_SIZE, True
        )
        col_idx += 1

    # Fill data rows
    for row_idx, item in enumerate(items, start=1):
        is_even = row_idx % 2 == 0
        bg_color = (206, 170, 102) if is_even else (255, 255, 255)
        alpha = 0.3 if is_even else 0.0

        # Name column
        cell = table.cell(row_idx, 0)
        if is_even:
            set_cell_fill_with_alpha(cell, bg_color, alpha)
        set_cell_text_centered(
            cell, item.name, BODY_FONT_SIZE, False
        )

        # Profile columns
        col_idx = 1
        for profile in profiles:
            target, pct_range = item.profile_data.get(
                profile, ("", "")
            )

            # Target cell
            cell = table.cell(row_idx, col_idx)
            if is_even:
                set_cell_fill_with_alpha(cell, bg_color, alpha)
            set_cell_text_centered(
                cell, target, BODY_FONT_SIZE, False
            )
            col_idx += 1

            # Range cell
            cell = table.cell(row_idx, col_idx)
            if is_even:
                set_cell_fill_with_alpha(cell, bg_color, alpha)
            set_cell_text_centered(
                cell, pct_range, BODY_FONT_SIZE, False
            )
            col_idx += 1


# -------- Streamlit UI --------

st.set_page_config(
    page_title="Allocation Table Parser", layout="wide"
)

st.markdown(
    "<h2 style='text-align: center;'>Multi-Profile Allocation Parser</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    "Parse multi-profile asset allocation tables with hierarchical structure."
)

placeholder = """Paste allocation table here...
Example:

Asset Allocation	Conservador	 	Moderado	 	Sofisticado	 
	Alvo (%)	Faixa Tolerada (%)	Alvo (%)	Faixa Tolerada (%)	Alvo (%)	Faixa Tolerada (%)
Liquidez	17,5%	10% - 50%	12,5%	10% - 40%	10,0%	0% - 30%
Fundos Liquidez D+0 e D+1	17,5%	10% - 25%	12,5%	15% - 25%	10,0%	0% - 20%
Renda Fixa - Títulos Públicos	0,0%	0% - 25%	0,0%	0% - 25%	0,0%	0% - 25%
"""

col1, col2 = st.columns([1, 1])

with col1:
    titulo = st.text_input("Título da Tabela", "Asset Allocation")

with col2:
    pass

texto = st.text_area(
    "Texto da Alocação", height=300, placeholder=placeholder
)

if st.button("Parsear Tabela", type="primary"):
    if texto.strip():
        try:
            parsed = parse_allocation_table(texto)
            st.session_state.parsed = parsed
            st.success("Tabela parseada com sucesso!")
        except Exception as e:
            st.error("Erro ao parsear")
            st.exception(e)

st.divider()

# Display parsed data
if "parsed" in st.session_state:
    parsed = st.session_state.parsed
    profiles = parsed.get("profiles", [])
    items = parsed.get("items", [])

    st.subheader("Dados Parseados")

    # Build display DataFrame
    data_rows = []
    for item in items:
        row = {"Alocação": item.name}
        for profile in profiles:
            target, pct_range = item.profile_data.get(
                profile, ("", "")
            )
            row[f"{profile} (Alvo)"] = target
            row[f"{profile} (Faixa)"] = pct_range
        data_rows.append(row)

    if data_rows:
        df = pd.DataFrame(data_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Export to PPTX
        if st.button("Gerar PowerPoint", type="primary"):
            try:
                prs = Presentation()
                prs.slide_width = SLIDE_WIDTH
                prs.slide_height = SLIDE_HEIGHT

                create_allocation_slide(
                    prs, titulo, parsed
                )

                bio = io.BytesIO()
                prs.save(bio)
                bio.seek(0)

                st.download_button(
                    label="Baixar PPTX",
                    data=bio,
                    file_name="allocation_table.pptx",
                    mime=(
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "presentationml.presentation"
                    ),
                )
                st.success("PPTX gerado!")
            except Exception as e:
                st.error("Erro ao gerar PPTX")
                st.exception(e)
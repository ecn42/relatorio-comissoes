import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
from playwright.sync_api import sync_playwright
import time
import re


# Colors from report_format_guide.md
BRAND_BROWN = "#825120"
BRAND_BROWN_DARK = "#6B4219"
POSITIVE_COLOR = "green"
NEGATIVE_COLOR = "red"
CERES_COLORS = [
    "#013220",  # Dark green
    "#57575a",  # Gray
    "#b08568",  # Tan/brown
    "#09202e",  # Navy
    "#582308",  # Dark red-brown
    "#7a6200",  # Olive/gold
]

def calculate_payoff(price_range, options_df, multiplier=1.0, brokerage_per_contract=0.0):
    """
    Calculates the total payoff of the options strategy.
    
    Logic changed to respect Signed Premiums:
    - Premium Negative: Debit (Pays)
    - Premium Positive: Credit (Receives)
    
    Total PnL = ((PositionSign * Intrinsic) + Premium) * Multiplier * Qty - (Brokerage * Qty)
    """
    results = pd.DataFrame({'Spot': price_range})
    results['Total Payoff'] = 0.0
    
    for index, row in options_df.iterrows():
        if not row['Ativo']:
            continue
            
        strike = float(row['Strike'])
        premium = float(row['Prêmio'])
        quantity = float(row['Quantidade'])
        opt_type = row['Tipo'] # Call / Put
        position = row['Posição'] # Long / Short
        
        # Calculate Intrinsic Value at Expiration
        if opt_type == 'Call':
            intrinsic = np.maximum(price_range - strike, 0)
        else: # Put
            intrinsic = np.maximum(strike - price_range, 0)
            
        # Determine Position Sign for Intrinsic Value
        # Buy (Long): +Intrinsic
        # Sell (Short): -Intrinsic
        pos_sign = 1.0 if position == 'Buy' else -1.0
        
        # Calculate PnL per unit (price terms)
        # Note: Premium is already signed (Input). 
        # PnL = (Sign * Intrinsic) + Premium
        pnl_no_mult = (pos_sign * intrinsic) + premium
        
        # Apply Multiplier and Quantity
        leg_pnl = (pnl_no_mult * quantity * multiplier) - (brokerage_per_contract * quantity)
        
        results['Total Payoff'] += leg_pnl
        
    return results

def html_to_pdf_single_page(html_content: str) -> bytes:
    """Render HTML to single-page PDF with dynamic height."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            raise RuntimeError(f"Playwright launch failed: {e}")
            
        page = browser.new_page()
        page.set_viewport_size({"width": 794, "height": 1123})
        page.set_content(html_content, wait_until="load")
        page.wait_for_load_state("networkidle")
        
        # Measure content height
        height = page.evaluate("() => Math.ceil(document.body.scrollHeight + 20)")
        
        pdf_bytes = page.pdf(
            print_background=True,
            prefer_css_page_size=False,
            width="210mm",
            height=f"{max(297, height)}px",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()
    
    return pdf_bytes

def generate_html_report(fig, legs_df, explanation, initial_cost_str, brokerage_str, total_cost_str, hide_vals=False):
    # Convert Markdown bold to HTML bold
    explanation_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', explanation)
    explanation_html = explanation_html.replace('\n', '<br>')

    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Prepare display dataframe (drop 'Ativo' if present)
    display_df = legs_df.drop(columns=['Ativo'], errors='ignore')
    
    # Prevent pandas from adding border="1" which causes thick black lines
    legs_html = display_df.to_html(classes='table-style', index=False, border=0)
    
    css = """
    <style>
        body { font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif; padding: 40px; color: #333; background: #fff; font-size: 10px; }
        h1 { color: #825120; margin-bottom: 5px; font-size: 20px; font-weight: 700; }
        .sub-header { color: #666; font-size: 12px; margin-bottom: 20px; }
        h2 { 
            color: #6B4219; 
            border-bottom: 1px solid #E0D5CA; 
            border-top: 1px solid #E0D5CA; 
            padding: 5px 0; 
            margin-top: 20px; 
            background: #F8F8F8;
            font-size: 13px;
            font-weight: 700;
        }
        
        /* Table Style matching report_format_guide.md */
        .table-style { 
            width: 100%; 
            border-collapse: collapse; /* Ensure borders collapse cleanly */
            border-spacing: 0; 
            margin: 10px 0; 
            font-size: 10px; 
            border: none; /* No outer border */
        }
        .table-style th { 
            background: #F5F5F5; 
            text-align: right; /* Match data alignment */
            padding: 3px 4px; 
            border-bottom: 2px solid #E0D5CA; 
            font-weight: 700;
            color: #333;
            border-top: none;
            border-left: none;
            border-right: none;
        }
        .table-style th:first-child { text-align: left; }
        
        .table-style td { 
            padding: 3px 4px; 
            border-bottom: 1px solid #E0D5CA; 
            text-align: right; 
            white-space: nowrap;
            border-top: none;
            border-left: none;
            border-right: none;
        }
        .table-style td:first-child { 
            text-align: left; 
            font-weight: 600; 
        }
        
        .scenario-box { 
            background: #f9f9f9; 
            padding: 15px; 
            border-left: 5px solid #825120; 
            margin-top: 10px; 
            font-size: 11px;
            line-height: 1.5;
        }
        
        .cost-box { 
            background: #F5F5F5; 
            padding: 10px; 
            border: 1px solid #E0D5CA;
            margin: 15px 0; 
            font-weight: 700; 
            display: inline-block;
        }
        .footer {
            margin_top: 40px;
            font-size: 8px;
            color: #777;
            text-align: justify;
            border-top: 1px solid #E0D5CA;
            padding-top: 10px;
        }
    </style>
    """
    
    # Disclaimer Texts
    top_disc = "Esta é uma simulação meramente ilustrativa com valores indicativos. Não configura oferta pública, oferta particular ou recomendação de investimento de valores mobiliários. Os valores apresentados são estimativas sujeitas a alteração sem aviso prévio."
    bottom_disc = "As informações aqui contidas têm caráter meramente indicativo e ilustrativo, não representando oferta pública, oferta particular, ou solicitação de oferta para aquisição de valores mobiliários. Não constitui, ainda, recomendação de investimento, análise de risco ou garantia de rentabilidade. Os valores e cenários apresentados são simulações baseadas em premissas variáveis e podem não refletir resultados futuros. Recomenda-se a leitura cuidadosa do regulamento/prospecto antes de qualquer decisão de investimento."
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>{css}</head>
    <body>
        <h1>Simulação de Estrutura de Opções</h1>
        <p style="font-size: 10px; color: #555;">{top_disc}</p>
        
        <div class="cost-box">
            Resultado Inicial (Prêmio): {initial_cost_str}<br>
            Custo Operacional: {brokerage_str}<br>
            <br>
            <strong>{total_cost_str}</strong>
        </div>
        
        <h2>Estrutura</h2>
        {legs_html}
        
        <h2>Payoff</h2>
        <div>{chart_html}</div>
        
        <h2>Análise de Cenários</h2>
        <div class="scenario-box">{explanation_html}</div>
        
        <div class="footer">
            {bottom_disc}
        </div>
    </body>
    </html>
    """
    return html

# Page Layout
st.title("🛠️ Calculadora de Payoff de Opções")
st.caption("Esta é uma simulação meramente ilustrativa com valores indicativos. Não configura oferta pública, oferta particular ou recomendação de investimento de valores mobiliários. Os valores apresentados são estimativas sujeitas a alteração sem aviso prévio.")
st.markdown("---")

st.markdown("### Estrutura")
st.info("Adicione abaixo as pernas da sua estratégia (Long/Short, Call/Put).")

# Session state initialization
if 'options_data' not in st.session_state:
    st.session_state.options_data = pd.DataFrame([
        {
            "Ativo": True,
            "Tipo": "Call",
            "Posição": "Buy",
            "Strike": 105.0,
            "Prêmio": -2.50, # Pays (Debit)
            "Quantidade": 100.0
        },
        {
            "Ativo": True,
            "Tipo": "Call",
            "Posição": "Sell",
            "Strike": 110.0,
            "Prêmio": 1.20, # Receives (Credit)
            "Quantidade": 100.0
        }
    ])

column_config = {
    "Ativo": st.column_config.CheckboxColumn("Ativo", width="small"),
    "Tipo": st.column_config.SelectboxColumn(
        "Tipo",
        options=["Call", "Put"],
        required=True,
        width="small"
    ),
    "Posição": st.column_config.SelectboxColumn(
        "Posição",
        options=["Buy", "Sell"],
        required=True,
        width="small"
    ),
    "Strike": st.column_config.NumberColumn(
        "Strike",
        min_value=0.0,
        format="%.2f"
    ),
    "Prêmio": st.column_config.NumberColumn(
        "Prêmio",
        format="%.2f"
    ),
    "Quantidade": st.column_config.NumberColumn(
        "Qtd",
        format="%.2f"
    ),
}

edited_df = st.data_editor(
    st.session_state.options_data,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

col_config, col_graph = st.columns([1, 2])

with col_config:
    st.subheader("Parâmetros")
    
    current_price = st.number_input(
        "Preço Atual do Ativo",
        value=100.0,
        step=0.1,
        format="%.2f",
        help="Preço spot atual do ativo objeto"
    )
    
    st.markdown("### Multiplicador de Contrato")
    contract_type = st.selectbox(
        "Tipo de Contrato / Commodity",
        options=["Padrão / Ações (x1)", "Boi Gordo (BGI) - 330@ (x330)", "Soja (SJC) - 450 sacas (x450)", "Milho (CCM) - 450 sacas (x450)", "Customizado"],
        index=0
    )
    
    multiplier = 1.0
    if "BGI" in contract_type:
        multiplier = 330.0
    elif "SJC" in contract_type or "CCM" in contract_type:
        multiplier = 450.0
    elif contract_type == "Customizado":
        multiplier = st.number_input("Multiplicador Personalizado", value=1.0, step=1.0, min_value=0.0)
    
    st.markdown("### Configuração do Gráfico")
    range_pct = st.slider("Extensão do Eixo X (%)", 10, 100, 30, 5) / 100.0
    
    hide_values = st.toggle("Ocultar Valores Financeiros (Modo Apresentação)", value=False)
    
    # Always hide sidebar and header for this page as requested
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
            [data-testid="stHeader"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    # Brokerage input
    corretagem = st.number_input(
        "Corretagem por contrato (R$)",
        value=0.0,
        min_value=0.0,
        step=0.01,
        format="%.2f",
        help="Custo operacional por contrato (não afetado pelo multiplicador)"
    )

with col_graph:
    active_legs = edited_df[edited_df['Ativo'] == True]
    
    if active_legs.empty:
        st.warning("Nenhuma perna ativa selecionada.")
    else:
        # Generate Price Range
        min_price = current_price * (1 - range_pct)
        max_price = current_price * (1 + range_pct)
        # Ensure we have enough resolution
        price_steps = np.linspace(min_price, max_price, 1000)
        
        # Calculate Payoff
        payoff_df = calculate_payoff(price_steps, edited_df, multiplier, corretagem)
        
        # Calculate Initial Structure Cost (always shown)
        # Sum of (Premium * Multiplier * Qty) - (Brokerage * Qty) is the Total PnL at standard.
        # Initial Cash Flow = Sum of (Premium * Multiplier * Qty).
        # Initial Cost = Total Cash Flow including Brokerage.
        # Note: Brokerage is an outflow (negative). User inputs Positive Cost logic?
        # User input "Corretagem" usually implies Cost. So Outflow.
        # CashFlow = Sum(Premium * Mult * Qty) - Sum(Brokerage * Qty).
        
        total_qty = active_legs['Quantidade'].sum()
        total_brokerage = total_qty * corretagem
        
        # Calculate premium cashflow
        premium_cashflow = 0.0
        for _, row in active_legs.iterrows():
            premium_cashflow += (float(row['Prêmio']) * float(row['Quantidade']) * multiplier)
            
        # Net Flow is just the premium cashflow (Brokerage is separate)
        net_initial_flow = premium_cashflow
        
        # Determine label
        if net_initial_flow < 0:
            cost_label = f"Custo Inicial (A Pagar): R$ {abs(net_initial_flow):,.2f}"
            cost_desc = "Você paga para montar esta estrutura."
        else:
            cost_label = f"Crédito Inicial (A Receber): R$ {net_initial_flow:,.2f}"
            cost_desc = "Você recebe para montar esta estrutura."
            

        
        # Calculate Total Cost (Premium - Brokerage)
        net_total_flow = net_initial_flow - total_brokerage
        if net_total_flow < 0:
             total_label = f"Custo Total (C. Inicial + Operacional): R$ {abs(net_total_flow):,.2f}"
        else:
             total_label = f"Crédito Total (C. Inicial + Operacional): R$ {net_total_flow:,.2f}"
             
        st.info(f"**{total_label}**\n\n({cost_label} | Corretagem: R$ {total_brokerage:,.2f})")
        
        # Logic for coloring areas
        y_values = payoff_df['Total Payoff']
        
        # Visualize
        fig = go.Figure()
        
        # 1. Zero Line
        fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.3)
        
        # 2. Current Price Line
        fig.add_vline(
            x=current_price, 
            line_dash="dash", 
            line_color="blue", # Or use a ceres color like #09202e
            annotation_text=f"Spot: {current_price:.2f}",
            annotation_position="top left"
        )
        
        # 3. Payoff Line
        # We want to fill green above 0 and red below 0
        # To do this cleanly in Plotly with a single line is tricky for filling.
        # Approach: Add two invisible lines at y=0 and fill to them?
        # Simpler approach that works well:
        # Plot the main line.
        # Add a trace for positive area (masked) filled to y=0.
        # Add a trace for negative area (masked) filled to y=0.
        
        pos_y = np.where(y_values >= 0, y_values, 0)
        neg_y = np.where(y_values <= 0, y_values, 0)
        
        # Green Area
        fig.add_trace(go.Scatter(
            x=payoff_df['Spot'],
            y=pos_y,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.1)', # Light green
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Red Area
        fig.add_trace(go.Scatter(
            x=payoff_df['Spot'],
            y=neg_y,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)', # Light red
            hoverinfo='skip',
            showlegend=False
        ))
        
        # The Main Line
        
        hover_temp = "Spot: %{x:.2f}<br>PnL: R$ %{y:.2f}<extra></extra>"
        if hide_values:
            hover_temp = "Spot: %{x:.2f}<br>PnL: (Oculto)<extra></extra>"

        fig.add_trace(go.Scatter(
            x=payoff_df['Spot'],
            y=y_values,
            mode='lines',
            name='Resultado',
            line=dict(color=BRAND_BROWN, width=3),
            hovertemplate=hover_temp
        ))
        
        # Calculate Breakeven points (Interpolated for precision)
        sign_changes = np.diff(np.sign(y_values))
        be_indices = np.where(sign_changes != 0)[0]
        
        be_points = []
        for idx in be_indices:
            # Linear interpolation: y = mx + c
            # x_zero = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            x1, x2 = price_steps[idx], price_steps[idx+1]
            y1, y2 = y_values.iloc[idx], y_values.iloc[idx+1]
            
            if (y2 - y1) != 0:
                zero_cross_x = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            else:
                zero_cross_x = x1
                
            be_points.append(zero_cross_x)
            
            text_content = f"BE: {zero_cross_x:.2f}"
            if hide_values:
                 text_content = "BE"
                 
            fig.add_annotation(
                x=zero_cross_x,
                y=0,
                text=text_content,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#555"
            )

        # Layout Configuration
        fig.update_layout(
            title="Gráfico de Payoff da Estrutura",
            xaxis_title="Preço do Ativo no Vencimento",
            yaxis_title="Lucro / Prejuízo Total (R$)",
            template="plotly_white",
            height=600,
            hovermode="x unified",
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)



        # Stats - Global Extremes Calculation
        # 1. Check extremities (Slope at Infinity)
        # Only Calls affect slope at large prices.
        # Buy Call (+), Sell Call (-)
        net_delta_inf = 0.0
        for _, row in active_legs.iterrows():
            if row['Tipo'] == 'Call':
                sign = 1.0 if row['Posição'] == 'Buy' else -1.0
                net_delta_inf += (sign * row['Quantidade'])
        
        # 2. Check Finite Critical Points (0 + Strikes)
        critical_prices = [0.0] + sorted(active_legs['Strike'].unique().tolist())
        finite_pnls = []
        for p in critical_prices:
            # We can use our helper function or calculate directly
             imp_price = np.array([p])
             res = calculate_payoff(imp_price, active_legs, multiplier, corretagem)
             finite_pnls.append(res['Total Payoff'].iloc[0])
        
        global_max = max(finite_pnls)
        global_min = min(finite_pnls)
        
        # Determine Display Strings
        if not hide_values:
            # Profit
            if net_delta_inf > 0.001: # Positive Slope -> Infinite Profit
                profit_lbl = "Ilimitado 🚀"
            else:
                profit_lbl = f"R$ {global_max:,.2f}"
            
            # Loss
            if net_delta_inf < -0.001: # Negative Slope -> Infinite Loss
                loss_lbl = "Ilimitada ⚠️"
            else:
                loss_lbl = f"R$ {global_min:,.2f}"
        else:
            profit_lbl = "---"
            loss_lbl = "---"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Lucro Máximo", profit_lbl)
        c2.metric("Perda Máxima", loss_lbl)
        
        # Riskbox / Description
        st.markdown("#### Detalhes da Estrutura")
        st.dataframe(active_legs, use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Análise de Cenários")

        # Logic to generate robust plain text explanation
        # 1. Define Critical Points: Strikes + Breakeven Points
        strikes = active_legs['Strike'].unique().tolist()
        # Round BE points to 2 decimals for interval logic to avoid micro-segments
        # But keep precision for sorting
        
        all_points = set(strikes + be_points)
        sorted_points = sorted(list(all_points))
        
        # Deduplicate close points (e.g. Strike 100 and BE 100.0001)
        unique_points = []
        if sorted_points:
            unique_points.append(sorted_points[0])
            for p in sorted_points[1:]:
                if abs(p - unique_points[-1]) > 0.01: # 1 cent tolerance
                    unique_points.append(p)
        
        # Define intervals: (-inf, P1), (P1, P2), ..., (Pn, +inf)
        intervals = []
        if not unique_points:
           intervals.append((-float('inf'), float('inf'))) 
        else:
            intervals.append((-float('inf'), unique_points[0]))
            for i in range(len(unique_points) - 1):
                intervals.append((unique_points[i], unique_points[i+1]))
            intervals.append((unique_points[-1], float('inf')))
            
        explanation_text = ""
        
        # Add Breakeven Info header
        if be_points:
            be_str = ", ".join([f"R$ {p:.2f}" for p in be_points])
            explanation_text += f"**Pontos de Breakeven (Zero a Zero):** {be_str}\n\n"
        
        # Helper to calc pnl at a single point (reuse)
        def get_pnl_at_price(price, legs):
            total = 0.0
            for _, row in legs.iterrows():
                imp_price = np.array([price])
                res = calculate_payoff(imp_price, pd.DataFrame([row]), multiplier, corretagem)
                total += res['Total Payoff'].iloc[0]
            return total

        for i, (lower, upper) in enumerate(intervals):
            # 1. Determine Sample Point for State (Profit/Loss) and Slope
            if lower == -float('inf'):
                # Use a point slightly below the upper bound
                test_point = upper - 1.0 
                range_desc = f"Abaixo de R$ {upper:.2f}"
            elif upper == float('inf'):
                test_point = lower + 1.0
                range_desc = f"Acima de R$ {lower:.2f}"
            else:
                test_point = (lower + upper) / 2.0
                range_desc = f"Entre R$ {lower:.2f} e R$ {upper:.2f}"
            
            pnl_at_point = get_pnl_at_price(test_point, active_legs)
            
            # Determine Trend (Slope)
            epsilon = 0.01
            pnl_next = get_pnl_at_price(test_point + epsilon, active_legs)
            slope = (pnl_next - pnl_at_point) / epsilon
            
            # 2. Determine State Label
            # Since we split by BE, the sign should be constant in the interval (mostly)
            if pnl_at_point > 0.001:
                state = "Lucro"
            elif pnl_at_point < -0.001:
                state = "Prejuízo"
            else:
                state = "Zero a Zero (Neutro)"
            
            # 3. Determine Evolution Description
            evolution = ""
            if abs(slope) < 0.001:
                # Constant
                if state == "Lucro":
                    evolution = "O lucro se mantém estável."
                elif state == "Prejuízo":
                    evolution = "O prejuízo se mantém estável (limitado)."
                else:
                    evolution = "O resultado é neutro."
            elif slope > 0:
                # Increasing Result
                if state == "Lucro":
                    evolution = "O lucro **aumenta** conforme o ativo sobe."
                elif state == "Prejuízo":
                    evolution = "O prejuízo **diminui** (melhora) conforme o ativo sobe."
                else:
                    evolution = "O resultado melhora."
            else: # slope < 0
                # Decreasing Result
                if state == "Lucro":
                    evolution = "O lucro **diminui** conforme o ativo sobe."
                elif state == "Prejuízo":
                    evolution = "O prejuízo **aumenta** conforme o ativo sobe."
                else:
                    evolution = "O resultado piora."

            # 4. Construct Text
            scenario_num = i + 1
            
            # Start/End values (if finite and not hidden)
            vals_str = ""
            if not hide_values:
                # Calc Start PnL
                pnl_start = get_pnl_at_price(lower, active_legs) if lower != -float('inf') else None
                pnl_end = get_pnl_at_price(upper, active_legs) if upper != float('inf') else None
                
                if pnl_start is not None and pnl_end is not None:
                     vals_str = f" (Caminha de R$ {pnl_start:,.2f} para R$ {pnl_end:,.2f})"
                elif pnl_start is not None: # Infinity at end
                     dir_str = "ilimitado" if slope > 0 else "perda ilimitada"
                     vals_str = f" (Parte de R$ {pnl_start:,.2f} rumo a {dir_str})"
                elif pnl_end is not None: # Infinity at start
                     vals_str = f" (Vem de valores extremos até R$ {pnl_end:,.2f})"

            explanation_text += f"**Cenário {scenario_num} ({range_desc}):**\n"
            explanation_text += f"- Zona de **{state}**. {evolution}{vals_str}\n\n"
            
        st.info(explanation_text)
        
        st.markdown("---")
        st.subheader("📤 Exportar Relatório")
        
        # Generate HTML
        html_rep = generate_html_report(
            fig, 
            active_legs, 
            explanation_text, 
            cost_label, 
            f"R$ {total_brokerage:,.2f}", 
            total_label,
            hide_values
        )
        
        c_exp1, c_exp2 = st.columns(2)
        
        with c_exp1:
             st.download_button(
                 label="Download HTML",
                 data=html_rep,
                 file_name="payoff_options.html",
                 mime="text/html"
             )
        
        with c_exp2:
            if st.button("Gerar PDF"):
                with st.spinner("Gerando PDF..."):
                    try:
                        pdf_bytes = html_to_pdf_single_page(html_rep)
                        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="payoff_options.pdf" target="_blank">Clique para Baixar PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erro ao gerar PDF: {e}")
                        st.caption("Verifique se o Playwright está instalado ou utilize a opção HTML.")
        
        st.markdown("---")
        st.caption("As informações aqui contidas têm caráter meramente indicativo e ilustrativo, não representando oferta pública, oferta particular, ou solicitação de oferta para aquisição de valores mobiliários. Não constitui, ainda, recomendação de investimento, análise de risco ou garantia de rentabilidade. Os valores e cenários apresentados são simulações baseadas em premissas variáveis e podem não refletir resultados futuros. Recomenda-se a leitura cuidadosa do regulamento/prospecto antes de qualquer decisão de investimento.")

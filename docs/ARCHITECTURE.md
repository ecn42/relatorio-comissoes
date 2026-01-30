# Architecture Overview

## System Architecture

The Dashboard Ceres Wealth follows a modular Streamlit architecture with clear separation of concerns.

## Application Structure

### Entry Point

**`relatorio_comissoes.py`** serves as the main application entry point:
- Authentication gate (password protection)
- Navigation setup with page groupings
- Home dashboard with module overview

### Page Organization

Pages are organized in `pages/` directory with numbered prefixes for ordering:

```
pages/
├── 1_comissoes.py              # Commission reports (Legacy XP)
├── 2_Relatorio_Positivador.py  # Positivator reports
├── 3_Graph_Studio.py           # Chart creation tools
├── 5_baixar_cdi.py             # CDI download utility
├── 6_rent_fundos_gt.py         # Fund performance
├── 7_factsheet.py              # Excel one-pager generation
├── 9_NEW_Portfolio_parser.py   # Portfolio parsing
├── 10_tabela_fornico.py        # Asset allocation tables
├── 11_carteira_fundos_FINAL.py # Fund portfolio management
├── 12_Gorila_API_novo.py       # Gorila API integration
├── 14_RELATORIO_CREDITO.py     # Credit reports
├── 15_Texto_Carteira_Acoes.py  # Stock portfolio text
├── 17_Pictet_to_PMV.py         # Pictet to PMV mapper
├── 18_CUSTOM_ASSETS_MANUAL_PARSER.py  # Custom assets editor
├── 19_Gorila_API_RODRIGOCABRAL.py     # Gorila API data
├── 20_TESTE_ONEPAGER_HTML.py   # HTML/PDF generation
├── 21_Add_Ratings_BR.py        # Brazilian ratings
├── 22_Rating_To_Fitch.py       # Rating conversion
├── 23_Puxar_FIDCS.py           # FIDCS retrieval
├── 25_Carteiras_RV.py          # RV portfolios
├── 26_Carteiras_Mes_Atual.py   # Current month portfolios
├── 27_Relatorio_Mercado.py     # Market reports
├── 28_Gerador_Trade_Ideas.py   # Trade idea generator
├── 29_Database_Economatica.py  # Economatica database
├── 30_Analise_Acoes_Economatica.py  # Stock analysis
├── 31_Adequacao_Perfil_Inv.py  # Investor profile matching
├── 32_Database_OnePager_Credito.py  # Credit database
├── 33_Gerar_Onepager_Credito.py     # Credit one-pager
└── ceres_logo.py               # Logo utilities
```

### Data Layer

**Databases** (`databases/`):
- SQLite databases for various data sources
- Profile management and configuration
- Cached financial data

**Data Archives** (`data_cda_fi/`):
- Historical CDA FI data in ZIP format
- Monthly archives from 2024-2025

## Module Groupings

The application organizes pages into logical sections:

### 1. Legado (XP)
Legacy reporting modules from XP platform:
- Commission reports
- Positivator reports

### 2. Ferramentas
Analysis and visualization tools:
- Chart Studio (interactive plotting)
- Carteiras RV (RV portfolio analysis)
- Trade Idea Generator
- Stock Analysis (Economatica)
- Investor Profile Matching

### 3. One Pager Fundos
Fund reporting workflow:
1. Download CDI data
2. Manage fund portfolio
3. Calculate returns
4. Generate Excel reports
5. Export HTML/PDF
6. Retrieve FIDCS data

### 4. Formatação
Asset allocation and formatting:
- Stock portfolio text generation
- Asset allocation text
- Allocation tables

### 5. Gorila/Relatórios de Risco
Risk management and credit analysis:
- Gorila API integration
- Credit reports
- Pictet to PMV mapping
- Custom asset editing
- Rating management (BR and Fitch)
- Market reports

### 6. One Pager Crédito
Credit-specific one-pager generation

### 7. Databases
Database management interfaces:
- Economatica database
- Credit one-pager database

## Technical Stack

### Core Framework
- **Streamlit**: Web application framework
- **Python 3.12**: Programming language

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **SQLite**: Local database storage

### Visualization
- **Plotly**: Interactive charts
- **Matplotlib**: Static plots
- **Streamlit-native**: Simple visualizations

### External Integrations
- **Gorila API**: Risk management data
- **Economatica**: Financial data feeds
- **Snowflake**: Cloud data warehouse (via connector)
- **Playwright**: Web scraping and automation

### Report Generation
- **Kaleido**: Plotly to image/PDF conversion
- **ReportLab**: PDF generation
- **XlsxWriter**: Excel file creation
- **python-pptx**: PowerPoint generation

### Utilities
- **Requests**: HTTP client
- **BeautifulSoup**: HTML parsing
- **Selenium/WebDriver**: Browser automation
- **python-dotenv**: Environment management

## Authentication Flow

```
User → Streamlit App → Password Check → Access Granted
                             ↓
                     st.session_state["authenticated"]
```

- Password stored in `.streamlit/secrets.toml` or environment variable
- Session-based authentication using Streamlit session state
- Simple but effective for internal tools

## Data Flow

```
External APIs (Gorila, Economatica)
         ↓
   Python Modules
         ↓
   SQLite Databases
         ↓
   Streamlit Pages
         ↓
   User Interface
```

## Docker Architecture

The containerized deployment uses:
- **Base Image**: `python:3.12.11-slim-bookworm`
- **Chrome**: Embedded for Kaleido PDF generation
- **Multi-stage**: Dependencies installed before app code
- **Port**: 8080 exposed

See `Dockerfile` for complete configuration.

## Extension Points

To add new functionality:

1. **New Page**: Create `pages/XX_page_name.py`
2. **Update Navigation**: Add to `relatorio_comissoes.py` pages dictionary
3. **Database**: Add SQLite file to `databases/` or use existing
4. **API Integration**: Create service module in new `src/` folder

## Performance Considerations

- Databases are SQLite (file-based, no server needed)
- Caching via Streamlit's `@st.cache_data` decorator
- Large data archives in `data_cda_fi/` are gitignored
- Playwright browsers cached in container image

## Security Model

- Single password authentication
- No multi-user support (single tenant)
- Data stored locally in SQLite
- Secrets managed via environment or Streamlit secrets
- No encryption at rest for databases

## Future Improvements

- Move business logic to `src/` modules
- Add comprehensive test suite
- Implement user management
- Add API rate limiting
- Database encryption for sensitive data

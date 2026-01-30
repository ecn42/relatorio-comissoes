# Dashboard Ceres Wealth

A comprehensive Streamlit-based dashboard for financial reporting and analysis, designed for wealth management operations.

## Overview

This dashboard provides tools for:
- Commission reporting and analysis
- Portfolio management and fund analysis
- Credit reporting and risk assessment
- Market analysis and trade idea generation
- Custom asset allocation and formatting

## Features

### Core Modules

| Module | Description |
|--------|-------------|
| **Legado (XP)** | Legacy commission and positivator reports |
| **Ferramentas** | Chart studio, stock analysis, trade idea generator |
| **One Pager Fundos** | Fund portfolio management and reporting |
| **Formatação** | Asset allocation formatting tools |
| **Gorila/Relatórios de Risco** | Risk reports and credit analysis |
| **One Pager Crédito** | Credit one-pager generation |
| **Databases** | Database management tools |

### Key Capabilities

- 📊 **Interactive Charts**: Plotly and Streamlit visualizations
- 📈 **Financial Analysis**: Stock comparison, fund performance tracking
- 📄 **Report Generation**: HTML/PDF one-pagers, Excel exports
- 🔗 **API Integrations**: Gorila API, Economatica data feeds
- 🗄️ **Database Management**: SQLite databases for various data sources
- 🔐 **Authentication**: Password-protected access

## Architecture

```
.
├── relatorio_comissoes.py    # Main application entry point
├── pages/                     # Streamlit page modules
│   ├── 1_comissoes.py
│   ├── 2_Relatorio_Positivador.py
│   └── ...
├── databases/                 # SQLite databases
├── data_cda_fi/              # CDA FI data archives
├── defasado/                 # Deprecated modules
├── Dockerfile                # Container configuration
└── requirements.txt          # Python dependencies
```

## Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)
- Chrome/Chromium (for PDF generation via Kaleido)

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd relatorio-comissoes
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   ```bash
   mkdir -p .streamlit
   echo 'auth = { PASSWORD = "your-password" }' > .streamlit/secrets.toml
   ```

5. **Run the application**
   ```bash
   streamlit run relatorio_comissoes.py
   ```

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t ceres-dashboard .
   ```

2. **Run the container**
   ```bash
   docker run -p 8080:8080 \
     -e PASSWORD=your-password \
     ceres-dashboard
   ```

   Or use the provided `entrypoint.sh` which reads from environment variables.

## Configuration

### Secrets

Create `.streamlit/secrets.toml` with:
```toml
[auth]
PASSWORD = "your-secure-password"
```

For production, use environment variables instead of the secrets file.

### Environment Variables

- `PASSWORD`: Authentication password
- `BROWSER_PATH`: Path to Chrome/Chromium for PDF generation
- `PLAYWRIGHT_BROWSERS_PATH`: Playwright browsers installation path

## Usage

1. Access the dashboard at `http://localhost:8080` (or configured port)
2. Login with the configured password
3. Navigate through different sections using the top menu
4. Each module provides specific financial tools and reports

## Documentation

- [Architecture Overview](./docs/ARCHITECTURE.md)
- [Setup Guide](./docs/SETUP.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

## Data Sources

The dashboard integrates with multiple data sources:
- CDA FI (Complementary Data from Investment Funds)
- Economatica financial data
- Gorila risk management platform
- Internal SQLite databases

## Security

- Password-based authentication
- Secrets managed via environment variables or Streamlit secrets
- No sensitive data committed to version control

## Maintenance

### Deprecated Files

Deprecated modules are stored in `defasado/` for reference but excluded from version control via `.gitignore`.

### Cache Files

`__pycache__/` directories are automatically excluded from version control.

## Support

For issues or questions, please contact the development team or open an issue in the repository.

## License

[License](./LICENSE) - Proprietary software for Ceres Wealth internal use.

---

**Note**: This is an internal tool. Ensure all financial data handling complies with company policies and regulatory requirements.

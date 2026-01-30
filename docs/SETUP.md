# Setup Guide

Complete setup instructions for the Dashboard Ceres Wealth.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Setup](#docker-setup)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Python 3.12+**: [Download](https://www.python.org/downloads/)
- **Git**: For cloning the repository
- **Chrome/Chromium** (optional): For PDF generation features

### System Requirements

- **OS**: Linux, macOS, or Windows
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB for application + data

## Local Development Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd relatorio-comissoes
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes due to heavy dependencies (Playwright, Snowflake connectors, etc.)

### Step 4: Install Playwright Browsers

```bash
playwright install chromium
```

### Step 5: Configure Secrets

Create the secrets file:

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
[auth]
PASSWORD = "your-secure-password-here"
EOF
```

**Security Note**: Never commit this file. It's already in `.gitignore`.

### Step 6: Run Application

```bash
streamlit run relatorio_comissoes.py
```

The app will open at `http://localhost:8501` (or another port if 8501 is in use).

## Docker Setup

### Building the Image

```bash
docker build -t ceres-dashboard:latest .
```

Build time: ~10-15 minutes (includes Chrome installation)

### Running the Container

**Basic run:**
```bash
docker run -p 8080:8080 \
  -e PASSWORD=your-secure-password \
  ceres-dashboard:latest
```

**With volume for data persistence:**
```bash
docker run -p 8080:8080 \
  -e PASSWORD=your-secure-password \
  -v $(pwd)/databases:/app/databases \
  ceres-dashboard:latest
```

**Detached mode:**
```bash
docker run -d \
  --name ceres-dashboard \
  -p 8080:8080 \
  -e PASSWORD=your-secure-password \
  ceres-dashboard:latest
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PASSWORD=${PASSWORD}
    volumes:
      - ./databases:/app/databases
    restart: unless-stopped
```

Run:
```bash
export PASSWORD=your-secure-password
docker-compose up -d
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PASSWORD` | Authentication password | Required |
| `BROWSER_PATH` | Chrome/Chromium binary path | `/opt/chrome/chrome-wrapper` |
| `PLAYWRIGHT_BROWSERS_PATH` | Playwright browsers location | `/ms-playwright` |
| `STREAMLIT_SERVER_PORT` | Streamlit server port | `8080` |
| `STREAMLIT_SERVER_ADDRESS` | Server bind address | `0.0.0.0` |

### Streamlit Configuration

Create `.streamlit/config.toml` for additional settings:

```toml
[server]
port = 8080
address = "0.0.0.0"
headless = true

[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Database Setup

The application uses SQLite databases stored in `databases/`:

- `data_fundos.db`: Fund data
- `trade_ideas.db`: Trade ideas
- `dados_economatico.db`: Economatica data
- `onepager_credito.db`: Credit one-pager data
- `cnpj-fundos.db`: CNPJ fund information
- `commission_data.db`: Commission data
- `carteira_fundos.db`: Fund portfolio data
- `carteiras_rv_mes_atual.db`: Current month RV portfolios
- `rentabilidade_carteiras_btg.db`: BTG portfolio returns
- `df_debentures.db`: Debentures data

**Note**: Databases are gitignored but tracked in development. Ensure production databases are backed up.

## Troubleshooting

### Common Issues

#### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find and kill process using port 8080
lsof -ti:8080 | xargs kill -9

# Or use different port
streamlit run relatorio_comissoes.py --server.port 8502
```

#### Playwright Browser Not Found

**Error**: `Browser not found` or `Executable doesn't exist`

**Solution**:
```bash
playwright install chromium
playwright install-deps chromium
```

#### Chrome/Chromium Issues (Docker)

**Error**: PDF generation fails with Chrome errors

**Solution**: The Dockerfile already handles this, but if issues persist:
```bash
docker run --rm -it ceres-dashboard /opt/chrome/chrome-wrapper --version
```

#### Permission Denied (Linux/macOS)

**Error**: `Permission denied` when running scripts

**Solution**:
```bash
chmod +x entrypoint.sh
```

#### Memory Issues

**Error**: `Killed` or out of memory during pip install

**Solution**:
```bash
# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Or increase swap space
```

#### Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure virtual environment is activated
which python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Debug Mode

Enable debug logging:

```bash
export STREAMLIT_LOG_LEVEL=debug
streamlit run relatorio_comissoes.py
```

### Getting Help

1. Check application logs in `gorila_debug.log`
2. Review error in `debug_graph_error.txt`
3. Open an issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version)

## Next Steps

After successful setup:

1. **Login**: Use the password configured in secrets
2. **Explore**: Navigate through different modules
3. **Customize**: Modify page configurations in `relatorio_comissoes.py`
4. **Extend**: Add new pages following the existing pattern

See [Architecture Overview](./ARCHITECTURE.md) for technical details.

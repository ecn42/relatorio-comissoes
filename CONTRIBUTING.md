# Contributing to Dashboard Ceres Wealth

Thank you for your interest in contributing to the Dashboard Ceres Wealth! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Submitting Changes](#submitting-changes)
5. [Review Process](#review-process)
6. [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- Access to the repository
- Docker (for testing)

### Setup

1. Fork the repository (if external contributor)
2. Clone your fork:
   ```bash
   git clone <your-fork-url>
   cd relatorio-comissoes
   ```
3. Set up development environment (see [docs/SETUP.md](./docs/SETUP.md))
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/add-new-chart-type`
- `bugfix/fix-login-redirect`
- `docs/update-readme`
- `refactor/optimize-database-queries`

### Commit Messages

Follow conventional commit format:
```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(reports): add export to PowerPoint functionality

fix(api): handle timeout errors in Gorila API calls

docs: update deployment guide with AWS ECS instructions
```

### Code Organization

When adding new features:

1. **New Pages**: Add to `pages/` with numbered prefix
2. **Shared Code**: Create `src/` modules for reusable code
3. **Configuration**: Update `relatorio_comissoes.py` for navigation
4. **Tests**: Add corresponding tests in `tests/`

Example structure for new feature:
```
.
├── pages/35_New_Analysis_Tool.py
├── src/
│   └── analysis/
│       └── calculator.py
└── tests/
    └── test_calculator.py
```

## Code Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 88 characters (Black formatter default)
- **Quotes**: Double quotes for strings
- **Imports**: Grouped by standard lib, third-party, local
- **Type hints**: Use where appropriate

### Formatting

Use Black for consistent formatting:
```bash
pip install black
black .
```

### Linting

Use Ruff for fast linting:
```bash
pip install ruff
ruff check .
ruff check . --fix  # Auto-fix issues
```

### Type Checking

Use mypy for type checking:
```bash
pip install mypy
mypy src/
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings:
  ```python
  def calculate_returns(data: pd.DataFrame) -> float:
      """Calculate total returns from price data.
      
      Args:
          data: DataFrame with 'price' and 'date' columns
          
      Returns:
          Total return as a percentage
          
      Raises:
          ValueError: If data is empty or missing required columns
      """
  ```

### Streamlit Best Practices

1. **Caching**: Use `@st.cache_data` for expensive operations
2. **Session State**: Use `st.session_state` for user state
3. **Error Handling**: Wrap external API calls in try-except
4. **Progress**: Show progress for long operations
5. **Layouts**: Use `st.columns()` and `st.container()` for organization

Example:
```python
@st.cache_data(ttl=3600)
def fetch_market_data(ticker: str) -> pd.DataFrame:
    """Fetch market data with caching."""
    return yf.download(ticker)

with st.spinner("Loading data..."):
    data = fetch_market_data("AAPL")
```

### Database Guidelines

- Use SQLAlchemy ORM for database operations
- Add migration scripts for schema changes
- Never commit sensitive data to databases
- Include indexes for frequently queried columns

## Submitting Changes

### Before Submitting

1. **Test locally**:
   ```bash
   streamlit run relatorio_comissoes.py
   ```

2. **Run linting and formatting**:
   ```bash
   black .
   ruff check .
   ```

3. **Update documentation**:
   - Add/update relevant documentation in `docs/`
   - Update README.md if needed

4. **Check for secrets**:
   ```bash
   git diff --cached | grep -i "password\|secret\|key\|token"
   ```

### Pull Request Process

1. **Create PR** with descriptive title:
   - `feat: Add credit risk visualization dashboard`
   - `fix: Resolve timeout in Economatica data fetch`

2. **Fill out PR template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tested locally
   - [ ] Added unit tests
   - [ ] Tested with Docker
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No secrets committed
   ```

3. **Request review** from at least one team member

4. **Address feedback** promptly

5. **Merge** after approval (squash and merge for clean history)

## Review Process

### Reviewer Guidelines

1. **Check functionality**: Does it work as intended?
2. **Code quality**: Is it maintainable and well-structured?
3. **Security**: Are there any security concerns?
4. **Performance**: Any obvious performance issues?
5. **Tests**: Are there adequate tests?
6. **Documentation**: Is it documented?

### Review Checklist

- [ ] Code follows project conventions
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is appropriate
- [ ] No unnecessary dependencies added
- [ ] Documentation is clear and accurate
- [ ] Backwards compatibility maintained (or breaking changes documented)

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backwards compatible)
- Patch: Bug fixes

### Release Steps

1. **Update version** in relevant files
2. **Update CHANGELOG.md** with release notes
3. **Create release branch**:
   ```bash
   git checkout -b release/v1.2.0
   ```
4. **Final testing** in staging environment
5. **Merge to main** via PR
6. **Tag release**:
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```
7. **Deploy** to production
8. **Monitor** for issues

### Hotfix Process

For critical production fixes:

1. Create branch from `main`:
   ```bash
   git checkout -b hotfix/fix-critical-bug
   ```
2. Make minimal fix
3. Fast-track review
4. Merge and deploy immediately
5. Tag with patch version bump

## Questions?

If you have questions about contributing:
- Review existing code for examples
- Check [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for technical details
- Ask in team chat or open a discussion issue

## Code of Conduct

### Our Standards

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the team and clients
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other unethical or unprofessional conduct

## Attribution

This contributing guide is adapted from various open-source project templates and customized for Ceres Wealth internal development practices.

---

Thank you for contributing to Dashboard Ceres Wealth!

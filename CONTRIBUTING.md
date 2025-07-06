# Contributing to Solar Power ML Pipeline

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/solar-power-ml-pipeline.git
   cd solar-power-ml-pipeline
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Code Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Documentation
- Write clear docstrings for all functions and classes
- Update README.md for significant changes
- Add inline comments for complex logic
- Keep documentation up to date

### Testing
- Write unit tests for new functions
- Maintain test coverage above 80%
- Use descriptive test names
- Test edge cases and error conditions

## Contribution Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests**
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Types of Contributions

### Bug Reports
- Use the bug report template
- Include steps to reproduce
- Provide system information
- Include relevant logs or error messages

### Feature Requests
- Use the feature request template
- Explain the use case
- Describe the proposed solution
- Consider implementation complexity

### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation updates
- Test improvements

## Review Process

1. **Automated Checks**
   - All CI/CD checks must pass
   - Code coverage must not decrease
   - No linting errors

2. **Manual Review**
   - Code quality and style
   - Test coverage and quality
   - Documentation completeness
   - Performance considerations

3. **Approval and Merge**
   - At least one approval required
   - Squash and merge preferred
   - Delete feature branch after merge

## Release Process

1. **Version Bumping**
   - Follow semantic versioning (MAJOR.MINOR.PATCH)
   - Update version in relevant files
   - Update CHANGELOG.md

2. **Release Notes**
   - Summarize new features
   - List bug fixes
   - Note breaking changes
   - Include migration instructions if needed

## Questions?

If you have questions about contributing, please:
- Check existing issues and discussions
- Create a new discussion for general questions
- Create an issue for specific problems
- Contact maintainers directly if needed

Thank you for contributing! 🎉

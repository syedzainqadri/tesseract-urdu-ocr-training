# Contributing to Tesseract Urdu OCR Training System

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tesseract-urdu-ocr-training.git
   cd tesseract-urdu-ocr-training
   ```
3. **Set up the environment**
   ```bash
   ./setup_environment.sh
   ```

## 🛠️ Development Setup

### Prerequisites
- macOS (primary support)
- Python 3.x with tkinter
- Tesseract 4.x with training tools
- Homebrew

### Installation
```bash
# Install dependencies
brew install tesseract tesseract-lang python-tk

# Install Python packages
pip3 install Pillow pytesseract numpy
```

## 📝 How to Contribute

### 1. Dataset Improvements
- **Add new fonts**: Update `FONTS` dictionary in `create_dataset.py`
- **Improve text samples**: Expand `URDU_TEXTS` array with diverse content
- **New image variations**: Add processing functions in `apply_variation()`

### 2. GUI Enhancements
- **Training interface**: Improve `tesseract_gui.py`
- **OCR testing**: Enhance `enhanced_ocr_app.py`
- **Progress monitoring**: Add new metrics and visualizations

### 3. Documentation
- **Usage examples**: Add screenshots and tutorials
- **API documentation**: Document functions and classes
- **Troubleshooting**: Add common issues and solutions

### 4. Testing
- **Unit tests**: Add tests for core functions
- **Integration tests**: Test complete workflows
- **Performance tests**: Benchmark training and OCR speed

## 🔧 Code Style

### Python
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused and small

### Shell Scripts
- Use bash shebang: `#!/bin/bash`
- Add error checking with `set -e`
- Include helpful comments
- Use descriptive variable names

## 📋 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, focused commits
   - Test your changes thoroughly
   - Update documentation if needed

3. **Submit a pull request**
   - Use a descriptive title
   - Explain what your changes do
   - Include screenshots for UI changes
   - Reference any related issues

## 🐛 Bug Reports

When reporting bugs, please include:
- **Environment**: OS version, Python version, Tesseract version
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Screenshots**: If applicable
- **Error messages**: Full error output

## 💡 Feature Requests

For new features, please:
- **Check existing issues** to avoid duplicates
- **Describe the use case** clearly
- **Explain the benefit** to users
- **Consider implementation** complexity

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private inquiries

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special thanks for major improvements

Thank you for helping make this project better! 🎉

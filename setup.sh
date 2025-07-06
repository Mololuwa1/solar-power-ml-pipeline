#!/bin/bash

# Solar Power ML Pipeline Setup Script

echo "🌟 Setting up Solar Power ML Pipeline..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
required_version="3.8"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install development requirements (optional)
read -p "Install development requirements? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
    echo "✅ Development requirements installed"
fi

# Set up pre-commit hooks (if dev requirements installed)
if command -v pre-commit &> /dev/null; then
    echo "🪝 Setting up pre-commit hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/raw data/processed data/baseline
mkdir -p models/trained models/artifacts
mkdir -p logs

# Set up AWS CLI (optional)
read -p "Configure AWS CLI? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v aws &> /dev/null; then
        aws configure
        echo "✅ AWS CLI configured"
    else
        echo "⚠️ AWS CLI not found. Please install it manually."
    fi
fi

# Create sample configuration
echo "⚙️ Creating sample configuration..."
cp config/model_config.json config/model_config_local.json
cp config/aws_config.json config/aws_config_local.json

echo "📝 Please update the configuration files in config/ with your specific settings."

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update configuration files in config/"
echo "2. Add your data to data/raw/"
echo "3. Run: jupyter notebook notebooks/01_data_exploration.ipynb"
echo ""
echo "For more information, see README.md"

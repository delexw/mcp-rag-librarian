#!/bin/bash

# Build and Upload Script for MCP RAG Librarian
# Usage: ./publish.sh [test|prod]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦸‍♀️ MCP RAG Librarian Publisher${NC}"
echo "=================================="

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo -e "${RED}Error: pyproject.toml not found. Run this script from the project root.${NC}"
    exit 1
fi

# Parse arguments
MODE="prod"
if [[ $1 == "test" ]]; then
    MODE="test"
elif [[ $1 == "prod" ]] || [[ -z $1 ]]; then
    MODE="prod"
else
    echo -e "${RED}Usage: $0 [test|prod]${NC}"
    echo "  test - Upload to TestPyPI"
    echo "  prod - Upload to PyPI (default)"
    exit 1
fi

echo -e "${YELLOW}Mode: $MODE${NC}"

# Clean previous builds
echo -e "${BLUE}🧹 Cleaning previous builds...${NC}"
rm -rf dist/ build/ src/*.egg-info/

# Install/upgrade build tools
echo -e "${BLUE}🔧 Installing/upgrading build tools...${NC}"
pip install --upgrade build twine

# Build the package
echo -e "${BLUE}📦 Building package...${NC}"
python -m build

# Check if build was successful
if ! ls dist/*.whl 1> /dev/null 2>&1 || ! ls dist/*.tar.gz 1> /dev/null 2>&1; then
    echo -e "${RED}Build failed! No distribution files found.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build successful!${NC}"
ls -la dist/

# Check the package before upload
echo -e "${BLUE}🔍 Checking package...${NC}"
python -m twine check dist/*

# Upload based on mode
if [[ $MODE == "test" ]]; then
    echo -e "${YELLOW}🧪 Uploading to TestPyPI...${NC}"

    if [[ -n "$TWINE_PASSWORD" ]]; then
        echo "Using token from environment variable"
        python -m twine upload --repository testpypi dist/* --username __token__ --password "$TWINE_PASSWORD"
    else
        echo "Set your TestPyPI token: export TWINE_PASSWORD=pypi-your-token-here"
        echo "Or get one from: https://test.pypi.org/manage/account/token/"
        python -m twine upload --repository testpypi dist/*
    fi

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Successfully uploaded to TestPyPI!${NC}"
        echo -e "${BLUE}Test installation:${NC}"
        echo "pip install --index-url https://test.pypi.org/simple/ mcp-rag-librarian"
    fi

elif [[ $MODE == "prod" ]]; then
    echo -e "${RED}⚠️  WARNING: You're about to upload to PRODUCTION PyPI!${NC}"
    echo -e "${YELLOW}This cannot be undone. Make sure you've tested thoroughly.${NC}"
    echo ""
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🚀 Uploading to PyPI...${NC}"

        if [[ -n "$TWINE_PASSWORD" ]]; then
            echo "Using token from environment variable"
            python -m twine upload dist/* --username __token__ --password "$TWINE_PASSWORD"
        else
            echo "Set your PyPI token: export TWINE_PASSWORD=pypi-your-token-here"
            echo "Or get one from: https://pypi.org/manage/account/token/"
            python -m twine upload dist/*
        fi

        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}🎉 Successfully published MCP RAG Librarian!${NC}"
            echo -e "${BLUE}Installation:${NC}"
            echo "pip install mcp-rag-librarian"
            echo "uvx mcp-rag-librarian"
            echo ""
            echo -e "${BLUE}Claude Code setup:${NC}"
            echo "claude mcp add mcp-rag-librarian"
            echo ""
            echo -e "${GREEN}Your superhero librarian is now live! 📚🦸‍♀️✨${NC}"
        fi
    else
        echo -e "${YELLOW}Upload cancelled.${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${GREEN}Done! 🎉${NC}"
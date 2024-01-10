#!/bin/bash
set -e

echo "Running mypy..."
mypy infobip_service

echo "Running bandit..."
bandit -c pyproject.toml -r infobip_service

echo "Running semgrep..."
semgrep scan --config auto --error

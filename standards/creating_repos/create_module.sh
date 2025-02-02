#!/bin/bash

# To make this script executable:`chmod +x create_module.sh`
# To run: `./create_module.sh`

# Get project and module names
read -p "Enter project name: " project_name
read -p "Install LLM dependencies [y/n]: " INSTALL_LLM_DEPS
read -p "Make installable package: [y/n]" INSTALLABLE_PKG

# Pin Python 3.11
PYTHON_VERSION=3.11

# Check of project dir already exists
if [ -d "$project_name" ]; then
    echo "Directory $project_name already exists"
    exit 1
fi

# Create root directory
mkdir -p "$project_name"
cd "$project_name"

# Create directory structure
<<COMMENT
# Project Structure
{project_name}/
├── README.md
├── LICENSE            # Open-source license
├── CHANGELOG.md       # Changelog with format based on https://keepachangelog.com/en/1.1.0/
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .env               # Python-dotenv reads key-value pairs from .env and sets them as environment variables
├── docs/              # Project docs
├── output_data/       # Data generated from scripts
├── input_data/        # Static, externally obtained input data files
├── models/            # Trained and serialized models and model details
├── references/        # Static, external reference documents 
├── reports/           # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/       # Generated graphics and figures to be used in reporting
├── scripts/
├── notebooks/         # Jupyter notebooks
├── src/
│   └── {project_name}/
│       └── __init__.py
└── tests/
COMMENT

# Create directory structure
mkdir -p src/"$project_name" tests docs references scripts notebooks output_data input_data models
touch src/"$project_name"/__init__.py
mkdir -p reports/figures

# Create empty files
touch README.md LICENSE CHANGELOG.md
touch .gitignore .pre-commit-config.yaml .python-version .env

# Add README.md
cat > README.md << EOL
# ${project_name}
EOL

# Create LICENSE with BSD 3-Clause
cat > LICENSE << EOL
BSD 3-Clause License

Copyright (c) $(date +%Y), ${project_name} developers

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
EOL

# Create pyproject.toml
cat > pyproject.toml << EOL
[project]
name = "${project_name}"
version = "0.1.0"
description = ""
authors = []
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=${PYTHON_VERSION},<3.12"
classifiers = [
  "Programming Language :: Python",
  "Programming Language :: Python :: 3 :: Only",
]
EOL

if [ "$INSTALLABLE_PKG" = "y" ]; then
    cat >> pyproject.toml << EOL
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOL
fi

# Create .gitignore
cat > .gitignore << EOL
# Byte-compiled / optimized / DLL files
__pycache__/

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
.venv

# Testing and linting
.pytest_cache/
.ruff_cache/

# IDEs
.idea/
.vscode/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Logs
*.log
logs/
*.out

# Environment variables
.env
EOL

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOL

# RUN ``pre-commit install`` to set up the pre-commit hooks

# to manually run all pre-commit hooks on a repository: 
# ``pre-commit run --all-files``

# To run individual hooks
# ``pre-commit run <hook_id>``

# Use ``git commit --no-verify`` or the shorter form
# ``git commit -n`` to skip pre-commit hooks for a
# specific commit. Example:
# ``git commit -n -m "<git commit msg>"``

repos:
- repo: local
  hooks:
    - id: ruff
      name: ruff
      description: "ruff: Python linting"
      entry: ruff check . --fix
      language: python
      types_or: [python]
      require_serial: true
    - id: ruff-format
      name: ruff-format
      description: "ruff: Python formatting"
      entry: ruff format .
      language: python
      types_or: [python]
      require_serial: true
EOL

# Finish
echo "Project structure created successfully in ./${project_name}"

# Initialize conda for shell
eval "$(conda shell.bash hook)"

# Check if uv is installed in conda base env, if so update, else install
conda activate base
if conda list | grep -q "^uv "; then
   echo "Updating uv..."
   conda update -n base uv -y
else
   echo "Installing uv..."
   conda install -n base uv -y
fi

# Create a virtual environment with uv
uv venv --python ${PYTHON_VERSION}

# Deactivate base environment
conda deactivate

# Activate venv before installing
source .venv/bin/activate

# Add core dependencies
uv add requests pandas openpyxl altair vegafusion vegafusion-python-embed vl-convert-python seaborn statsmodels scikit-learn scikit-learn-intelex python-dotenv

if [ "$INSTALL_LLM_DEPS" = "y" ] || [ "$INSTALL_LLM_DEPS" = "yes" ]; then
    uv add transformers gradio tokenizers huggingface-hub optimum accelerate bitsandbytes safetensors einops sentencepiece
fi

# Add dev dependencies
uv add --group dev uv ruff pre-commit jupyterlab pytest

# Set up pre-commit hooks
git init .
pre-commit install
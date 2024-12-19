# Variables
PYTHON = python
PIP = pip
VENV_DIR = venv
SRC_DIR = src
TEST_DIR = tests

# Règles
.PHONY: all setup install_deps run_tests clean

all: setup install_deps run_tests

setup:
    @echo "Setting up virtual environment..."
    $(PYTHON) -m venv $(VENV_DIR)
    @echo "Virtual environment created."

install_deps: setup
    @echo "Installing dependencies..."
    $(VENV_DIR)/Scripts/$(PIP) install -r requirements.txt
    @echo "Dependencies installed."

run_tests:
    @echo "Running tests..."
    $(VENV_DIR)/Scripts/$(PYTHON) -m unittest discover -s $(TEST_DIR)
    @echo "Tests completed."

clean:
    @echo "Cleaning up..."
    rm -rf $(VENV_DIR)
    find . -type f -name '*.pyc' -delete
    find . -type d -name '__pycache__' -delete
    @echo "Cleanup completed."
#!/bin/bash

# Upgrade pip to its latest version
python3 -m pip install --upgrade pip

# Update the required Python libraries
pip install --upgrade matplotlib numpy pandas scikit-learn tensorflow joblib

pip install numba


echo "Libraries have been updated to their latest versions."

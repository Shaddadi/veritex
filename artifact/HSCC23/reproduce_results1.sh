#!/bin/bash

# This script automates the verification and repair processes for ACASXu and other models,
# and generates figures and tables for reporting results.
# It uses absolute paths to ensure reliability across different execution contexts.

# Define the script's directory to ensure robustness against execution from different locations.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Verification of ACASXu models.
# It runs the verification process for all instances of ACASXu.
python3 "$SCRIPT_DIR/../../examples/ACASXu/verify/verify_all_instances.py"

# Repair of ACASXu using Veritex.
# It applies repair algorithms to the neural networks of ACASXu.
python3 "$SCRIPT_DIR/../../examples/ACASXu/repair/repair_nnets.py"

# Repair of DNN agents in DRL (Deep Reinforcement Learning) using Veritex.
# It focuses on repairing neural network agents within a DRL setting.
python3 "$SCRIPT_DIR/../../examples/DRL/repair/repair_nnets.py"

# Repair of ACASXu using ART (Adversarial Robustness Toolbox).
# It specifically uses the ART toolbox for experimenting with ACASXu model repairs.
python3 "$SCRIPT_DIR/../../artifact/HSCC23/ART/art/exp_acas.py"

# Generate figures and tables for the experimental results.
# This includes generating data for Figures 2, Figures 3 and 4, and Tables 2 and 3,
# which are likely used for visualization and analysis in reports or publications.
python3 "$SCRIPT_DIR/../../artifact/HSCC23/figure2_generator.py"
python3 "$SCRIPT_DIR/../../artifact/HSCC23/figure_3_4_generator.py"
python3 "$SCRIPT_DIR/../../artifact/HSCC23/table_2_3_generator.py"

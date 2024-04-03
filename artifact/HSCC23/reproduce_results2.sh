#!/bin/bash

# Path to the script's directory, regardless of where it's called from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Verification of ACASXu
python3 "$SCRIPT_DIR/../../examples/ACASXu/verify/verify_all_instances.py"

# Repair of ACASXu using veritex
python3 "$SCRIPT_DIR/../../examples/ACASXu/repair/repair_nnets.py" --all

# Repair of DNN agent in DRL using veritex
python3 "$SCRIPT_DIR/../../examples/DRL/repair/repair_nnets.py"

# Repair of ACASXu using ART
python3 "$SCRIPT_DIR/../../artifact/HSCC23/ART/art/exp_acas.py"

# Display results
python3 "$SCRIPT_DIR/../../artifact/HSCC23/figure2_generator.py"
python3 "$SCRIPT_DIR/../../artifact/HSCC23/figure_3_4_generator.py"
python3 "$SCRIPT_DIR/../../artifact/HSCC23/table_2_3_generator.py"
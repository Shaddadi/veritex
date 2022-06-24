#!/bin/bash

# repair of acasxu using veritex
cd ../../examples/ACASXu/repair
python3 repair_nnets.py
python3 repair_nnets_minimal.py

# repair of acasxu using art
cd ../../../artifact/FORMATS22/ART/art
python3 exp_acas.py

# display results
cd ../../
python3 figure3_generator.py
python3 figure4_5_generator.py
python3 table1_2_generator.py



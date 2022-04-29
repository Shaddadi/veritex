#!/bin/bash

# verification of acasxu
cd ../examples/ACASXu/verify
python3 verify_all_instances.py

# repair of acasxu
cd ../repair
python3 repair_nnets.py -all

# repair of DNN agent in DRL using veritex
cd ../../DRL/repair
python3 repair_nnets.py

# ART
cd ../../../cav22_artifact/ART/art
python3 exp_acas.py

# display results
cd ../../../cav22_artifact
python3 figure2_generator.py
python3 figure3\&4_generator.py
python3 table2\&3_generator.py



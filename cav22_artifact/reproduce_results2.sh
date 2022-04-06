#!/bin/bash

# verification of acasxu
cd ../examples/ACASXu/verify
python3 verify_p1-p4_vnnlib.py
python3 verify_p5_vnnlib.py
python3 verify_p6_vnnlib.py
python3 verify_p7_vnnlib.py
python3 verify_p8_vnnlib.py
python3 verify_p9_vnnlib.py
python3 verify_p10_vnnlib.py

# repair of acasxu
cd ../repair
python3 repair_all_nnets.py

cd ../../../cav22_artifact/ART/art
python3 exp_acas.py

# display results
cd ../../../cav22_artifact
python3 figure2_generator.py
python3 figure3\&4_generator.py
python3 table2\&3_generator.py



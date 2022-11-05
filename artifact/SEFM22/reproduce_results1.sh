#!/bin/bash

# verification of acasxu
cd ../../examples/ACASXu/verify
python3 verify_all_instances.py

# repair of acasxu using veritex
cd ../repair
python3 repair_nnets.py

# repair of DNN agent in DRL using veritex
cd ../../DRL/repair
python3 repair_nnets.py

# repair of acasxu using art
cd ../../../artifact/SEFM22/ART/art
python3 exp_acas.py

# display results
cd ../../
python3 figure2_generator.py
python3 figure_3_4_generator.py
python3 table_2_3_generator.py


#!/bin/bash

TIMEOUT=20m

timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 2 --n1 1 --n2 2 --compute_unsafety
timeout --foreground --signal=SIGQUIT $TIMEOUT matlab -nodisplay -nodesktop -r 'run display_unsafe_inputs.m;clear;quit'


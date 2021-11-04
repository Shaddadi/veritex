#!/bin/bash

TIMEOUT=1h

timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --n1 1 --n2 1
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --n1 1 --n2 2
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --n1 1 --n2 3
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --n1 1 --n2 4
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --n1 1 --n2 5




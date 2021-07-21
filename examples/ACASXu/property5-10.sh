#!/bin/bash

TIMEOUT=2h

timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 5 --n1 1 --n2 1
sleep 1.0
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 6.1 --n1 1 --n2 1
sleep 1.0
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 6.2 --n1 1 --n2 1
sleep 1.0
#timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 7 --n1 1 --n2 9
#sleep 1.0
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 8 --n1 2 --n2 9
sleep 1.0
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 9 --n1 3 --n2 3
sleep 1.0
timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py --property 10 --n1 4 --n2 5
sleep 1.0



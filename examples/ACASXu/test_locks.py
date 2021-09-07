import numpy as np
import multiprocessing as mp
import time

def main_func0(shared_state0, shared_state1):
    with shared_state0.get_lock():
        print('In the lock')
        # shared_state0[0] = 1
        # print('changed shared_state0[0]')
        time.sleep(5)
        shared_state1[1] = 1

# def main_func1(shared_state):
#     while True:
#         print('assigning')
#         shared_state[1] = 1
#         print('assigned')
#         print(shared_state[1])

def main_func1(shared_state0, shared_state1):
    while True:
        print('a loop')
        with shared_state0.get_lock():
            if shared_state0[1] == 0:
                print('passed')

if __name__ == "__main__":

    shared_state0 = mp.Array('i', [0]*2, lock=False)
    shared_state1 = mp.Array('i', [0] * 2)
    p0 = mp.Process(target= main_func0, args=(shared_state0, shared_state1))
    p1 = mp.Process(target= main_func1, args=(shared_state0, shared_state1))
    p0.start()
    p1.start()
    p0.join()
    p0.join()



import random
import time
import numpy as np
import copy as cp
from collections import deque

idle_workers = 1
class Worker:
    def __init__(self, dnn):
        self.dnn = dnn
        self.private_deque = deque()
        self.output_sets = []
        self.worker_id = None
        self.shared_state = None


    def inital_assgin(self):
        # get inputs from shared state
        while True:
            try:
                one_work = self.shared_state.shared_queue.get_nowait()
                self.private_deque.append(one_work)
                with self.shared_state.shared_queue_len.get_lock():
                    self.shared_state.shared_queue_len.value = 0
            except:
                break

        with self.shared_state.initial_completed_workers.get_lock():
            self.shared_state.initial_completed_workers.value += 1

        if self.shared_state.initial_completed_workers.value == self.shared_state.num_workers:
            self.shared_state.initial_steal_assign.set()
        else:
            self.shared_state.initial_steal_assign.wait()


    def main_func(self, indx, shared_state):
        self.worker_id = indx
        self.shared_state = shared_state
        self.inital_assgin()

        # initial computation, breath-first computation
        while self.private_deque:

            tuple_state = self.private_deque.popleft()
            self.state_spawn_breath_first(tuple_state)

            if len(self.private_deque) >= 5000:

                self.shared_state.steal_assign_ready.set()
                print('This is Worker '+str(self.worker_id))
                with self.shared_state.num_valid_busy_workers.get_lock():
                    self.shared_state.num_valid_busy_workers.value -= 1

                random.shuffle(self.private_deque)
                self.shared_state.compute_steal_rate()
                self.steal_from_this_worker()
                # self.shared_state.work_assign_ready.wait()
                with self.shared_state.initial_comput.get_lock(): # disable it forever
                    self.shared_state.initial_comput.value = 0
                break

        # print('Worker '+str(self.worker_id)+' works: ', len(self.private_deque))

        # normal computation, depth-first computation
        while not self.shared_state.work_done.is_set():
            while self.private_deque:
                # print('Worker ' + str(self.worker_id) + ' normal computation')
                tuple_state = self.private_deque.pop()
                self.state_spawn_depth_first(tuple_state)

                if self.shared_state.work_steal_ready.is_set() and self.shared_state.workers_stole_status[self.worker_id] == 0:
                    self.shared_state.workers_stole_status[self.worker_id] = 1
                    with self.shared_state.num_valid_busy_workers.get_lock():
                        self.shared_state.num_valid_busy_workers.value -= 1
                        print('Worker '+str(self.worker_id)+ ' Stole here I')

                    self.steal_from_this_worker()

            # private deques are empty and assign
            with self.shared_state.num_current_idle_workers.get_lock():
                self.shared_state.num_current_idle_workers.value += 1

            print('Worker ' + str(self.worker_id) + ' workers_stole_status: ', self.shared_state.workers_stole_status[self.worker_id])
            # print('Worker ' + str(self.worker_id) + ' after Stole here I, num_valid_busy_workers: ',self.shared_state.num_valid_busy_workers.value)
            if self.shared_state.workers_stole_status[self.worker_id] == 0:
                # self.shared_state.workers_stole_status[self.worker_id] = 1
                with self.shared_state.num_valid_busy_workers.get_lock():
                    self.shared_state.num_valid_busy_workers.value -= 1
                    print('Worker '+str(self.worker_id)+' Stole here II')

            print('Worker '+str(self.worker_id)+' after Stole here II, num_valid_busy_workers: ', self.shared_state.num_valid_busy_workers.value)
            # assert self.shared_state.num_valid_busy_workers.value < self.shared_state.num_workers
            if self.shared_state.work_steal_ready.is_set() and self.shared_state.num_valid_busy_workers.value == 0:
                self.shared_state.work_steal_ready.clear()
                self.shared_state.work_assign_ready.set()

            self.shared_state.get_idle_worker(self.worker_id)

            if self.shared_state.work_done.is_set():  # work completed
                print('Work is done!')
                break

            with self.shared_state.num_workers_need_assigned.get_lock():
                self.shared_state.num_workers_need_assigned.value += 1

            # print('Worker '+str(self.worker_id)+' num_workers_need_assigned: ', self.shared_state.num_workers_need_assigned.value)
            # print('Worker '+str(self.worker_id)+' num_current_idle_workers: ', self.shared_state.num_current_idle_workers.value)
            if self.shared_state.num_workers_need_assigned.value == self.shared_state.num_current_idle_workers.value:  # steal work from other workers
                print('Start stealing in Worker '+ str(self.worker_id))

                # assert self.steal_assign_ready
                self.shared_state.work_assign_done.clear()
                self.shared_state.steal_assign_ready.clear()
                self.shared_state.compute_steal_rate()
                with self.shared_state.num_empty_assign.get_lock():
                    self.shared_state.num_empty_assign.value = 0
                self.shared_state.work_steal_ready.set()

            print('Worker '+str(self.worker_id)+ ' is waiting for work_assign_ready')
            self.shared_state.work_assign_ready.wait()
            print('Worker ' + str(self.worker_id) + '  passed work_assign_ready')
            self.shared_state.work_steal_ready.clear()
            self.asssign_to_this_worker()

        print('Processor '+str(self.worker_id)+' Output: ', len(self.output_sets))
        print('Processor '+str(self.worker_id)+' is done!')



    def steal_from_this_worker(self):
        steal_rate = self.shared_state.work_steal_rate.value
        assert steal_rate != 0

        num_stealed_works = 0
        if len(self.private_deque)>=10:
            num_stealed_works = np.floor(len(self.private_deque) * steal_rate).astype(np.int64)
            self.shared_state.increase_queue_len(num_stealed_works)
            for n in range(num_stealed_works):
                stealed_work = self.private_deque.pop()
                self.shared_state.shared_queue.put(stealed_work)
                with self.shared_state.stole_works.get_lock():
                    self.shared_state.stole_works.value += 1

        # print('Stole Worker '+str(self.worker_id) + ' works: ', num_stealed_works)

        # self.shared_state.got_from_one_worker(self.worker_id)
        assert self.shared_state.num_valid_busy_workers.value >= 0
        if self.shared_state.num_valid_busy_workers.value == 0:
            self.shared_state.work_steal_ready.clear()
            for indx in range(len(self.shared_state.workers_stole_status)):
                self.shared_state.workers_stole_status[indx] = 0
            self.shared_state.work_assign_ready.set()


    def asssign_to_this_worker(self):
        num_work_assigned = np.ceil(self.shared_state.shared_queue_len.value / self.shared_state.num_workers_need_assigned.value).astype(np.int64)

        if num_work_assigned == 0:
            num = -1
            with self.shared_state.num_empty_assign.get_lock():
                self.shared_state.num_empty_assign.value += 1
        else:
            for num in range(num_work_assigned): # actual number of works should be smaller than num_work_assigned
                one_work = self.shared_state.shared_queue.get()
                self.private_deque.append(one_work)

                with self.shared_state.assigned_works.get_lock():
                    self.shared_state.assigned_works.value += 1
                with self.shared_state.shared_queue_len.get_lock():
                    self.shared_state.shared_queue_len.value -= 1
                    if self.shared_state.shared_queue_len.value == 0:
                        break

        # print('Assigned Worker ' + str(self.worker_id) + ' works: ', num+1)

        # self.shared_state.assigned_one_worker(self.worker_id)
        with self.shared_state.num_assigned_workers.get_lock():
            self.shared_state.num_assigned_workers.value += 1

        # # self.workers_busy_status[work_id] = 1
        # with self.num_valid_busy_workers.get_lock():
        #     self.num_valid_busy_workers.value += 1

        with self.shared_state.num_current_idle_workers.get_lock():
            self.shared_state.num_current_idle_workers.value -= 1

        print('Worker '+str(self.worker_id)+' num_workers_need_assigned: ', self.shared_state.num_workers_need_assigned.value)
        print('Worker '+str(self.worker_id)+' num_assigned_workers: ', self.shared_state.num_assigned_workers.value)
        assert self.shared_state.num_assigned_workers.value <= self.shared_state.num_workers_need_assigned.value
        if self.shared_state.num_assigned_workers.value == self.shared_state.num_workers_need_assigned.value:  # complete assigning
            assert self.shared_state.stole_works.value == self.shared_state.assigned_works.value

            if self.shared_state.num_empty_assign.value == self.shared_state.num_assigned_workers.value:
                self.shared_state.work_done.set()
                # print()
                print('work is done >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            with self.shared_state.num_valid_busy_workers.get_lock():
                self.shared_state.num_valid_busy_workers.value = self.shared_state.num_workers - self.shared_state.num_current_idle_workers.value

            print('Assignment is done, num_valid_busy_workers: ', self.shared_state.num_valid_busy_workers.value)

            # print('num_workers: ', self.num_workers)
            # print('num_current_idle_workers: ', self.num_current_idle_workers.value)
            # print('num_valid_busy_workers: ', self.num_valid_busy_workers.value)
            # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            self.shared_state.work_assign_ready.clear()
            self.shared_state.reset_after_assgin()
            self.shared_state.steal_assign_ready.set()

            xxx = [xx for xx in self.shared_state.workers_stole_status]
            print('Assignment is done, workers_stole_status', xxx)

            self.shared_state.work_assign_done.set()

        else:
            print('Worker ' + str(self.worker_id) + ' is waiting for work_assign_done')
            self.shared_state.work_assign_done.wait()
            print('Worker ' + str(self.worker_id) + ' passed work_assign_done')




    def state_spawn_depth_first(self, tuple_state):
        next_tuple_states = self.dnn.compute_state(tuple_state)
        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            self.output_sets.append(next_tuple_states)
            return

        if len(next_tuple_states) == 2:
            self.private_deque.append(next_tuple_states[1])

        self.state_spawn_depth_first(next_tuple_states[0])



    def state_spawn_breath_first(self, tuple_state):
        next_tuple_states = self.dnn.compute_state(tuple_state)

        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            self.output_sets.append(next_tuple_states)
            return

        for one_state in next_tuple_states:
            self.private_deque.append(one_state)







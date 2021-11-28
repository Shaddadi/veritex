
import random
import time
import numpy as np
import copy as cp
from collections import deque
import itertools

idle_workers = 1
class Worker:
    def __init__(self, dnn, output_len=np.infty):
        self.dnn = dnn
        self.private_deque = deque()
        self.output_sets = []
        self.worker_id = None
        self.shared_state = None
        self.output_len = output_len
        self.inital_num = 500
        self.inital_layer = 2


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

        t0 = time.time()
        # initial computation, breath-first computation
        while self.private_deque:

            tuple_state = self.private_deque.popleft()
            self.state_spawn_breath_first(tuple_state)

            if len(self.private_deque) >= self.inital_num or tuple_state[1]==self.inital_layer:
                # self.private_deque = deque(itertools.islice(self.private_deque, 295, 296))
                # self.private_deque.appendleft(self.private_deque[295])
                self.shared_state.steal_assign_ready.set()
                # #print('This is Worker '+str(self.worker_id))
                # with self.shared_state.num_valid_busy_workers.get_lock():
                #     self.shared_state.num_valid_busy_workers.value -= 1

                self.shared_state.workers_valid_status[self.worker_id] = 0
                # random.shuffle(self.private_deque)
                # self.shared_state.compute_steal_rate()
                self.shared_state.work_steal_ready.wait()
                self.steal_from_this_worker()
                # self.shared_state.work_assign_ready.wait()
                with self.shared_state.initial_comput.get_lock(): # disable it forever
                    self.shared_state.initial_comput.value = 0
                break

        # print('Worker '+str(self.worker_id)+' works: ', len(self.private_deque))

        # normal computation, depth-first computation
        while not self.shared_state.work_done.is_set():
            while self.private_deque:

                tuple_state = self.private_deque.popleft()
                self.state_spawn_depth_first(tuple_state)
                # if time.time() -t0 >= 60:
                #     self.shared_state.work_done.set()
                if self.shared_state.work_done.is_set():
                    self.shared_state.work_assign_ready.set()
                    break

                if self.shared_state.work_steal_ready.is_set() and (not self.shared_state.steal_disabled.is_set()):
                    if self.shared_state.workers_valid_status[self.worker_id] == 1:
                        # #print('Worker ' + str(self.worker_id) + ' after workers_valid_status[self.worker_id] == 1')
                        # #print('Worker ' + str(self.worker_id) + ' steal_ready flag: ',
                        #       self.shared_state.work_steal_ready.is_set())
                        # #print('Worker ' + str(self.worker_id) + ' steal_assign_ready flag: ',
                        #       self.shared_state.steal_assign_ready.is_set())
                        # #print('Worker ' + str(self.worker_id) + ' will be stolen')

                        self.steal_from_this_worker()

                    # #print('Worker ' + str(self.worker_id) + ' skiped workers_valid_status[self.worker_id] == 1')

            # private deques are empty and assign
            #print('Worker ' + str(self.worker_id) + ' becomes idle')
            self.shared_state.workers_idle_status[self.worker_id] = 1
            #>>>>>>>>>>>>>>>>>>>>>>>>>>lock>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
            self.shared_state.lock.acquire()
            # #print('Worker ' + str(self.worker_id) +' lock.acquire() II')
            self.shared_state.workers_valid_status[self.worker_id] = 0

            if self.shared_state.work_steal_ready.is_set() and sum(self.shared_state.workers_valid_status) == 0:
                self.shared_state.work_steal_ready.clear()
                with self.shared_state.works_to_assign_per_worker.get_lock():
                    self.shared_state.works_to_assign_per_worker.value = \
                        np.ceil(self.shared_state.shared_queue_len.value / self.shared_state.num_workers_need_assigned.value).astype(np.int64)
                self.shared_state.work_assign_ready.set()

            self.shared_state.lock.release()
            #<<<<<<<<<<<<<<<<<<<<<<<<<<lock<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
            if sum(self.shared_state.workers_idle_status) == self.shared_state.num_workers:
                self.shared_state.steal_assign_ready.set()
                self.shared_state.work_done.set()

            # print('Worker ' + str(self.worker_id) + ' is waiting for steal_assign_ready')
            self.shared_state.steal_assign_ready.wait()
            # print('Worker ' + str(self.worker_id) + ' passed steal_assign_ready')

            with self.shared_state.num_workers_need_assigned.get_lock():
                self.shared_state.num_workers_need_assigned.value += 1

                # print('self.shared_state.num_workers_need_assigned.value', self.shared_state.num_workers_need_assigned.value)
                # print('self.shared_state.workers_idle_status', sum(self.shared_state.workers_idle_status))
                if self.shared_state.num_workers_need_assigned.value == sum(self.shared_state.workers_idle_status):  # steal work from other workers
                    self.shared_state.steal_assign_ready.clear()
                    #print('shared_state.steal_assign_ready.clear()')
                    self.shared_state.compute_steal_rate()

                    with self.shared_state.num_empty_assign.get_lock():
                        self.shared_state.num_empty_assign.value = 0

                    self.shared_state.work_steal_ready.set()
                    if sum(self.shared_state.workers_idle_status) == self.shared_state.num_workers:
                        self.shared_state.work_done.set()

                    self.shared_state.all_idle_ready.set()

            # print('Worker ' + str(self.worker_id) + ' is waiting for all_idle_ready')
            self.shared_state.all_idle_ready.wait()
            # print('Worker ' + str(self.worker_id) + ' passed all_idle_ready')
            if self.shared_state.work_done.is_set() or self.shared_state.steal_disabled.is_set():
                self.shared_state.steal_assign_ready.set()
                self.shared_state.work_assign_ready.set()
                break
            # print('Worker ' + str(self.worker_id) + ' is waiting for work_assign_ready')
            self.shared_state.work_assign_ready.wait()
            # print('Worker ' + str(self.worker_id) + ' passed work_assign_ready')
            self.shared_state.work_steal_ready.clear()
            # if self.shared_state.work_done.is_set() or self.shared_state.steal_disabled.is_set():
            #     self.shared_state.steal_assign_ready.set()
            #     break
            if self.shared_state.work_done.is_set() or self.shared_state.steal_disabled.is_set():
                break
            self.asssign_to_this_worker()

        # print('Processor '+str(self.worker_id)+' Output: ', len(self.output_sets))
        # print('Processor '+str(self.worker_id)+' is done!')



    def steal_from_this_worker(self):
        steal_rate = self.shared_state.work_steal_rate.value

        try:
            assert steal_rate != 0
        except:
            print('Worker ' + str(self.worker_id) + ' idle workers: ', [xx for xx in self.shared_state.workers_idle_status])
            print('Worker ' + str(self.worker_id) + ' steal_ready flag: ', self.shared_state.work_steal_ready.is_set())
            print('Worker ' + str(self.worker_id) + ' steal_assign_ready flag: ', self.shared_state.steal_assign_ready.is_set())
            print('Worker ' + str(self.worker_id) + ' steal_rate == 0 *****************************************************')

        assert steal_rate != 0

        num_stealed_works = 0
        if len(self.private_deque)>=3:
            with self.shared_state.shared_queue_len.get_lock():
                num_stealed_works = np.floor(len(self.private_deque) * steal_rate).astype(np.int64)
                if num_stealed_works == 0:
                    num_stealed_works = 1
                # self.shared_state.increase_queue_len(num_stealed_works)
                for n in range(num_stealed_works):
                    stealed_work = self.private_deque.popleft() # states ahead
                    self.shared_state.shared_queue.put(stealed_work)
                    self.shared_state.shared_queue_len.value += 1
                    with self.shared_state.stolen_works.get_lock():
                        self.shared_state.stolen_works.value += 1

                # self.shared_state.got_from_one_worker(self.worker_id)
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
                self.shared_state.lock.acquire()
                self.shared_state.workers_valid_status[self.worker_id] = 0
                if sum(self.shared_state.workers_valid_status) == 0:
                    self.shared_state.work_steal_ready.clear()
                    # #print('Worker ' + str(self.worker_id) + 'waiting for shared_queue_len usage I')
                    if self.shared_state.stolen_works.value!=self.shared_state.shared_queue_len.value:
                        print('self.shared_state.stolen_works.value', self.shared_state.stolen_works.value)
                        print('self.shared_state.shared_queue_len.value', self.shared_state.shared_queue_len.value)
                    assert self.shared_state.stolen_works.value==self.shared_state.shared_queue_len.value
                    with self.shared_state.works_to_assign_per_worker.get_lock():
                        self.shared_state.works_to_assign_per_worker.value = np.ceil(
                            self.shared_state.shared_queue_len.value / self.shared_state.num_workers_need_assigned.value).astype(
                            np.int64)
                    self.shared_state.work_assign_ready.set()

                # #print('Worker ' + str(self.worker_id) + ' lock.release() I')
                self.shared_state.lock.release()
                #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


    def asssign_to_this_worker(self):
        # # #print('Worker ' + str(self.worker_id) + ' shared_queue_len: ', self.shared_state.shared_queue_len.value)
        # # #print('Worker ' + str(self.worker_id) + ' num_work_assigned: ', num_work_assigned)
        if self.shared_state.works_to_assign_per_worker.value == 0 or self.shared_state.shared_queue_len == 0:
            with self.shared_state.num_empty_assign.get_lock():
                self.shared_state.num_empty_assign.value += 1
                # print('Worker ' + str(self.worker_id) + ' empty assigned')
        else:
            with self.shared_state.assigned_works.get_lock():
                for num in range(self.shared_state.works_to_assign_per_worker.value): # actual number of works should be smaller than num_work_assigned
                    if self.shared_state.shared_queue_len.value == 0:
                        # # #print('shared_queue_len becomes 0')
                        break
                    one_work = self.shared_state.shared_queue.get()
                    self.private_deque.append(one_work)

                    self.shared_state.assigned_works.value += 1
                    with self.shared_state.shared_queue_len.get_lock():
                        self.shared_state.shared_queue_len.value -= 1
                    # print('self.shared_state.shared_queue_len.value ', self.shared_state.shared_queue_len.value )

            # # #print('Worker ' + str(self.worker_id) + ' num_work_to_assign: ', self.shared_state.works_to_assign_per_worker.value)
            # # #print('Worker ' + str(self.worker_id) + ' num_work_assigned: ', num + 1)

                # # #print('Assigned Worker ' + str(self.worker_id) + ' works: ', num+1)

                # self.shared_state.assigned_one_worker(self.worker_id)

                with self.shared_state.num_assigned_workers.get_lock():
                    self.shared_state.num_assigned_workers.value += 1
                    self.shared_state.workers_idle_status[self.worker_id] = 0

                    assert self.shared_state.num_assigned_workers.value <= self.shared_state.num_workers_need_assigned.value
                    if self.shared_state.num_assigned_workers.value == self.shared_state.num_workers_need_assigned.value:  # complete assigning
                        try:
                            assert self.shared_state.stolen_works.value == self.shared_state.assigned_works.value
                        except:
                            print('stolen_works: ', self.shared_state.stolen_works.value )
                            print('assigned_works', self.shared_state.assigned_works.value)
                            print('shared_queue_len', self.shared_state.shared_queue_len.value)
                            print('num_workers_need_assigned', self.shared_state.num_workers_need_assigned.value)
                            print('num_assigned_workers',  self.shared_state.num_assigned_workers.value)
                            print('works_to_assign_per_worker', self.shared_state.works_to_assign_per_worker.value)

                        assert self.shared_state.stolen_works.value == self.shared_state.assigned_works.value

                        if self.shared_state.num_empty_assign.value == self.shared_state.num_assigned_workers.value:
                            if self.shared_state.num_empty_assign.value/self.shared_state.num_workers >= 0.8:
                                self.shared_state.steal_disabled.set()

                        with self.shared_state.workers_idle_status.get_lock():
                            for n, ele in enumerate(self.shared_state.workers_idle_status):
                                if ele == 0: # not idle
                                    self.shared_state.workers_valid_status[n] = 1
                                else:
                                    self.shared_state.workers_valid_status[n] = 0

                        # print('Assignment is done, num_valid_busy_workers: ', sum(self.shared_state.workers_valid_status))
                        self.shared_state.work_assign_ready.clear()
                        self.shared_state.reset_after_assgin(self.worker_id)
                        self.shared_state.steal_assign_ready.set()
                        self.shared_state.all_idle_ready.clear()
                        # self.shared_state.work_assign_done.set()


        # print('Worker ' + str(self.worker_id) + ' is waiting for work_assign_done')
        # self.shared_state.work_assign_done.wait()
        # print('Worker ' + str(self.worker_id) + ' passed work_assign_done')

    def collect_results(self, vfl):
        if self.dnn.config_repair:
            unsafe_input_sets = self.dnn.backtrack(vfl)
            if unsafe_input_sets:
                with self.shared_state.outputs_len.get_lock():
                    for aset in unsafe_input_sets:
                        unsafe_input = aset.vertices[[0]]
                        unsafe_output = np.dot(aset.vertices[0], vfl.M.T) + vfl.b.T
                        self.shared_state.outputs_len.value += 1
                        self.shared_state.outputs.put([unsafe_input, unsafe_output])
                        if self.shared_state.outputs_len.value >= self.output_len:
                            self.shared_state.work_done.set()

        elif self.dnn.config_verify:
            unsafe = self.dnn.verify(vfl)
            if unsafe:
                self.shared_state.outputs.put(unsafe)
                self.shared_state.work_done.set()

        elif self.dnn.config_unsafe_input and (not self.dnn.config_exact_output):
            unsafe_inputs = self.dnn.backtrack(vfl)
            with self.shared_state.outputs_len.get_lock():
                self.shared_state.outputs_len.value += 1
                if unsafe_inputs:
                    self.shared_state.outputs.put(unsafe_inputs)
                if self.shared_state.outputs_len.value >= self.output_len:
                    self.shared_state.work_done.set()

        elif (not self.dnn.config_unsafe_input) and self.dnn.config_exact_output:
            with self.shared_state.outputs_len.get_lock():
                self.shared_state.outputs_len.value += 1
                self.shared_state.outputs.put(vfl)

        elif self.dnn.config_unsafe_input and self.dnn.config_exact_output:
            unsafe_inputs = self.dnn.backtrack(vfl)
            with self.shared_state.outputs_len.get_lock():
                self.shared_state.outputs.put([unsafe_inputs, vfl])
                self.shared_state.outputs_len.value += 1
        else:
            raise ValueError('Reachability configuration error!')



    def state_spawn_depth_first(self, tuple_state):
        next_tuple_states = self.dnn.compute_state(tuple_state)
        if len(next_tuple_states) == 0:
            return
        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            self.collect_results(next_tuple_states[0][0])
            return

        # if self.dnn.config_relu_linear or self.dnn.config_repair:
        # # if self.dnn.config_relu_linear:
        #     assert (not self.dnn.config_verify)
        #     temp_list = []
        #     for one_state in next_tuple_states:
        #         over_app_set = self.dnn.reach_over_app(one_state)
        #         safe = self.dnn.verify_vzono(over_app_set)
        #         if not safe:
        #             temp_list.append(one_state)
        #     if len(temp_list) != 0:
        #         next_tuple_states = temp_list
        #     else:
        #         return

        if len(next_tuple_states) == 2:
            self.private_deque.append(next_tuple_states[1])

        self.state_spawn_depth_first(next_tuple_states[0])



    def state_spawn_breath_first(self, tuple_state):
        next_tuple_states = self.dnn.compute_state(tuple_state)
        if len(next_tuple_states) == 0:
            return

        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            self.collect_results(next_tuple_states[0][0])
            return

        for one_state in next_tuple_states:
            self.private_deque.append(one_state)



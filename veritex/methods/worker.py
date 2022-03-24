
import time
import numpy as np
from collections import deque


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
        self.tuple_number_track = 0


    def inital_assgin(self):
        # get inputs from veritex.methods.shared state
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
                self.shared_state.steal_assign_ready.set()
                self.shared_state.workers_valid_status[self.worker_id] = 0
                self.shared_state.work_steal_ready.wait()
                self.steal_from_this_worker()
                # self.shared_state.work_assign_ready.wait()
                with self.shared_state.initial_comput.get_lock(): # disable it forever
                    self.shared_state.initial_comput.value = 0
                break

        # normal computation, depth-first computation
        # while (not self.shared_state.work_done.is_set()) and (not self.shared_state.work_interrupted.is_set()):
        while not (self.shared_state.work_interrupted.is_set()):
            while self.private_deque:
                tuple_state = self.private_deque.popleft()
                self.state_spawn_depth_first(tuple_state)
                if self.shared_state.workers_valid_status[self.worker_id] == 1 and self.shared_state.work_steal_ready.is_set():
                    self.steal_from_this_worker()
                if self.shared_state.work_interrupted.is_set():
                    break

            if self.shared_state.work_done.is_set():
                self.shared_state.workers_valid_status[self.worker_id] = 0
                self.shared_state.workers_idle_status[self.worker_id] = 1
                break

            with self.shared_state.workers_valid_status.get_lock():
                self.shared_state.workers_valid_status[self.worker_id] = 0
                if self.shared_state.work_steal_ready.is_set() and sum(self.shared_state.workers_valid_status) == 0:
                    self.shared_state.work_steal_ready.clear()

                    with self.shared_state.works_to_assign_per_worker.get_lock():
                        self.shared_state.works_to_assign_per_worker.value = \
                            np.ceil(self.shared_state.shared_queue_len.value / sum(self.shared_state.workers_to_assign)).astype(np.int64)
                    # print('self.shared_state.num_workers_to_assign.value: ',
                    #       sum(self.shared_state.workers_to_assign))
                    # print('self.shared_state.works_to_assign_per_worker.value: ',
                    #       self.shared_state.works_to_assign_per_worker.value)
                    # print('self.shared_state.shared_queue_len.value: ', self.shared_state.shared_queue_len.value)
                    self.shared_state.work_assign_ready.set()

            # print('Worker '+ str(self.worker_id) + ' before self.shared_state.workers_idle_status.get_lock()')
            with self.shared_state.workers_idle_status.get_lock():
                # print('Worker ' + str(self.worker_id) + ' after self.shared_state.workers_idle_status.get_lock()')
                self.shared_state.workers_idle_status[self.worker_id] = 1
                if sum(self.shared_state.workers_idle_status) == self.shared_state.num_workers:
                    if self.shared_state.shared_queue_len.value == 0:
                        self.shared_state.work_done.set()
                        # print('self.shared_state.work_done.set()')
                        self.shared_state.steal_assign_ready.set()
                        self.shared_state.work_steal_ready.clear()
                    self.shared_state.work_assign_ready.set()

            self.shared_state.steal_assign_ready.wait()
            if self.shared_state.work_done.is_set() or self.shared_state.work_interrupted.is_set():
                self.shared_state.work_steal_ready.clear()
                self.shared_state.steal_assign_ready.set()
                self.shared_state.work_assign_ready.set()
                break

            with self.shared_state.workers_to_assign.get_lock():
                self.shared_state.workers_to_assign[self.worker_id] = 1
                with self.shared_state.workers_idle_status.get_lock():
                    # print('self.shared_state.work_assign_ready.is_set(): ', self.shared_state.work_assign_ready.is_set())
                    # print('Worker '+str(self.worker_id)+' sum(self.shared_state.workers_to_assign) : ', sum(self.shared_state.workers_to_assign) )
                    # print('Worker '+str(self.worker_id)+' num_target_workers: ', sum(self.shared_state.workers_idle_status))
                    # if self.shared_state.num_workers_to_assign.value == num_target_workers:  # steal work from other workers
                    if sum(self.shared_state.workers_to_assign) == sum(self.shared_state.workers_idle_status):
                        # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                        self.shared_state.steal_assign_ready.clear()
                        self.shared_state.compute_steal_rate()
                        with self.shared_state.num_empty_assign.get_lock():
                            self.shared_state.num_empty_assign.value = 0
                        # print('self.shared_state.outputs_len.value: ', self.shared_state.outputs_len.value)
                        if (sum(self.shared_state.workers_to_assign) == self.shared_state.num_workers): # or self.shared_state.work_interrupted.is_set():
                        # if self.shared_state.work_done.is_set() or self.shared_state.work_interrupted.is_set():
                            self.shared_state.work_done.set()
                            self.shared_state.steal_assign_ready.set()
                            # print('self.shared_state.work_done.set()222')
                            assert self.shared_state.shared_queue_len.value==0
                            self.shared_state.work_steal_ready.clear()
                            self.shared_state.work_assign_ready.set()
                            break

                        self.shared_state.work_steal_ready.set()

            if self.shared_state.work_done.is_set() or self.shared_state.work_interrupted.is_set():
                self.shared_state.work_steal_ready.clear()
                self.shared_state.steal_assign_ready.set()
                self.shared_state.work_assign_ready.set()
                # print('Worker ' + str(self.worker_id) + 'break222')
                break

            self.shared_state.work_assign_ready.wait()
            self.shared_state.work_steal_ready.clear()
            self.asssign_to_this_worker()

        if self.shared_state.work_done.is_set() and (not self.shared_state.work_interrupted.is_set()): # handle leftover in shared_queue
            with self.shared_state.num_workers_done.get_lock():
                self.shared_state.num_workers_done.value += 1
                if self.shared_state.num_workers_done.value == self.shared_state.num_workers:
                    self.shared_state.all_workers_done.set()
            self.shared_state.all_workers_done.wait()
            while True:
                with self.shared_state.shared_queue_len.get_lock():
                    if self.shared_state.shared_queue_len.value == 0:
                        break
                    one_state = self.shared_state.shared_queue.get()
                    self.private_deque.append(one_state)
                    self.shared_state.shared_queue_len.value -= 1
                while self.private_deque:
                    tuple_state = self.private_deque.popleft()
                    self.state_spawn_depth_first(tuple_state)

            assert self.shared_state.shared_queue_len.value == 0
            if len(self.private_deque) != 0:
                print('error')







    def steal_from_this_worker(self):
        steal_rate = self.shared_state.work_steal_rate.value
        assert steal_rate != 0

        num_stolen_works = 0
        if len(self.private_deque)>=5:
            with self.shared_state.shared_queue_len.get_lock():
                num_stolen_works = np.floor(len(self.private_deque) * steal_rate).astype(np.int64)
                if num_stolen_works == 0:
                    num_stolen_works = 1
                # self.shared_state.increase_queue_len(num_stolen_works)
                for n in range(num_stolen_works):
                    stolen_work = self.private_deque.popleft() # states ahead
                    self.shared_state.shared_queue.put(stolen_work)
                    self.shared_state.shared_queue_len.value += 1
                    with self.shared_state.stolen_works.get_lock():
                        self.shared_state.stolen_works.value += 1

        with self.shared_state.workers_valid_status.get_lock():
            self.shared_state.workers_valid_status[self.worker_id] = 0
            if sum(self.shared_state.workers_valid_status) == 0:
                self.shared_state.work_steal_ready.clear()
                # print('self.shared_state.work_steal_ready.clear()')
                assert self.shared_state.stolen_works.value==self.shared_state.shared_queue_len.value
                with self.shared_state.works_to_assign_per_worker.get_lock():
                    # self.shared_state.works_to_assign_per_worker.value = np.ceil(
                    #     self.shared_state.shared_queue_len.value / self.shared_state.num_workers_to_assign.value).astype(
                    #     np.int64)
                    self.shared_state.works_to_assign_per_worker.value = np.ceil(
                        self.shared_state.shared_queue_len.value / sum(self.shared_state.workers_to_assign)).astype(
                        np.int64)
                    # print('self.shared_state.num_workers_to_assign.value: ',
                    #       sum(self.shared_state.workers_to_assign))
                    # print('self.shared_state.works_to_assign_per_worker.value: ',
                    #       self.shared_state.works_to_assign_per_worker.value)
                    # print('self.shared_state.shared_queue_len.value: ', self.shared_state.shared_queue_len.value)
                self.shared_state.work_assign_ready.set()


    def asssign_to_this_worker(self):
        # print('Worker ' + str(self.worker_id) + ' before self.shared_state.assigned_works.get_lock()')
        # assert self.shared_state.works_to_assign_per_worker.value*sum(self.shared_state.workers_to_assign)>=self.shared_state.stolen_works.value
        with self.shared_state.assigned_works.get_lock():
            # print('Worker ' + str(self.worker_id) + ' after self.shared_state.assigned_works.get_lock()')
            num = 0
            while num < self.shared_state.works_to_assign_per_worker.value:
            # for num in range(self.shared_state.works_to_assign_per_worker.value): # actual number of works should be smaller than num_work_assigned
                with self.shared_state.shared_queue_len.get_lock():
                    if self.shared_state.shared_queue_len.value == 0:
                        break
                    one_work = self.shared_state.shared_queue.get()
                    self.private_deque.append(one_work)
                    self.shared_state.assigned_works.value += 1
                    self.shared_state.shared_queue_len.value -= 1
                num += 1
            # while num < self.shared_state.works_to_assign_per_worker.value:
            #     try:
            #         one_work = self.shared_state.shared_queue.get_nowait()
            #         self.private_deque.append(one_work)
            #         self.shared_state.assigned_works.value += 1
            #     except:
            #         with self.shared_state.shared_queue_len.get_lock():
            #             self.shared_state.shared_queue_len.value = 0
            #         break
            #     num += 1

            if num == 0: #empty assign
                with self.shared_state.num_empty_assign.get_lock():
                    self.shared_state.num_empty_assign.value += 1
            else:
                with self.shared_state.workers_idle_status.get_lock():
                    self.shared_state.workers_idle_status[self.worker_id] = 0

            # print('Worker ID '+ str(self.worker_id) + ' assigned works: ', num)
            # print('Worker ID '+ str(self.worker_id) + ' works_to_assign_per_worker', self.shared_state.works_to_assign_per_worker.value)
            with self.shared_state.workers_assigned.get_lock():
                self.shared_state.workers_assigned[self.worker_id] = 1
                if not (sum(self.shared_state.workers_assigned) <= sum(self.shared_state.workers_to_assign)):
                    # print('sum(self.shared_state.workers_assigned): ',sum(self.shared_state.workers_assigned))
                    # print('sum(self.shared_state.workers_to_assign): ', sum(self.shared_state.workers_to_assign))
                    assert sum(self.shared_state.workers_assigned) <= sum(self.shared_state.workers_to_assign)
                if sum(self.shared_state.workers_assigned) == sum(self.shared_state.workers_to_assign):
                    for _ in range(self.shared_state.stolen_works.value-self.shared_state.assigned_works.value):  # actual number of works should be smaller than num_work_assigned
                        with self.shared_state.shared_queue_len.get_lock():
                            one_work = self.shared_state.shared_queue.get()
                            self.private_deque.append(one_work)
                            self.shared_state.shared_queue_len.value -= 1
                    assert self.shared_state.shared_queue_len.value == 0
                    # if self.shared_state.stolen_works.value != self.shared_state.assigned_works.value and \
                    #         not (self.shared_state.work_done.is_set() or self.shared_state.work_interrupted.is_set()):
                    #     print('self.shared_state.num_assigned_workers.value: ', sum(self.shared_state.workers_assigned))
                    #     print('self.shared_state.work_steal_rate.value: ', self.shared_state.work_steal_rate.value)
                    #     print('self.shared_state.stolen_works.value: ', self.shared_state.stolen_works.value)
                    #     print('self.shared_state.assigned_works.value: ', self.shared_state.assigned_works.value)
                    #     print('self.shared_state.works_to_assign_per_worker.value: ', self.shared_state.works_to_assign_per_worker.value)
                    #     print('self.shared_state.work_interrupted.is_set(): ', self.shared_state.work_interrupted.is_set())
                    #     assert self.shared_state.stolen_works.value == self.shared_state.assigned_works.value
                    # # if self.shared_state.num_empty_assign.value == self.shared_state.num_assigned_workers.value:
                    # #     if self.shared_state.num_empty_assign.value/self.shared_state.num_workers >= 0.8:
                    # #         self.shared_state.steal_disabled.set()

                    with self.shared_state.workers_idle_status.get_lock():
                        for n, ele in enumerate(self.shared_state.workers_idle_status):
                            if ele == 0: # not idle
                                self.shared_state.workers_valid_status[n] = 1
                            else:
                                self.shared_state.workers_valid_status[n] = 0

                    # print('Assignment is done, num_valid_busy_workers: ', sum(self.shared_state.workers_valid_status))
                    self.shared_state.work_assign_ready.clear()
                    self.shared_state.reset_after_assgin()
                    self.shared_state.all_idle_ready.clear()
                    # self.shared_state.work_assign_done.set()
                    # self.shared_state.lock.acquire()
                    # print('Worker ' + str(self.worker_id) + '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    # self.shared_state.lock.release()
                    self.shared_state.steal_assign_ready.set()
                    # self.shared_state.work_assign_done.set()



    def collect_results(self, vfl):
        if self.dnn.config_repair:
            unsafe_input_sets = self.dnn.backtrack(vfl)
            with self.shared_state.outputs_len.get_lock():
                if not self.shared_state.work_interrupted.is_set():
                    for aset in unsafe_input_sets:
                        unsafe_input = aset.vertices[[0]]
                        unsafe_output = np.dot(aset.vertices[0], vfl.M.T) + vfl.b.T
                        self.shared_state.outputs_len.value += 1
                        self.shared_state.outputs.put([unsafe_input, unsafe_output])
                        if self.shared_state.outputs_len.value >= self.output_len:
                            self.shared_state.work_interrupted.set()
                            # print('self.shared_state.work_interrupted.set()')

        elif self.dnn.config_verify:
            if not self.shared_state.work_interrupted.is_set():
                unsafe = self.dnn.verify(vfl)
                if unsafe:
                    self.shared_state.outputs.put(unsafe)
                    self.shared_state.work_interrupted.set()

        elif self.dnn.config_unsafe_in_dom and (not self.dnn.config_exact_out_dom):
            unsafe_inputs = self.dnn.backtrack(vfl)
            with self.shared_state.outputs_len.get_lock():
                if not self.shared_state.work_interrupted.is_set():
                    for aset in unsafe_inputs:
                        self.shared_state.outputs_len.value += 1
                        self.shared_state.outputs.put(aset)
                        if self.shared_state.outputs_len.value >= self.output_len:
                            self.shared_state.work_interrupted.set()

        elif (not self.dnn.config_unsafe_in_dom) and self.dnn.config_exact_out_dom:
            with self.shared_state.outputs_len.get_lock():
                self.shared_state.outputs_len.value += 1
                self.shared_state.outputs.put(vfl)

        elif self.dnn.config_unsafe_in_dom and self.dnn.config_exact_out_dom:
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



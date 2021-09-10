
import random
import time
import numpy as np
import copy as cp
from collections import deque
import itertools

idle_workers = 1
class Worker:
    def __init__(self, dnn):
        self.dnn = dnn
        self.private_deque = deque()
        self.output_sets = []
        self.worker_id = None
        self.shared_state = None
        self.inital_num = 2000 #480
        self.inital_layer = 1


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
                # self.private_deque = deque(itertools.islice(self.private_deque, 170, 175))
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
                if self.shared_state.work_steal_ready.is_set():
                    # print('Worker ' + str(self.worker_id) + ' before workers_valid_status[self.worker_id] == 1')
                    if self.shared_state.workers_valid_status[self.worker_id] == 1 and self.shared_state.work_steal_ready.is_set():
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

            # print('Worker ' + str(self.worker_id) + ' is waiting for steal_assign_ready')
            self.shared_state.steal_assign_ready.wait()
            # print('Worker ' + str(self.worker_id) + ' passed steal_assign_ready')

            with self.shared_state.num_workers_need_assigned.get_lock():
                self.shared_state.num_workers_need_assigned.value += 1

                # print('Worker '+str(self.worker_id)+' num_workers_need_assigned: ', self.shared_state.num_workers_need_assigned.value)
                # print('Worker '+str(self.worker_id)+' num_current_idle_workers: ', sum(self.shared_state.workers_idle_status))
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

            self.shared_state.all_idle_ready.wait()
            if self.shared_state.work_done.is_set():
                self.shared_state.steal_assign_ready.set()
                break
            # print('Worker ' + str(self.worker_id) + ' is waiting for work_assign_ready')
            self.shared_state.work_assign_ready.wait()
            if self.shared_state.work_assign_done.is_set():
                self.shared_state.work_assign_done.clear()
            # print('Worker ' + str(self.worker_id) + ' passed work_assign_ready')
            self.shared_state.work_steal_ready.clear()
            if self.shared_state.work_done.is_set():
                self.shared_state.steal_assign_ready.set()
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
        # # #print('Worker '+str(self.worker_id) + ' works: ', len(self.private_deque))
        if len(self.private_deque)>=3:
            num_stealed_works = np.floor(len(self.private_deque) * steal_rate).astype(np.int64)
            if num_stealed_works == 0:
                num_stealed_works = 1
            self.shared_state.increase_queue_len(num_stealed_works)
            for n in range(num_stealed_works):
                stealed_work = self.private_deque.popleft() # states ahead
                self.shared_state.shared_queue.put(stealed_work)
                with self.shared_state.stole_works.get_lock():
                    self.shared_state.stole_works.value += 1

        #print('Stole Worker '+str(self.worker_id) + ' works: ', num_stealed_works)

        # self.shared_state.got_from_one_worker(self.worker_id)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
        self.shared_state.lock.acquire()
        self.shared_state.workers_valid_status[self.worker_id] = 0
        if sum(self.shared_state.workers_valid_status) == 0:
            self.shared_state.work_steal_ready.clear()
            # #print('Worker ' + str(self.worker_id) + 'waiting for shared_queue_len usage I')
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
        else:
            with self.shared_state.shared_queue_len.get_lock():
                for num in range(self.shared_state.works_to_assign_per_worker.value): # actual number of works should be smaller than num_work_assigned
                    if self.shared_state.shared_queue_len.value == 0:
                        # # #print('shared_queue_len becomes 0')
                        break
                    one_work = self.shared_state.shared_queue.get()
                    self.private_deque.append(one_work)
                    with self.shared_state.assigned_works.get_lock():
                        self.shared_state.assigned_works.value += 1

                    self.shared_state.shared_queue_len.value -= 1

            # # #print('Worker ' + str(self.worker_id) + ' num_work_to_assign: ', self.shared_state.works_to_assign_per_worker.value)
            # # #print('Worker ' + str(self.worker_id) + ' num_work_assigned: ', num + 1)



        # # #print('Assigned Worker ' + str(self.worker_id) + ' works: ', num+1)

        # self.shared_state.assigned_one_worker(self.worker_id)
        self.shared_state.workers_idle_status[self.worker_id] = 0
        with self.shared_state.num_assigned_workers.get_lock():
            self.shared_state.num_assigned_workers.value += 1

            #print('Worker ' + str(self.worker_id) + ' num_assigned_workers: ',  self.shared_state.num_assigned_workers.value)
            #print('Worker ' + str(self.worker_id) + ' num_workers_need_assigned: ', self.shared_state.num_workers_need_assigned.value)
            assert self.shared_state.num_assigned_workers.value <= self.shared_state.num_workers_need_assigned.value
            if self.shared_state.num_assigned_workers.value == self.shared_state.num_workers_need_assigned.value:  # complete assigning

                try:
                    assert self.shared_state.stole_works.value == self.shared_state.assigned_works.value
                except:
                    print('stole_works: ', self.shared_state.stole_works.value )
                    print('assigned_works', self.shared_state.assigned_works.value)
                    print('shared_queue_len', self.shared_state.shared_queue_len.value)
                    print('num_workers_need_assigned', self.shared_state.num_workers_need_assigned.value)
                    print('works_to_assign_per_worker', self.shared_state.works_to_assign_per_worker.value)

                assert self.shared_state.stole_works.value == self.shared_state.assigned_works.value


                if self.shared_state.num_empty_assign.value == self.shared_state.num_assigned_workers.value:
                    self.shared_state.work_done.set()
                    print('work is done >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

                with self.shared_state.workers_idle_status.get_lock():
                    for n, ele in enumerate(self.shared_state.workers_idle_status):
                        if ele == 0: # not idle
                            self.shared_state.workers_valid_status[n] = 1
                        else:
                            self.shared_state.workers_valid_status[n] = 0

                #print('Assignment is done, num_valid_busy_workers: ', sum(self.shared_state.workers_valid_status))
                self.shared_state.work_assign_ready.clear()
                self.shared_state.reset_after_assgin(self.worker_id)
                self.shared_state.steal_assign_ready.set()
                self.shared_state.all_idle_ready.clear()
                self.shared_state.work_assign_done.set()


        # print('Worker ' + str(self.worker_id) + ' is waiting for work_assign_done')
        self.shared_state.work_assign_done.wait()
        # print('Worker ' + str(self.worker_id) + ' passed work_assign_done')


    def state_spawn_depth_first(self, tuple_state):
        next_tuple_states = self.dnn.compute_state(tuple_state)
        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1

            if self.dnn.config_verify:
                unsafe = self.dnn.verify(next_tuple_states[0][0])
                if unsafe:
                    self.shared_state.outputs.put(unsafe)
                    self.shared_state.work_done.set()
            elif self.dnn.config_unsafe_input:
                unsafety = self.dnn.backtrack(next_tuple_states[0][0])
                with self.shared_state.outputs_len.get_lock():
                    self.shared_state.outputs_len.value += 1
                    self.shared_state.outputs.put(unsafety)
                    if self.shared_state.outputs_len.value >= self.dnn.outputs_len:
                        self.shared_state.work_done.set()
                        return
            elif self.dnn.config_exact_output:
                self.shared_state.outputs.put(next_tuple_states[0][0])

            return

        if self.dnn.config_relu_linear:
            temp_list = []
            for one_state in next_tuple_states:
                over_app_set = self.dnn.reachOverApp(one_state)
                safe = self.dnn.verifyVzono(over_app_set)
                if not safe:
                    temp_list.append(one_state)
            if len(temp_list) != 0:
                next_tuple_states = temp_list
            else:
                return

        if len(next_tuple_states) == 2:
            self.private_deque.append(next_tuple_states[1])

        self.state_spawn_depth_first(next_tuple_states[0])



    def state_spawn_breath_first(self, tuple_state):
        next_tuple_states = self.dnn.compute_state(tuple_state)

        # if next_tuple_states[0][1] == self.inital_layer or len(self.private_deque) >= self.inital_num: # reach to the next layer
        #     for one_state in next_tuple_states:
        #         self.private_deque.append(one_state)
        #     return
        #
        # for one_state in next_tuple_states:
        #     self.private_deque.appendleft(one_state)
        #
        # self.state_spawn_breath_first(self.private_deque.popleft())

        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            if self.dnn.config_verify:
                unsafe = self.dnn.verify(next_tuple_states[0][0])
                if unsafe:
                    self.shared_state.outputs.put(unsafe)
                    self.shared_state.work_done.set()
                return
            elif self.dnn.config_unsafe_input:
                unsafety = self.dnn.backtrack(next_tuple_states[0][0])
                with self.shared_state.outputs_len.get_lock():
                    self.shared_state.outputs_len.value += 1
                    self.shared_state.outputs.put(unsafety)
                    if self.shared_state.outputs_len.value >= self.dnn.outputs_len:
                        self.shared_state.work_done.set()
                        return
            elif self.dnn.config_exact_output:
                self.shared_state.outputs.put(next_tuple_states[0][0])

        for one_state in next_tuple_states:
            self.private_deque.append(one_state)



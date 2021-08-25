
import multiprocessing as mp
import numpy as np
idle_workers = 1

class SharedState: # todo: make it freezable
    def __init__(self, vfl_inputs, num_workers, net_layers):

        self.num_workers = num_workers
        self.net_layers = net_layers
        self.shared_queue = mp.Queue(3*10**4)
        self.shared_queue_len = mp.Value('i', 0)
        self.initialize_shared_queue(vfl_inputs)
        self.initial_comput = mp.Value('i', 1) # for initial computation
        # self.workers_status = mp.Array('i', [0]*num_workers) # 0: busy, 1: idle

        self.stole_works = mp.Value('i', 0)
        self.assigned_works = mp.Value('i', 0)

        self.work_steal_ready = mp.Event()
        self.work_assign_ready = mp.Event()
        self.steal_assign_ready = mp.Event()
        self.steal_assign_ready.set()
        self.work_done = mp.Event()

        self.work_steal_rate = mp.Value('f', 0.0) # initial
        self.num_current_idle_workers = mp.Value('i', 0) # initial
        self.num_workers_need_assigned = mp.Value('i', 0) # initial
        self.num_assigned_workers = mp.Value('i', 0) # should equal to num_workers_need_assigned
        # self.num_stealed_workers = mp.Value('i', 0) #
        self.workers_stole_status = mp.Array('i', [0]*num_workers) # 1 indicates it has been stolen
        # self.workers_busy_status = mp.Array('i',[1]*num_workers) # 1 indicates it is busy
        self.num_valid_busy_workers = mp.Value('i',num_workers)


    def initialize_shared_queue(self, vfl_inputs):
        for vfl in vfl_inputs:
            self.shared_queue.put((vfl, -1, np.array([])))

        self.increase_queue_len(len(vfl_inputs))


    def increase_queue_len(self, num):
        with self.shared_queue_len.get_lock():
            self.shared_queue_len.value += num


    def decrease_queue_len(self):
        with self.shared_queue_len.get_lock():
            self.shared_queue_len.value -= 1


    def got_from_one_worker(self, work_id):
        # with self.num_stealed_workers.get_lock():
        #     self.num_stealed_workers.value += 1

        # self.workers_stole_status[work_id] = 1

        # with self.num_valid_busy_workers.get_lock():
        #     self.num_valid_busy_workers.value -= 1

        # print('num_workers: ', self.num_workers)
        # print('num_stealed_workers: ', self.num_stealed_workers.value)
        # print('num_stolen_workers: ', sum(self.workers_stole_status))
        # assert self.num_stealed_workers.value <= self.num_workers - self.num_current_idle_workers.value
        # if self.num_stealed_workers.value == self.num_workers - self.num_current_idle_workers.value: # complete stealing

        assert self.num_valid_busy_workers.value >= 0
        if self.num_valid_busy_workers.value == 0:
            self.work_assign_ready.set()
            self.work_steal_ready.clear()


    def assigned_one_worker(self, work_id):
        with self.num_assigned_workers.get_lock():
            self.num_assigned_workers.value +=1

        # self.workers_busy_status[work_id] = 1
        with self.num_valid_busy_workers.get_lock():
            self.num_valid_busy_workers.value += 1

        with self.num_current_idle_workers.get_lock():
            self.num_current_idle_workers.value -= 1

        print('Worker '+str(work_id)+' num_workers_need_assigned: ', self.num_workers_need_assigned.value)
        print('Worker '+str(work_id)+' num_assigned_workers: ', self.num_assigned_workers.value)
        assert self.num_assigned_workers.value <= self.num_workers_need_assigned.value
        if self.num_assigned_workers.value == self.num_workers_need_assigned.value: # complete assigning
            assert self.stole_works.value == self.assigned_works.value
            print()
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            with self.num_valid_busy_workers.get_lock():
                self.num_valid_busy_workers.value = self.num_workers - self.num_current_idle_workers.value

            print('num_workers: ', self.num_workers)
            print('num_current_idle_workers: ', self.num_current_idle_workers.value)
            print('num_valid_busy_workers: ', self.num_valid_busy_workers.value)
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            self.work_assign_ready.clear()
            self.reset_after_assgin()
            self.steal_assign_ready.set()


    def get_idle_worker(self, worker_id):
        print('Worker ' + str(worker_id) + ' becomes idle')
        print('Worker '+str(worker_id)+' num_valid_busy_workers: ', self.num_valid_busy_workers.value)
        if self.initial_comput.value==1:
            # self.workers_status[worker_id] = 1
            with self.num_workers_need_assigned.get_lock():
                self.num_workers_need_assigned.value += 1
            assert self.num_current_idle_workers.value <= self.num_workers
            assert self.steal_assign_ready.is_set()
        else:

            print('Worker ' + str(worker_id) + ' is waiting for steal_assign_ready')
            self.steal_assign_ready.wait()
            print('Worker ' + str(worker_id) + ' passed steal_assign_ready')

            with self.num_workers_need_assigned.get_lock():
                self.num_workers_need_assigned.value += 1

            if self.num_workers_need_assigned.value == self.num_workers:  # work completed
                print('Work is done!')
                self.work_done.set()
                return

            if self.num_workers_need_assigned.value >= idle_workers: # steal work from other workers
                print('Start stealing')
                self.steal_assign_ready.clear()

                self.compute_steal_rate()
                self.work_steal_ready.set()



    def compute_steal_rate(self):
        with self.work_steal_rate.get_lock():
            self.work_steal_rate.value = self.num_workers_need_assigned.value / self.num_workers


    def reset_after_assgin(self):
        assert self.initial_comput.value == 0
        assert not self.work_steal_ready.is_set()
        assert not self.work_assign_ready.is_set()
        assert self.shared_queue_len.value == 0

        # should be empty
        while True:
            try:
                _ = self.shared_queue.get_nowait()
                print('Should be None!')
            except:
                break

        # with self.num_current_idle_workers.get_lock():
        #     self.num_current_idle_workers.value = 0

        with self.work_steal_rate.get_lock():
            self.work_steal_rate.value = 0.0

        with self.num_assigned_workers.get_lock():
            self.num_assigned_workers.value = 0

        # with self.num_stealed_workers.get_lock():
        #     self.num_stealed_workers.value = 0

        with self.stole_works.get_lock():
            self.stole_works.value = 0

        with self.assigned_works.get_lock():
            self.assigned_works.value = 0

        with self.num_workers_need_assigned.get_lock():
            self.num_workers_need_assigned.value = 0

        for n in range(self.num_workers):
            self.workers_stole_status[n] = 0
            # self.workers_busy_status[n] = 1








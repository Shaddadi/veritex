
import multiprocessing as mp
import numpy as np
idle_workers = 1

class SharedState: #
    def __init__(self, vfl_inputs, num_workers):

        self.num_workers = num_workers
        self.shared_queue = mp.Queue()
        self.shared_queue_len = mp.Value('i', 0)

        self.outputs = mp.Manager().Queue()
        self.outputs_len = mp.Value('i', 0)

        self.initial_steal_assign = mp.Event()
        self.initial_completed_workers = mp.Value('i', 0)
        self.initialize_shared_queue(vfl_inputs)
        self.initial_comput = mp.Value('i', 1) # for initial computation

        self.stolen_works = mp.Value('i', 0)
        self.assigned_works = mp.Value('i', 0)
        self.num_empty_assign = mp.Value('i', 0)
        self.works_to_assign_per_worker = mp.Value('i', 0)
        self.num_workers_passed_assign_done = mp.Value('i', 0)

        self.work_steal_ready = mp.Event()
        self.work_assign_ready = mp.Event()
        # self.work_assign_done = mp.Event()
        self.all_idle_ready = mp.Event()
        self.steal_assign_ready = mp.Event()
        self.work_done = mp.Event()

        self.lock = mp.Lock()

        self.work_steal_rate = mp.Value('f', 0.0) # initial
        self.num_workers_need_assigned = mp.Value('i', 0) # initial
        self.num_assigned_workers = mp.Value('i', 0) # should equal to num_workers_need_assigned

        self.workers_valid_status = mp.Array('i',[1]*num_workers, lock=False)
        self.workers_idle_status = mp.Array('i', [0]*num_workers)


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


    def compute_steal_rate(self):
        with self.work_steal_rate.get_lock():
            self.work_steal_rate.value = self.num_workers_need_assigned.value / self.num_workers


    def reset_after_assgin(self, worker_id):
        if not self.work_done.is_set():
            assert self.initial_comput.value == 0
            assert not self.work_steal_ready.is_set()
            assert not self.work_assign_ready.is_set()
            assert self.shared_queue_len.value == 0

        assert self.shared_queue.empty()

        with self.work_steal_rate.get_lock():
            self.work_steal_rate.value = 0.0

        with self.num_assigned_workers.get_lock():
            self.num_assigned_workers.value = 0

        with self.stolen_works.get_lock():
            self.stolen_works.value = 0

        with self.assigned_works.get_lock():
            self.assigned_works.value = 0

        with self.num_workers_need_assigned.get_lock():
            self.num_workers_need_assigned.value = 0










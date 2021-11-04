from nnet import FFNN
import multiprocessing
from functools import partial
import copy as cp
import numpy as np
import time

class Methods:
    def __init__(self, dnn:FFNN, properties: list):
        self.dnn = dnn
        self.properties = properties

    def verify(self, relu_linear=False):
        self.dnn.config_only_verify = True
        self.dnn.config_relu_linear = relu_linear
        self.dnn.config_unsafe_input = False
        self.dnn.config_exact_output = False

        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus)

        assert self.properties is not None
        for n, p in enumerate(self.properties):
            initial_input = cp.deepcopy(p.input_set)
            self.dnn.unsafe_domains = p.unsafe_domains

            input_sets = self.dnn.singleLayerOutput(initial_input, 0)
            output_results = []
            output_results.extend(pool.imap(partial(self.dnn.reach, start_layer=1), input_sets))
            verification = [item[0] for sublist in output_results for item in sublist]
            return verification
  



    def nnReach(self, relu_linear=False, unsafe_input=False, exact_output=False):
        assert not (relu_linear and exact_output), \
            "ReLU linearation is not allowed in computing of the exact reachable domain"
        self.dnn.config_only_verify = False
        self.dnn.config_relu_linear = relu_linear
        self.dnn.config_unsafe_input = unsafe_input
        self.dnn.config_exact_output = exact_output

        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus)

        assert self.properties is not None
        for n, p in enumerate(self.properties):
            initial_input = cp.deepcopy(p.input_set)
            self.dnn.unsafe_domains = p.unsafe_domains

            input_sets = self.dnn.singleLayerOutput(initial_input, 0)
            output_results = []
            output_results.extend(pool.imap(partial(self.dnn.reach, start_layer=1), input_sets))
            output_results = [item[1] for sublist in output_results for item in sublist]
            return output_results




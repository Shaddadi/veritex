import sys

import numpy as np

sys.path.insert(0, '../../src')
import torch
from veritex.utils.sfproperty import Property
import math

lbs_input = [-0.2, 0.02, -0.5, -1.0, -20 * math.pi / 180, -0.2, 0.0, 0.0, 0.0, -1.0, -15 * math.pi / 180]
ubs_input = [0.2, 0.5, 0.5, 1.0, 20 * math.pi / 180, 0.2, 0.0, 0.0, 1.0, 1.0, 15 * math.pi / 180]
input_ranges = [lbs_input, ubs_input]


lb_p0 = [-0.2, 0.02, -0.5, -1.0, -20 * math.pi / 180, -0.2, 0.0, 0.0, 0.0, -1.0, -15 * math.pi / 180]
ub_p0 = [0.2, 0.5, 0.5, 1.0, -6 * math.pi / 180, -0.0, 0.0, 0.0, 1.0, 0.0, 0 * math.pi / 180]
lb_p0 = ((np.array(ub_p0) - np.array(lb_p0))*0.5 + np.array(lb_p0)).tolist()

lb_p1 = [-0.2, 0.02, -0.5, -1.0, 6 * math.pi / 180, 0.0, 0.0, 0.0, 0.0, 0.0, 0 * math.pi / 180]
ub_p1 = [0.2, 0.5, 0.5, 1.0, 20 * math.pi / 180, 0.2, 0.0, 0.0, 1.0, 1.0, 15 * math.pi / 180]
lb_p1 = ((np.array(ub_p1) - np.array(lb_p1))*0.5 + np.array(lb_p1)).tolist()

A_unsafe0 = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
d_unsafe0 = torch.tensor([[0.0], [0.0]])
unsafe_domains0 = [[A_unsafe0, d_unsafe0]]

A_unsafe1 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
d_unsafe1 = torch.tensor([[0.0], [0.0]])
unsafe_domains1 = [[A_unsafe1, d_unsafe1]]


property0 = Property([lb_p0, ub_p0], unsafe_domains0, input_ranges=input_ranges)
property1 = Property([lb_p1, ub_p1], unsafe_domains1, input_ranges=input_ranges)

all_properties = {'property0': property0, 'property1': property1}

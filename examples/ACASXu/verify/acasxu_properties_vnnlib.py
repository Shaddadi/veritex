
import torch
from sfproperty import Property
from vnnlib import read_vnnlib_simple

properties = []
for n in range(1,11):
    vnnlib_specs = read_vnnlib_simple('../nets/prop_'+str(n)+'.vnnlib', 5, 5)
    for spec in vnnlib_specs:
        lbs = [item[0] for item in spec[0]]
        ubs = [item[1] for item in spec[0]]
        input_domain = [lbs, ubs]

        unsafe_domains = [[torch.tensor(item[0]),torch.tensor([-item[1]]).T] for item in spec[1]]
        properties.append(Property(input_domain, unsafe_domains))


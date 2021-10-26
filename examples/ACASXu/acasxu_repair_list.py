
from acasxu_properties import *

# the list of neural networks that does not violate any of properties 1-10
safe_nnet_list = [[1,1], [3,3], [4,2]]

# the list of properties that are violated by at least one neural network
# property7 is only for nnet19, property is only for nnet29
# violated_property = [property1, property2, property3, property4, property7, property8]

# create neural networks that need to be repaired and their properties
repair_list = []
for i in range(1,6):
    for j in range(1,10):
        nnet = [i,j]
        if nnet in safe_nnet_list:
            continue
        property_ls =[property1, property2, property3, property4]
        if nnet == [1,7]:
            property_ls.pop()
            property_ls.pop()
        elif nnet == [1,8]:
            property_ls.pop()
            property_ls.pop()
        elif nnet == [1,9]:
            property_ls.pop()
            property_ls.pop()
            property_ls.append(property7)
        elif nnet == [2,9]:
            property_ls.append(property8)

        repair_list.append([nnet, property_ls])







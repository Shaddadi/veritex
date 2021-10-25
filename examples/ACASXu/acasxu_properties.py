import sys
sys.path.insert(0, '../../src')

from scipy.io import loadmat
import numpy as np
from sfproperty import Property


ranges = np.array([6.02610000e+04, 6.28318531e+00, 6.28318531e+00, 1.10000000e+03,
 1.20000000e+03, 3.73949920e+02])
means = np.array([1.97910910e+04, 0.00000000e+00, 0.00000000e+00, 6.50000000e+02,
 6.00000000e+02, 7.51888402e+00])

#  [Clear-of-Conflict, weak left, weak right, strong left, strong right]

lbs_input = [0.0, -3.141593, -3.141593, 100.0, 0.0]
ubs_input = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]
input_ranges = [lbs_input, ubs_input]


for n in range(5):
    lbs_input[n] = (lbs_input[n] - means[n]) / ranges[n]
    ubs_input[n] = (ubs_input[n] - means[n]) / ranges[n]

# property 1
lbs1 = [55947.691, -3.141592, -3.141592, 1145, 0]
ubs1 = [60760, 3.141592, 3.141592, 1200, 60]
for n in range(5):
    lbs1[n] = (lbs1[n] - means[n]) / ranges[n]
    ubs1[n] = (ubs1[n] - means[n]) / ranges[n]

input_domain = [lbs1, ubs1]
A_unsafe = np.array([[-1,0,0,0,0]])
d_unsafe = np.array([3.9911])
unsafe_domains = [[A_unsafe,d_unsafe]]
property1 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)

# property 2
lbs2 = [55947.691, -3.141592, -3.141592, 1145, 0]
ubs2 = [60760, 3.141592, 3.141592, 1200, 60]
for n in range(5):
    lbs2[n] = (lbs2[n] - means[n]) / ranges[n]
    ubs2[n] = (ubs2[n] - means[n]) / ranges[n]
input_domain = [lbs2, ubs2]
A_unsafe = np.array([[-1.0, 1.0, 0, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 0, 1, 0], [-1, 0, 0, 0, 1]])
d_unsafe = np.array([[0.0], [0.0], [0.0], [0.0]])
unsafe_domains = [[A_unsafe,d_unsafe]]
property2 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)

# property 3
lbs3 = [1500, -0.06, 3.1, 980, 960]
ubs3 = [1800, 0.06, 3.141592, 1200, 1200]
for n in range(5):
    lbs3[n] = (lbs3[n] - means[n]) / ranges[n]
    ubs3[n] = (ubs3[n] - means[n]) / ranges[n]
input_domain = [lbs3, ubs3]
A_unsafe = np.array([[1.0, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
d_unsafe = np.array([[0.0], [0.0], [0.0], [0.0]])
unsafe_domains = [[A_unsafe,d_unsafe]]
property3 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)

# property 4
lbs4 = [1500, -0.06, 0, 1000, 700]
ubs4 = [1800, 0.06, 0.000001, 1200, 800]
for n in range(5):
    lbs4[n] = (lbs4[n] - means[n]) / ranges[n]
    ubs4[n] = (ubs4[n] - means[n]) / ranges[n]
input_domain = [lbs4, ubs4]
A_unsafe = np.array([[1.0, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
d_unsafe = np.array([[0.0], [0.0], [0.0], [0.0]])
unsafe_domains = [[A_unsafe,d_unsafe]]
property4 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)

# property 5
lbs5 = [250, 0.2, -3.141592, 100, 0]
ubs5 = [400, 0.4, -3.141592 + 0.005, 400, 400]
for n in range(5):
    lbs4[n] = (lbs4[n] - means[n]) / ranges[n]
    ubs4[n] = (ubs4[n] - means[n]) / ranges[n]
input_domain = [lbs4, ubs4]
unsafe_domains = []
for nn in [0,1,2,3]:
    A_unsafe = np.array([[0, 0, 0, 0, -1.0], ])
    d_unsafe = np.array([[0.0]])
    A_unsafe[0,nn] = 1.0
    unsafe_domains.append([A_unsafe, d_unsafe])

property5 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)


# property 6.1
lbs6 = [12000, 0.7, -3.141592, 100, 0]
ubs6 = [62000, 3.141592, -3.141592 + 0.005, 1200, 1200]
for n in range(5):
    lbs6[n] = (lbs6[n] - means[n]) / ranges[n]
    ubs6[n] = (ubs6[n] - means[n]) / ranges[n]
input_domain = [lbs6, ubs6]
unsafe_domains = []
for nn in [1,2,3,4]:
    A_unsafe = np.array([[-1.0, 0, 0, 0, 0], ])
    d_unsafe = np.array([[0.0]])
    A_unsafe[0,nn] = 1.0
    unsafe_domains.append([A_unsafe, d_unsafe])

property6_1 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)


# property 6.2
lbs6 = [12000, -3.141592, -3.141592, 100, 0]
ubs6 = [62000, -0.7, -3.141592 + 0.005, 1200, 1200]
for n in range(5):
    lbs6[n] = (lbs6[n] - means[n]) / ranges[n]
    ubs6[n] = (ubs6[n] - means[n]) / ranges[n]
input_domain = [lbs6, ubs6]
unsafe_domains = []
for nn in [1,2,3,4]:
    A_unsafe = np.array([[-1.0, 0, 0, 0, 0], ])
    d_unsafe = np.array([[0.0]])
    A_unsafe[0,nn] = 1.0
    unsafe_domains.append([A_unsafe, d_unsafe])

property6_2 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)


# property 7
lbs7 = [0, -3.141592, -3.141592, 100, 0]
ubs7 = [60760, 3.141592, 3.141592, 1200, 1200]
for n in range(5):
    lbs7[n] = (lbs7[n] - means[n]) / ranges[n]
    ubs7[n] = (ubs7[n] - means[n]) / ranges[n]
input_domain = [lbs7, ubs7]
A_unsafe0 = np.array([[-1.0, 0, 0, 1.0, 0], [0, -1.0, 0, 1.0, 0], [0, 0, -1.0, 1.0, 0], [0, 0, 0, 1.0, -1.0]])
d_unsafe0 = np.array([[0.0], [0.0],[0.0],[0.0]])
A_unsafe1 = np.array([[-1.0, 0, 0, 0, 1.0], [0, -1.0, 0, 0, 1.0], [0, 0, -1.0, 0, 1.0], [0, 0, 0, -1.0, 1.0]])
d_unsafe1 = np.array([[0.0], [0.0],[0.0],[0.0]])
unsafe_domains=[[A_unsafe0, d_unsafe0], [A_unsafe1, d_unsafe1]]

property7 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)


# property 8
lbs8 = [0, -3.141592, -0.1, 600, 600]
ubs8 =[60760, -0.75 * 3.141592, 0.1, 1200, 1200]
for n in range(5):
    lbs8[n] = (lbs8[n] - means[n]) / ranges[n]
    ubs8[n] = (ubs8[n] - means[n]) / ranges[n]
input_domain = [lbs8, ubs8]
# # x0>x1, x0>x2, x0>x3, x0>x4
# arry0 = [np.array([[-1.0, 1.0, 0, 0, 0]]),
#          np.array([[-1.0, 0, 1.0, 0, 0]]),
#          np.array([[-1.0, 0, 0, 1.0, 0]]),
#          np.array([[-1.0, 0, 0, 0, 1.0]]),]
# # x1>x0, x1>x2, x1>x3, x1>x4
# arry1 = [np.array([[1.0, -1.0, 0, 0, 0]]),
#          np.array([[0, -1.0, 1.0, 0, 0]]),
#          np.array([[0, -1.0, 0, 1.0, 0]]),
#          np.array([[0, -1.0, 0, 0, 1.0]]),
#          ]
#
# unsafe_domains = []
# for ii in range(4):
#     for jj in range(4):
#         if ii == 0 and jj ==0:
#             continue
#         A_unsafe = np.concatenate((arry0[ii], arry1[jj]),axis=0)
#         d_unsafe = np.array([[0.0],[0.0]])
#         unsafe_domains.append([A_unsafe, d_unsafe])
#
# property8 = Property(input_domain, unsafe_domains)

# the following is from vnn-comp 2021
arry0 = [np.array([[-1.0, 0, 1.0, 0, 0]]),
         np.array([[-1.0, 0, 0, 1.0, 0]]),
         np.array([[-1.0, 0, 0, 0, 1.0]]),]
arry1 = [np.array([[0, -1.0, 1.0, 0, 0]]),
         np.array([[0, -1.0, 0, 1.0, 0]]),
         np.array([[0, -1.0, 0, 0, 1.0]]),]
unsafe_domains = []
for ii in range(3):
    A_unsafe = np.concatenate((arry0[ii], arry1[ii]),axis=0)
    d_unsafe = np.array([[0.0],[0.0]])
    unsafe_domains.append([A_unsafe, d_unsafe])
property8 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)


# property 9
lbs9 = [2000, -0.4, -3.141592, 100, 0]
ubs9 = [7000, -0.14, -3.141592 + 0.01, 150, 150]
for n in range(5):
    lbs9[n] = (lbs9[n] - means[n]) / ranges[n]
    ubs9[n] = (ubs9[n] - means[n]) / ranges[n]
input_domain = [lbs9, ubs9]
unsafe_domains = []
for nn in [0,1,2,4]:
    A_unsafe = np.array([[0, 0, 0, -1.0, 0], ])
    d_unsafe = np.array([[0.0]])
    A_unsafe[0,nn] = 1.0
    unsafe_domains.append([A_unsafe, d_unsafe])

property9 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)


# property 10
lbs10 = [36000, 0.7, -3.141592, 900, 600]
ubs10 =  [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]
for n in range(5):
    lbs10[n] = (lbs10[n] - means[n]) / ranges[n]
    ubs10[n] = (ubs10[n] - means[n]) / ranges[n]
input_domain = [lbs10, ubs10]
unsafe_domains = []
for nn in [1,2,3,4]:
    A_unsafe = np.array([[-1.0, 0, 0, 0, 0], ])
    d_unsafe = np.array([[0.0]])
    A_unsafe[0,nn] = 1.0
    unsafe_domains.append([A_unsafe, d_unsafe])

property10 = Property(input_domain, unsafe_domains, input_ranges=input_ranges)

import os.path
import matplotlib.pyplot as plt
import numpy as np
from vnncomp2021_results.process_vnncomp_results import process_results
from textwrap import wrap


def collect_veritex(filepath):
    veritex_times = []
    log_paths = [f for f in os.listdir(filepath) if f.endswith('.log')]
    assert len(log_paths) == 51
    for indx, log_path in enumerate(log_paths):
        with open(filepath+log_path) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            if indx == 2: # total time for the verification of property 6 on N11
                t = 0.0
                for l in lines:
                    if l[31:44] == "Running Time:":
                        t += float(l.split(" ")[6])
                veritex_times.append(t)
                continue

            for l in lines:
                if l[31:44] == "Running Time:":
                    t = float(l.split(" ")[6])
                    veritex_times.append(t)

    our_times = np.sort(np.array(veritex_times))
    our_times = our_times - min(our_times)*0.5 # overhead
    return our_times



if __name__ == "__main__":
    all_vnncomp_times = process_results()
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    filepath = '../examples/ACASXu/verify/'
    veritex_times = collect_veritex(filepath)
    veritex_times_sum = np.sum(veritex_times)
    # print('Veritex: ', veritex_times_sum)
    for k in all_vnncomp_times.keys():
        temp_times = np.sort(np.array(all_vnncomp_times[k]))
        # if len(temp_times) == 186:
        #     print(k+': ', np.sum(temp_times), ', x', np.sum(temp_times)/veritex_times_sum)
        plt.plot(temp_times, label = k)

    plt.plot(veritex_times,'b', label='Veritex')
    plt.yscale("log")
    plt.xlabel('Number of instances verified', fontsize=13)
    plt.ylabel('Time (sec)', fontsize=13)
    plt.xlim([-1, 186])
    plt.ylim([5e-04, 116])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(ncol=3, fontsize=11)
    plt.title('Fig. 2: Cactus plot of the running time of the safety verification \n for ACAS Xu from VNN-COMP’21 \n'
                             'Caution: the results of related works are from VNNVCOMP\'21 run \n'
                             'on AWS, CPU: r5.12xlarge, 48vCPUs, 384 GB memory, no GPU.')
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/Figure2.png')

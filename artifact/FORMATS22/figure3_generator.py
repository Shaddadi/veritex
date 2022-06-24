import numpy as np
import matplotlib.pyplot as plt
import glob


def collect_veritex_accuracy_runtime():
    log_path = glob.glob("../../examples/ACASXu/repair/logs/*.log")[-1]
    results = get_log_info_veritex(log_path)
    return results


def get_log_info_veritex(log_path):
    def get_time(l):
        t = ""
        start = False
        for ch in l:
            if start:
                t += ch
            if ch == " " and start:
                break
            elif ch == "(":
                start = True

        return float(t)

    with open(log_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        net_IDs = []
        all_accurcys = []
        run_times = []
        safety = []
        accu_per_network = []
        for indx, l in enumerate(lines):
            if l[31:45] == 'Neural Network':
                net_IDs.append((int(l[46]), int(l[48])))
            elif l[31:62] == 'The accurate and safe candidate':
                if l[79:] == 'True':
                    safety.append(True)
                elif l[79:] == 'False':
                    safety.append(False)

                accus = np.fromstring(lines[indx-2][58:-1],sep=',')
                accu_per_network.append(accus[-1])
                all_accurcys.append(np.array(accu_per_network,dtype=object))
                accu_per_network = []
                t = float(lines[indx+1].split(" ")[7])
                run_times.append(t)

            elif l[31:58] == 'Accuracy on the test data: ':
                accus = np.fromstring(l[58:-1], sep='%')
                accu_per_network.append(accus)

        assert len(net_IDs)==len(all_accurcys) and len(net_IDs)==len(run_times) and len(net_IDs)==len(safety)
        return [net_IDs, all_accurcys, run_times, safety]


def collect_art_accuracy_runtime():
    art_log_path_refine = glob.glob("ART/results/acas/art_test_goal_safety/*.log")[-1]
    art_log_path_no_refine = glob.glob("ART/results/acas/art_test_goal_safety_no_refine/*.log")[-1]

    results_refine = get_log_info_art(art_log_path_refine)
    results_no_refine = get_log_info_art(art_log_path_no_refine)
    return results_refine, results_no_refine


def get_log_info_art(log_path):
    def get_time(l):
        t = ""
        start = False
        for ch in l:
            if start:
                t += ch
            if ch == " " and start:
                break
            elif ch == "(":
                start = True

        return float(t)

    with open(log_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        net_IDs = []
        accurcys = []
        run_times = []
        safety = []
        for indx, l in enumerate(lines):
            if l[32:45] == 'For AcasNetID':
                net_IDs.append((int(l[46]), int(l[48])))
            elif l[32:55] == 'Accuracy at every epoch':
                accus = np.fromstring(l[58:-1],sep=',')
                accurcys.append(accus)
                t = get_time(lines[indx+1])
                run_times.append(t)
                if lines[indx+2][78:] == 'True':
                    safety.append(True)
                elif lines[indx+2][78:] == 'False':
                    safety.append(False)


        assert len(net_IDs)==len(accurcys) and len(net_IDs)==len(run_times) and len(net_IDs)==len(safety)
        return [net_IDs, accurcys, run_times, safety]


results_refine_art, _ = collect_art_accuracy_runtime()
results_veritex = collect_veritex_accuracy_runtime()

plt.figure()
for arr in results_refine_art[1]:
    plt.plot(arr,color='#808080')
plt.xlim([0, 25])
plt.ylim([0.75,1.0])
plt.xlabel('Repair iteration',fontsize=13)
plt.ylabel('Accuracy',fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('ART with refinement')
plt.tight_layout()
plt.savefig('results/Figure3(a).png')
plt.close()

plt.figure()
for arr in results_veritex[1]:
    plt.plot(arr*0.01,color='#808080')
plt.xlim([0, 25])
plt.ylim([0.98,1.0])
plt.xlabel('Repair iteration',fontsize=13)
plt.ylabel('Accuracy',fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Our method \n'
          'Caution: the plot may not be the exact same as the one in the paper,\n because each repair cannot produce the exact same DNNs')
plt.tight_layout()
plt.savefig('results/Figure3(b).png')
plt.close()



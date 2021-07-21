import numpy as np
import matplotlib.pyplot as plt


# logs_facetv: facet-vertex
def get_data_facetv(facetv):
    file1 = open(facetv, "r")
    lines = file1.readlines()
    file1.close()
    val = float(lines[0][14:20])
    rel = lines[1][8:-1]
    if rel == 'safe':
        rell = 'SAT'
    else:
        rell = 'UNSAT'

    return [float(val), rell]


if __name__ == "__main__":
    time_facetv = []
    for p in range(1,5):
        for n1 in range(1,6):
            for n2 in range(1,10):
                facetv = 'logs/output_info_' + str(p) + '_' + str(n1) + '_' + str(n2) + '.txt'
                try:
                    time_facetv.append(get_data_facetv(facetv)[0])
                except:
                    pass

    time_facetv = np.sort(np.array(time_facetv)+1)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.semilogy(np.arange(len(time_facetv))+1, time_facetv)
    ax.set_yticks([1,10,100])
    plt.xlabel('Instances')
    plt.ylabel('Time(sec)')
    plt.title('Figure 3: Test on ACAS Xu networks Property 1-4')
    fig.savefig('Figure3.png')

    # property 5-10
    lines = 'Property 5, Network 11\n'
    filex = open('logs/output_info_5_1_1.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Property 6.1, Network 11\n'
    filex = open('logs/output_info_6.1_1_1.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Property 6.2, Network 11\n'
    filex = open('logs/output_info_6.2_1_1.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Property 7, Network 19\n'
    try:
        filex = open('logs/output_info_7_1_9.txt', "r")
        lines += filex.read()
        lines += '\n'
    except:
        lines += 'None\n\n'


    lines += 'Property 8, Network 29\n'
    filex = open('logs/output_info_8_2_9.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Property 9, Network 33\n'
    filex = open('logs/output_info_9_3_3.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Property 10, Network 45\n'
    filex = open('logs/output_info_10_4_5.txt', "r")
    lines += filex.read()
    lines += '\n'

    filex = open('Table1.txt',"w")
    filex.write(lines)
    filex.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import semilogy


if __name__ == "__main__":

    lines = 'Network 11\n'
    filex = open('logs/output_info11.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Network 12\n'
    filex = open('logs/output_info12.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Network 13\n'
    filex = open('logs/output_info13.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Network 14\n'
    filex = open('logs/output_info14.txt', "r")
    lines += filex.read()
    lines += '\n'

    lines += 'Network 15\n'
    filex = open('logs/output_info15.txt', "r")
    lines += filex.read()
    lines += '\n'


    filex = open('Table2.txt',"w")
    filex.write(lines)
    filex.close()

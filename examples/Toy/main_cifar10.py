
import torch
import matplotlib.pyplot as plt
from veritex.utils.plot_poly import plot_polytope2d
from veritex.networks.ffnn import FFNN



if __name__ == '__main__':

    # Load model
    torch_model = torch.load('toy_network.pt')

    # configure the verification
    dnn0 = FFNN(torch_model, verification=True, linearization=linearization)


    sims = reach_model.simulate(num=10000)

    # Plot output reachable sets and simulations
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dim0, dim1 = label.numpy(), target_label.numpy()

    for item in output_sets:
        out_vertices = item[0]
        plot_polytope2d(out_vertices[:, [dim0, dim1]], ax, color='b', alpha=1.0, edgecolor='k', linewidth=0.0,zorder=2)

    ax.plot(sims[:,dim0], sims[:,dim1],'k.',zorder=1)
    ax.autoscale()
    ax.set_xlabel(f'Correct class: $y_{dim0}$', fontsize=16)
    ax.set_ylabel(f'Adversarial class: $y_{dim1}$', fontsize=16)
    plt.title('Reachability analysis of CNNs with input pixels under perturbation. \nBlue area represents the reachable domain. \nBlack dots represent simultations')
    plt.tight_layout()
    plt.show()
    plt.close()




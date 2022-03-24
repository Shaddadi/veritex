
import sys
import os
import torch.multiprocessing
sys.path.insert(0, '../../src')
import matlab.engine
import argparse
import methods
import utils


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force=True)

    parser = argparse.ArgumentParser(description='Paramerers Settings')
    parser.add_argument('--image', type=int, default=1, help='target image')
    #parser.add_argument('--cores', type=int, default=1, help='number of cores')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon')
    parser.add_argument('--relaxation', type=float, default=0.01, help='relaxation')
    parser.add_argument('--falsify', action='store_true')
    #parser.add_argument('--is_cuda', action='store_true', help='GPU usage')

    args = parser.parse_args()
    n = args.image
    cores = torch.multiprocessing.cpu_count()-1
    epsilon = args.epsilon
    top_dims = args.relaxation
    falsify = args.falsify
    is_cuda = False
    pixel_block = [1,1]

    print('Running with '+str(cores)+' CPUs')
    model = utils.load_cifar10()
    image = utils.load_classfied_images_cifar10(model, n)

    file_path = './results'
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path += '/falsify_'+ str(falsify) +'_epsilon_'+str(epsilon)+'_top_dims_' +str(top_dims)
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_pathf = file_path + '/run_info' + str(n)
    reach_model = methods.analysis_method(model, 'cifar10', image, epsilon, file_pathf, pixel_block =pixel_block,
                                          top_dims=top_dims, num_core=cores, falsify=falsify, is_cuda=is_cuda)
    reach_model.reach()

    if not falsify:
        print("Plotting in Matlab")
        eng = matlab.engine.start_matlab()
        eng.plot_sets_cifar10(file_path, n, nargout=0)
        print("Plotting is done")






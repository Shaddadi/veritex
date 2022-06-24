from veritex.methods.reachplot import run
import os

if not os.path.isdir('./results'):
    os.mkdir('./results')

# Figure 4 (a)(b)(c)
prop_path_list = ['../../examples/ACASXu/nets/prop_1.vnnlib', '../../examples/ACASXu/nets/prop_2.vnnlib']
dims = (0,4)

# try:
#     network_path = '../../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
#     savename = 'results/figure4(a)_original_reach_property_1_2_y_1_5'
#     run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
# except:
#     print('figure4(a)_original_reach_property_1_2_y_1_5 is not successfully plotted')
#     pass
#
# try:
#     network_path = '../../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
#     savename = 'results/figure4(b)_veritex_reach_property_1_2_y_1_5'
#     run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
# except:
#     print('figure4(b)_veritex_reach_property_1_2_y_1_5 is not successfully plotted')
#     pass
#
# try:
#     network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
#     savename = 'results/figure4(c)_art_reach_property_1_2_y_1_5'
#     run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
# except:
#     print('figure4(c)_art_reach_property_1_2_y_1_5 is not successfully plotted')
#     pass


# dims = (0,2)
# try:
#     network_path = '../../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
#     savename = 'results/figure4(a)_original_reach_property_1_2_y_1_3'
#     run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
# except:
#     print('figure4(a)_original_reach_property_1_2_y_1_3 is not successfully plotted')
#     pass
#
# try:
#     network_path = '../../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
#     savename = 'results/figure4(b)_veritex_reach_property_1_2_y_1_3'
#     run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
# except:
#     print('figure4(b)_veritex_reach_property_1_2_y_1_3 is not successfully plotted')
#     pass
#
# try:
#     network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
#     savename = 'results/figure4(c)_art_reach_property_1_2_y_1_3'
#     run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
# except:
#     print('figure4(c)_art_reach_property_1_2_y_1_3 is not successfully plotted')
#     pass


prop_path_list = ['../../examples/ACASXu/nets/prop_3.vnnlib', '../../examples/ACASXu/nets/prop_4.vnnlib']
dims = (0,4)

try:
    network_path = '../../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
    savename = 'results/figure5(a)_original_reach_property_3_4_y_1_5'
    run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
except:
    print('figure5(a)_original_reach_property_1_2_y_1_5 is not successfully plotted')
    pass

try:
    network_path = '../../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
    savename = 'results/figure5(b)_veritex_reach_property_3_4_y_1_5'
    run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
except:
    print('figure5(b)_veritex_reach_property_1_2_y_1_5 is not successfully plotted')
    pass

try:
    network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
    savename = 'results/figure5(c)_art_reach_property_3_4_y_1_5'
    run(prop_path_list, network_path, dims, savename, figsize=(2.7, 2.0))
except:
    print('figure5(c)_art_reach_property_1_2_y_1_5 is not successfully plotted')
    pass



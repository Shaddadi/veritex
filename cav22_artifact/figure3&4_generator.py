from veritex.methods.reachplot import run

import time

t0 = time.time()
# Figure 3 (a)(b)(c)
prop_path_list = ['../examples/ACASXu/nets/prop_1.vnnlib', '../examples/ACASXu/nets/prop_2.vnnlib']
dims = (0,1)

network_path = '../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
savename = 'figure3(a)_original_reach_property_1_2_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
savename = 'figure3(b)_art_reach_property_1_2_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = '../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
savename = 'figure3(c)_veritex_reach_property_1_2_dims_0_1'
run(prop_path_list, network_path, dims, savename)

# Figure 3 (d)(e)(f)
prop_path_list = ['../examples/ACASXu/nets/prop_3.vnnlib', '../examples/ACASXu/nets/prop_4.vnnlib']
dims = (0,1)

print('time: ', time.time()-t0)

network_path = '../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
savename = 'figure3(d)_original_reach_property_3_4_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
savename = 'figure3(e)_art_reach_property_3_4_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = '../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
savename = 'figure3(f)_veritex_reach_property_3_4_dims_0_1'
run(prop_path_list, network_path, dims, savename)

print('time: ', time.time()-t0)
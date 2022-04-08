from veritex.methods.reachplot import run
import os

if not os.path.isdir('./results'):
    os.mkdir('./results')
# Figure 3 (a)(b)(c)
prop_path_list = ['../examples/ACASXu/nets/prop_1.vnnlib', '../examples/ACASXu/nets/prop_2.vnnlib']
dims = (0,1)

network_path = '../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
savename = 'results/figure3(a)_original_reach_property_1_2_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = '../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
savename = 'results/figure3(b)_veritex_reach_property_1_2_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
savename = 'results/figure3(c)_art_reach_property_1_2_dims_0_1'
run(prop_path_list, network_path, dims, savename)

# Figure 3 (d)(e)(f)
prop_path_list = ['../examples/ACASXu/nets/prop_3.vnnlib', '../examples/ACASXu/nets/prop_4.vnnlib']
dims = (0,1)

network_path = '../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx'
savename = 'results/figure3(d)_original_reach_property_3_4_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = '../examples/ACASXu/repair/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt'
savename = 'results/figure3(e)_veritex_reach_property_3_4_dims_0_1'
run(prop_path_list, network_path, dims, savename)

network_path = 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet'
savename = 'results/figure3(f)_art_reach_property_3_4_dims_0_1'
run(prop_path_list, network_path, dims, savename)


from examples.DRL.repair.agent_properties import *
# reachablity analysis of networks over small input domains
property1.lbs = ((np.array(property1.ubs)-np.array(property1.lbs))*0.5 + np.array(property1.lbs)).tolist()
property2.lbs = ((np.array(property2.ubs)-np.array(property2.lbs))*0.5 + np.array(property2.lbs)).tolist()
property1.construct_input()
property2.construct_input()


# Figure 4 (a)(b)(c)(d)
dims = (0,1)
network_path = '../examples/DRL/nets/unsafe_agent0.pt'
savename = 'results/figure4(a)_original_reach_property_1_2_dims_0_1'
run([property1, property2], network_path, dims, savename)

network_path = '../examples/DRL/repair/logs/agent_lr1e-06_epochs50_alpha1.0_beta0.0/repaired_model.pt'
savename = 'results/figure4(b)_veritex_reach_property_1_2_dims_0_1'
run([property1, property2], network_path, dims, savename)

dims = (0,2)
network_path = '../examples/DRL/nets/unsafe_agent0.pt'
savename = 'results/figure4(c)_original_reach_property_1_2_dims_0_2'
run([property1, property2], network_path, dims, savename)

network_path = '../examples/DRL/repair/logs/agent_lr1e-06_epochs50_alpha1.0_beta0.0/repaired_model.pt'
savename = 'results/figure4(d)_veritex_reach_property_1_2_dims_0_2'
run([property1, property2], network_path, dims, savename)

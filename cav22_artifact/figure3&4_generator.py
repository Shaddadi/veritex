import os

# Figure 3
os.system("python ../veritex/methods/reachplot.py --property 1,2 --dims 0 1"
          " --network_path '../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx' --savename 'figure3(a)_original_reach_property_1_2_dims_0_1' ")
os.system("python ../veritex/methods/reachplot.py --property 1,2 --dims 0 1"
          " --network_path 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet' --savename 'figure3(b)_art_reach_property_1_2_dims_0_1' ")
os.system("python ../veritex/methods/reachplot.py --property 1,2 --dims 0 1"
          " --network_path '../examples/ACASXu/repaired/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt' --savename 'figure3(c)_veritex_reach_property_1_2_dims_0_1' ")

os.system("python ../veritex/methods/reachplot.py --property 3,4 --dims 0 1"
          " --network_path '../examples/ACASXu/nets/ACASXU_run2a_2_1_batch_2000.onnx' --savename 'figure3(d)_original_reach_property_3_4_dims_0_1' ")
os.system("python ../veritex/methods/reachplot.py --property 3,4 --dims 0 1"
          " --network_path 'ART/results/acas/art_test_goal_safety/repaired_network_21_safe.nnet' --savename 'figure3(e)_art_reach_property_3_4_dims_0_1' ")
os.system("python ../veritex/methods/reachplot.py --property 3,4 --dims 0 1"
          " --network_path '../examples/ACASXu/repaired/logs/nnet21_lr0.001_epochs200_alpha1.0_beta0.0/repaired_model.pt' --savename 'figure3(f)_veritex_reach_property_3_4_dims_0_1' ")
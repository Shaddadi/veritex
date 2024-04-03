from veritex.methods.verify import run
import os

# get current directory
currdir = os.path.dirname(os.path.abspath(__file__))

# collect safety properties
properties = []
for n in range(1, 11):
    properties.append(f'{currdir}/../nets/prop_' + str(n) + '.vnnlib')

# verify all networks on properties 1-4
for i in range(1, 6):
    for j in range(1, 10):
        nn_path = f"{currdir}/../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        netname, propname = str(i)+str(j), ['1','2','3','4']
        run(properties[0:4], nn_path, netname, propname)

# verify network 11 on property 5
nn_path = f"{currdir}/../nets/ACASXU_run2a_1_1_batch_2000.onnx"
netname, propname = '11', ['5']
run([properties[4]], nn_path, netname, propname)

# verify network 11 on property 6
nn_path = f"{currdir}/../nets/ACASXU_run2a_1_1_batch_2000.onnx"
netname, propname = '11', ['6.1','6.2']
run([properties[5]], nn_path, netname, propname)

# verify network 19 on property 7
nn_path = f"{currdir}/../nets/ACASXU_run2a_1_9_batch_2000.onnx"
netname, propname = '19', ['7']
run([properties[6]], nn_path, netname, propname)

# verify network 29 on property 8
nn_path = f"{currdir}/../nets/ACASXU_run2a_2_9_batch_2000.onnx"
netname, propname = '29', ['8']
run([properties[7]], nn_path, netname, propname)

# verify network 33 on property 9
nn_path = f"{currdir}/../nets/ACASXU_run2a_3_3_batch_2000.onnx"
netname, propname = '33', ['9']
run([properties[8]], nn_path, netname, propname)

# verify network 45 on property 10
nn_path = f"{currdir}/../nets/ACASXU_run2a_4_5_batch_2000.onnx"
netname, propname = '45', ['10']
run([properties[9]], nn_path, netname, propname)
# Tool for Reachability Analysis and Repair of Neural Networks

Veritex is an object-oriented software programmed in Python. It takes in two inputs, the network model and safety properties. Veritex supports the standardized format ONNX and PyTorch for the network and the unified format Vnnnlib for the safety property. With the network model and its safety properties, Veritex can  
* compute the exact or over-approximated output reachable domain and also the entire unsafe input space if exists,
* plot 2 or 3-dimensional reachable domains,
* produce a provable safe network in ONNX or PyTorch format when the repair option is enabled.

<p align="center">
   <img src="veritex.png" style="width:80%">
</p>
<p align="center"> Figure: An overview of Veritex architecture.</p>

## Install

Clone this repository to your local machine.

```bash
git clone https://github.com/Shaddadi/veritex.git
cd veritex
```

### Option 1: docker installing as a User (recommend for CAV'22 artifact)

1. Build the image from the dockerfile.

    ```bash
    sudo docker build . -t veritex_image
    ```

1. Create and start the docker container.

    ```bash
    sudo docker run --rm -it veritex_image bash
    ```

### Option 2: installing as a User

This tool is confirmed with only Python3.7.
Simply you may install veritex pkg with pip.

```bash
python3.7 -m pip install .
```

### Option 3: installing as a Developer

This tool is confirmed with only Python3.7.

1. Install required python packages.

    ```bash
    python3.7 -m pip install -e .
    ```

1. Set path to /veritex under this repository.

    ```bash
    export PYTHONPATH='<YOUR_REPO_PATH>/veritex'
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    ```

## Run experiments

### CAV'22 Artifact

Linux systems are suggested. This artifact aims to reproduce results in the CAV'22 tool paper, including **Figure 2&3&4** and **Table 2&3**. Results are stored in 'veritex/cav22_artifact/results'. There are two versions for the artifact evaluation. The difference between these two versions is that the first one does not include the repair of two neural networks which consumes a large amount of memory and time. 

**Caution**: Reachable domains of networks in Figure 3&4 may be slightly different from the ones in the paper because each run of the repair method can not guarantee to produce the exact same safe network.
**Caution**: For *Windows* users who encounter the error '\r command not found' when implementing the artifact, please run the following commands before the shell script.
```bash
apt-get update
apt-get install dos2unix
dos2unix reproduce_results1.sh
dos2unix reproduce_results2.sh
```

1. The first version reproduces the results in the paper except for two hard instances (~170 mins), including
   * safety verification of all instances (data generation for Figure 2) (~2 mins),
   * repair of 33/35 unsafe instances (data generation for Figure 3 and most of results in Table 2&3) (~40 mins),
   * repair of an unsafe DNN agent (data generation for Figure 4) (~6 mins),
   * implementation of the related work ART for the repair comparison (~90 mins),
   * generation of figures and tables (~40 mins, majority of the time is spent on the plot of reachable domains).

   This version requires at least 32 GB memory.

   ```bash
   cd cav22_artifact
   bash reproduce_results1.sh
   ```

2. The second version reproduces all the results in the paper (~410 mins), including
   * safety verification of all instances (data generation for Figure 2) (~2 mins),
   * repair of all 35/35 unsafe instances (data generation for Figure 3 and Table 2&3) (~280 mins),
   * repair of an unsafe DNN agent (data generation for Figure 4) (~6 mins),
   * implementation of the related work ART for the repair comparison (~90 mins),
   * generation of figures and tables (~40 mins, majority of the time is spent on the plot of reachable domains).

   The hardware requirement for second version is AWS, CPU: r5.12xlarge, 48vCPUs, 384 GB memory, no GPU.

   ```bash
   cd cav22_artifact
   bash reproduce_results2.sh
   ```

### Demo

This demo includes the computation of the exact output reachable domain of a neural network using our reachability analysis method.
It also includes the computation of its exact unsafe input space that leads to safety violations in the output using our Backtracking algorithm.
The neural network consists of 3 inputs, 2 outputs, and 8 layers with each having 7 neurons.
Results will be saved in /images.

```bash
cd examples/Demo
python main_demo.py
```

<p align="center">
    <img src="examples/Demo/reach_demo.gif" style="width:70%">
</p>
<p align="center">
   Figure: Demo for our reachability analysis algorithms. Given an input domain (the blue box), our algorithms compute the exact output reachable domain and also the exact unsafe input subspace that leads to safety violation in the output domain.
</p>

### ACAS experiments

1. Run the verification of ACAS Xu neural networks. Info will be logged

    ```bash
    cd examples/ACASXu/verify
    ./verify_all_instances.sh
    ```

1. Run the repair of the unsafe ACAS Xu neural networks. Repaired networks will be saved in /logs and the info will be logged.

    ```bash
    cd examples/ACASXu/repair
    python repair_nnets.py
    ```

### Visualize Reachable Domains of ACASXu Networks

Visualize the output reachable domain

```bash
cd examples/ACASXu/repair

python ../../../veritex/methods/reachplot.py \
--property 'path_to_property1'...'path_to_propertyn' \
--network_path 'path_to_model' \
--dims x x \
--savename 'xxx'
```

Example:

```bash
python ../../../veritex/methods/reachplot.py \
--property '../nets/prop_3.vnnlib' '../nets/prop_4.vnnlib' \
--network_path '../nets/ACASXU_run2a_2_1_batch_2000.onnx' \
--dims 0 1 \
--savename 'figures/reachable_domain_property_3,4_dims0_1'


python ../../../veritex/methods/reachplot.py \
--property '../nets/prop_3.vnnlib' '../nets/prop_4.vnnlib' \
--network_path '../nets/ACASXU_run2a_2_1_batch_2000.onnx' \
--dims 0 2 \
--savename 'figures/reachable_domain_property_3,4_dims0_2'
```

<p align="center">
    <img src="examples/ACASXu/repair/figures/reachable_domain_property_3,4_dims0_1.png" style="width:20%">
    <img src="examples/ACASXu/repair/figures/reachable_domain_property_3,4_dims0_2.png" style="width:20%">
</p>
<p align="center">Figure: Output reachable domains of Network21 on Properties 3 and 4. They are projected on (y0,y1) and (y0, y2).
</p>

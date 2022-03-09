# Tool for Reachability Analysis and Repair of Neural Networks
Veritex is an object-oriented software programmed in Python. It takes in two inputs, the network model and safety properties. Veritex supports the standardized format ONNX and PyTorch for the network and the unified format Vnnnlib for the safety property. With the network model and its safety properties, Veritex can compute the exact or over-approximated output reachable domain and also the entire unsafe input space if exists. It supports the plotting of 2 or 3-dimensional polytopes. When the repair option is enabled, it will produce a provable safe network in ONNX or PyTorch format.
<p align="center">
   <img src="veritex.png" style="width:70%">
    <figcaption>Figure: An overview of Veritex architecture.</figcaption>
</p>

## Usage

### I. Docker setup

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/Shaddadi/veritex.git
    ```

2. Build the image from the dockerfile

    ```bash
    sudo docker build . -t veritex_image
    ```

3. Create the docker container

    ```bash
    sudo docker run --rm -it veritex_image bash
    ```

### II. Run Demo

This demo includes the computation of the exact output reachable domain of a neural network using our reachability analysis method. It also includes the computation of its exact unsafe input space that leads to safety violations in the output using our Backtracking algorithm. The neural network consists of 3 inputs, 2 outputs, and 8 layers with each having 7 neurons. Results will be saved in /images.

```bash
cd examples/Demo
python main_demo.py
````
<figure>
    <img src="examples/Demo/reach_demo.gif" style="width:70%">
    <figcaption>Figure: Demo for our reachability analysis algorithms. Given an input domain (the blue box), our algorithms compute the exact output reachable domain and also the exact unsafe input subspace that leads to safety violation in the output domain.</figcaption>
</figure>

### III. Run ACAS experiments

1. Run the verification of ACAS Xu neural networks. Info will be logged

    ```bash
    cd examples/ACASXu/verify
    ./verify_all_instances.sh
    ```

2. Run the repair of the unsafe ACAS Xu neural networks. Repaired networks will be saved in /logs and the info will be logged.

    ```bash
    cd examples/ACASXu/repair
    python main_repair_nnets.py
    ```

### IV. Visualize Reachable Domains of ACASXu Networks

Visualize the output reachable domain

```bash
cd examples/ACASXu/repair
python main_reachable_domain.py --property x,x --dims x x --network_path 'xxx'
```

Example:

```bash
python main_reachable_domain.py --property 3,4 --dims 0 1 --network_path '../nets/ACASXU_run2a_2_1_batch_2000.onnx'
python main_reachable_domain.py --property 3,4 --dims 0 2 --network_path '../nets/ACASXU_run2a_2_1_batch_2000.onnx'
```

<figure>
    <div>
        <img src="examples/ACASXu/repair/images/reachable_domain_property_3,4_dims0_1.png" style="width:20%">
        <img src="examples/ACASXu/repair/images/reachable_domain_property_3,4_dims0_2.png" style="width:20%">
    </div>
    <figcaption> Figure: Output reachable domains of Network21 on Properties 3 and 4. They are projected on (y0,y1) and (y0, y2).
    </figcaption>
</figure>


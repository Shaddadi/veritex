# Tool for Reachability Analysis and Repair of Neural Networks
## Run with Docker
1. Clone this repository to your local machine.
2. Build the image for the dockerfile
```bash
sudo docker build . -t veritex_image
```
4. Create the docker container
```bash
sudo docker run --rm -it veritex_image bash
```
6. In the container
## Demo
<figure>
    <img src="examples/Demo/reach_analysis.gif" style="width:70%"> 
    <figcaption>Figure: Demo for our reachability analysis algorithms. The network consists of 3 inputs, 2 outputs and 8 layers with each having 7 neurons. Given an input domain (the blue box), our algorithms compute the exact output reachable domain and also the exact unsafe input subspace that leads to safety violation in the output domain.</figcaption>
</figure>



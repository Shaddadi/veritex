# ART

The Head commit ID of this version is `2a734f960527f7b680cb42b1577f84fc9db68a2f`

The code for paper "ART: Abstraction Refinement-Guided Training for Provably
Correct Neural Networks" appearing in FMCAD'20. See tag `fmcad20`.


## Installation

In your virtual environment, either install directly from this repository by
```
git clone git@github.com:XuankangLin/ART.git
cd ART
pip install -r requirements.txt
```

To replay the evaluation, run individual script from `scripts/` by
```
bash scripts/...
```
The corresponding logs have been saved in `results/xxx/...` directory.

Alternatively, one can directly run via Docker at
[xuankanglin/art](https://hub.docker.com/r/xuankanglin/art). To run jupyter
notebook in Docker, use customizable commands like
```
docker run -p 8888:8888 --rm --gpus=all xuankanglin/art jupyter notebook --allow-root --ip 0.0.0.0 --no-browser --port 8888
```
then visit through `localhost:8888` in browser, assuming `nvidia-docker2` is
installed to enable GPU.


## License

The project is available open source under the terms of [MIT
License](https://opensource.org/licenses/MIT).

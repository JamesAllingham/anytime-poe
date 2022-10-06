# Any-time Produce of Experts Ensembles

When evaluating an ensemble it is not clear how many members are required to make an accurate prediction for a given example.
This is a particular problem when using computationally expensive NN models in compute-constrained or low-latency applications. 
We propose a novel product of experts based method for conditional-computation to address this issue.
We rely on the fact that the product of hard, e.g., uniform, probability distributions, has support less that or equalto that of the multiplicands. 
Thus, by setting a confidence threshold, we can stop computing ensemble members for a given input once the size
of the support has been sufficiently reduced.

## Getting Started

```bash
module load python-3.9.6-gcc-5.4.0-sbr552h
# ^ for Cambridge Uni HPC only
sudo apt-get install python3.9-venv
# ^ if not already installed
python3.9 -m venv ~/.virtualenvs/any-poe
source ~/.virtualenvs/any-poe/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
python3.9 -m ipykernel install --user --name=any-poe
# ^ optional, for easily running IPython/Jupyter notebooks with the virtual env.
```

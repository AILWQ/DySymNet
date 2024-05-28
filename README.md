## DySymNet

![overview](img/Overview.png)

This repository contains the official Pytorch implementation for the paper [***A Neural-Guided Dynamic Symbolic Network for Exploring Mathematical Expressions from Data***](https://openreview.net/forum?id=pTmrk4XPFx) [ICML 2024].

[![Paper](https://img.shields.io/badge/Paper-📖-blue?)](https://openreview.net/pdf?id=IejxxE9DO2)
![GitHub Repo stars](https://img.shields.io/github/stars/AILWQ/DySymNet?style=social)
[![Paper](https://img.shields.io/badge/arXiv-2309.13705-b31b1b)](https://arxiv.org/abs/2309.13705)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=AILWQ.DySymNet&left_color=red&right_color=green&left_text=Hello%20Visitors)



## 🔥 Highlights

- Our proposed ***<span style="font-variant: small-caps;">DySymNet</span>*** is a new search paradigm for symbolic regression (SR) that searches the symbolic network with various architectures instead of searching expressions in the large functional space.
- ***<span style="font-variant: small-caps;">DySymNet</span>*** possesses promising capabilities in solving high-dimensional problems and optimizing coefficients, which are lacking in current SR methods.
- Extensive numerical experiments demonstrated that ***<span style="font-variant: small-caps;">DySymNet</span>*** outperforms state-of-the-art baselines across various SR standard benchmark datasets and the well-known SRBench with more variables.

## 📦 Requirements

Install the conda environment and packages:

```setup
conda env create -f environment.yml
conda activate dysymnet
```

The packages have been tested on Linux.

## 📋 Getting started

### Configure runs

The main running script is `SymbolicRegression.py` and it relies on configuring runs via `params.py`. The `params.py` includes various hyperparameters of the controller RNN and the symbolic network. You can configure the following hyperparameters as required:

#### parameters for symbolic network structure

- `funcs_avail`  configures the operator library and It's part of the search space. You can add the additional operators in the `functions.py` by referring to existing operators and place them inside `funcs_avail`  if you want to use them.
- `n_layers` configures the number library of symbolic network layers. It's part of the search space.
- `num_func_layer` configure number library of operators in each layer. It's part of the search space.

#### parameters for controller RNN

- `num_epochs`  configures the epochs for sampling
- `batch_size`  configures the size for a batch sampling
- `input_size`  configures the input dim
- `optimizer`  configures the optimizer for training RNN
- `hidden_size`  configures the hidden dim
- `embedding_size`  configures the embedding dim
- `learning_rate1`  configures the learning rate for training RNN
- `risk_seeking`  configures using risk seeking policy gradient or not
- `risk_factor`  configures the risk factor
- `entropy_weight`  configures the entropy weight
- `reward_type`  configures the error type for computing reward. Default: mse

#### parameters for symbolic network training

- `learning_rate2` configures the learning rate
- `reg_weight`  configures the regularizaiton weight
- `threshold`  configures the prunning threshold
- `trials`  configures the training trials
- `n_epochs1`  configures the epochs for the first training stage
- `n_epochs2`  configures the epochs for the second training stage
- `summary_step`  configures to summary for every n training steps
- `clip_grad`  configures using adaptive gradient clipping or not
- `max_norm` configures the norm threshold for gradient clipping
- `window_size`  configures the window size for adaptive gradient clipping
- `refine_constants`  confifures refining constants or not
- `n_restarts`  configures the number of restarts for BFGS optimization
- `add_bias`  configures adding bias or not
- `verbose`  configures printing training process or not
- `use_gpu`  configures using cuda or not
- `plot_reward`  configures plotting reward curve or not

#### parameters for genearting input data

- `N_TRAIN`  configures the size of training dataset
- `N_VAL`  configures the size of validation dataset
- `NOISE` = 0  configures the standard deviation of noise for training dataset
- `DOMAIN`  configures the domain of input data
- `N_TEST`  configures the size of test dataset
- `DOMAIN_TEST`  configures the domain of test dataset

#### Additional parameters

`results_dir` configures the save path for all results

### 🤖 Symbolic Regression

We provide two ways to perform symbolic regression tasks.

#### Option1: Input ground truth expression

When you want to discover an expression for which the ground truth is known, for example to test a standard benchmark, you can edit the script `SymbolicRegression.py`  as follows:

```python
# SymbolicRegression.py
params = Params()  # configuration for a specific task
ground_truth_eq = "x_1 + x_2"  # variable names should be written as x_i, where i>=1.
eq_name = "x_1+x_2"
SR = SymbolicRegression(config=params, func=ground_truth_eq, fun_name=eq_name)  # A new folder named "func_name" will be created to store the result files.
eq, R2, error, relative_error = SR.solve_environment()  # return results
```

In this way, the function `generate_data` is used to automatically generate the corresponding data set $\mathcal{D}(X, y)$ for inference, instead of you generating the data yourself.

Then, you can run `SymbolicRegression.py` directly, or you can run it in the terminal as follows:

```python
python SymbolicRegression.py
```

After running this script, the results will be stored in path `./results/test/func_name`.

#### Option2: Load the data file

When you only have observed data and do not know the ground truth, you can perform symbolic regression by entering the path to the data file:

```python
# SymbolicRegression.py
params = Params()  # configuration for a specific task
data_path = 'path/to/your/data'
SR = SymbolicRegression(config=params, func_name='blackbox', data_path=data_path)  # you can rename the func_name as any other you want.
eq, R2, error, relative_error = SR.solve_environment()  # return results
```

**Note that** you should implement the `load_data(self, data)` funcion according to the data file. The shape  of returned $X$ should be (num_points, x_dim), and $y$ should be (num_points, 1).

Then, you can run `SymbolicRegression.py` directly, or you can run it in the terminal as follows:

```python
python SymbolicRegression.py
```

After running this script, the results will be stored in path `./results/test/blackbox`.

#### Output

Once the script stops early or finishes running, you will get the following output:

```
Expression: x_1 + x_2
R2: 1.0
error: 4.3591795754679974e-13
relative_error:  2.036015757767018e-06
log(1 + MSE):  4.3587355946774144e-13
```

## 🔗 Citing this work

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{li2023neural,
  title={A Neural-Guided Dynamic Symbolic Network for Exploring Mathematical Expressions from Data},
  author={Li, Wenqiang and Li, Weijun and Yu, Lina and Wu, Min and Liu, Jingyi and Li, Yanjie},
  journal={arXiv preprint arXiv:2309.13705},
  year={2023}
}
```


## A Neural-Guided Dynamic Symbolic Network for Exploring Mathematical Expressions from Data

![overview](img/Overview.png)

This repository is the official implementation of [***A Neural-Guided Dynamic Symbolic Network for Exploring Mathematical Expressions from Data***](https://openreview.net/forum?id=pTmrk4XPFx) submitted to NeurIPS'23.

## Highlights

- Our proposed ***<span style="font-variant: small-caps;">DySymNet</span>*** is a new search paradigm for symbolic regression (SR) that searches the symbolic network with various architectures instead of searching expressions in the large functional space.
- ***<span style="font-variant: small-caps;">DySymNet</span>*** possesses promising capabilities in solving high-dimensional problems and optimizing coefficients, which are lacking in current SR methods.
- Extensive numerical experiments demonstrated that ***<span style="font-variant: small-caps;">DySymNet</span>*** outperforms state-of-the-art baselines across various SR standard benchmark datasets and the well-known SRBench with more variables.

## Requirements

Install the conda environment and packages:

```setup
conda env create -f environment.yml
conda activate dysymnet
```

The packages have been tested on Linux.

## Getting started

### Configure runs

The main running script is `policy_gradient.py` and it relies on configuring runs via `params.py`. The `params.py` includes various hyperparameters of the controller RNN and the symbolic network. You can configure the following hyperparameters as required:

#### parameters of symbolic network structure

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

#### parameters for genearting dataset

- `N_TRAIN`  configures the size of training dataset
- `N_VAL`  configures the size of validation dataset
- `NOISE` = 0  configures the standard deviation of noise for training dataset
- `DOMAIN`  configures the domain of dataset - range from which we sample x
- `N_TEST`  configures the size of test dataset
- `DOMAIN_TEST`  configures the domain of test dataset

#### other parameters

`results_dir` configures the save path for all results

### Symbolic Regression

TODO

## Results

Our approach achieves the state-of-the-art  performance on **Standard benchmarks** and **SRBench benchmark**.

### Pareto plot on SRBench benchmark

**Our proposed *<span style="font-variant: small-caps;">DySymNet</span>* outperforms previous DL-based and GP-based SR methods in terms of fitting accuracy while maintaining a relatively small symbolic model size.** Pareto plot comparing the average test performance and model size of our method with baselines provided by the SRBench benchmark, both on *Feynman* dataset (left) and *Black-box* dataset (right).

![pareto](img/Pareto_DySymNet.png)

### Fitting accuracy comparison

![Fitting accuracy](img/Fitting_accuracy.png)

**See the [paper](https://openreview.net/forum?id=pTmrk4XPFx) for more experimental results.**

from .functions import *


class Params:
    # Optional operators during sampling
    funcs_avail = [Identity(),
                   Sin(),
                   Cos(),
                   # Tan(),
                   # Exp(),
                   # Log(),
                   # Sqrt(),
                   Square(),
                   # Pow(3),
                   # Pow(4),
                   # Pow(5),
                   # Pow(6),
                   Plus(),
                   Sub(),
                   Product(),
                   # Div()
                   ]
    n_layers = [2, 3, 4, 5]  # optional number of layers
    num_func_layer = [2, 3, 4, 5, 6]  # optional number of functions in each layer

    # symbolic network training parameters
    learning_rate2 = 1e-2
    reg_weight = 5e-3
    threshold = 0.05
    trials = 1  # training trials of symbolic network
    n_epochs1 = 10001
    n_epochs2 = 10001
    summary_step = 1000
    clip_grad = True  # clip gradient or not
    max_norm = 1  # norm threshold for gradient clipping
    window_size = 50  # window size for adaptive gradient clipping
    refine_constants = True  # refine constants or not
    n_restarts = 1  # number of restarts for BFGS optimization
    add_bias = False  # add bias or not
    verbose = True  # print training process or not
    use_gpu = False  # use cuda or not
    plot_reward = False  # plot reward or not

    # controller parameters
    num_epochs = 500
    batch_size = 10
    if isinstance(n_layers, list) or isinstance(num_func_layer, list):
        input_size = max(len(n_layers), len(num_func_layer))
    else:
        input_size = len(funcs_avail)
    optimizer = "Adam"
    hidden_size = 32
    embedding_size = 16
    learning_rate1 = 0.0006
    risk_seeking = True
    risk_factor = 0.5
    entropy_weight = 0.005
    reward_type = "mse"  # mse, nrmse

    # dataset parameters
    N_TRAIN = 100  # Size of training dataset
    N_VAL = 100  # Size of validation dataset
    NOISE = 0  # Standard deviation of noise for training dataset
    DOMAIN = (-1, 1)  # Domain of dataset - range from which we sample x. Default (-1, 1)
    # DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])   # Use this format if each input variable has a different domain
    N_TEST = 100  # Size of test dataset
    DOMAIN_TEST = (-1, 1)  # Domain of test dataset - should be larger than training domain to test extrapolation. Default (-2, 2)
    var_names = [f'x_{i}' for i in range(1, 21)]  # not used

    # save path
    results_dir = './results/test'

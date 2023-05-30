import json
import os
import pickle
import time
from inspect import signature
import os
import torch
import sympy as sp
from scipy.optimize import minimize
from functions import *
import functions
import collections
import numpy as np
import matplotlib.pyplot as plt
from sympy import parse_expr, symbols, Float
from torch import nn, optim
from controller import Agent
import torch.nn.functional as F
import pretty_print
from regularization import L12Smooth
from symbolic_network import SymbolicNet
from sklearn.metrics import r2_score
from params import Params
from utils import nrmse, R_Square, MSE, Relative_Error


def generate_data(func, N, range_min, range_max):
    """Generates datasets."""
    free_symbols = sp.sympify(func).free_symbols
    x_dim = free_symbols.__len__()
    sp_expr = sp.lambdify(free_symbols, func)
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[sp_expr(*x_i)] for x_i in x])
    return x, y


class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)  # sp.lambdify()
        self.x = x
        return self.fun_value


class PolicyGradient:
    def __init__(self, config, func=None, func_name=None, data_path=None):
        """
        Args:
            config: All configs in the Params class, type: Params
            func: the function to be predicted, type: str
            func_name: the name of the function, type: str
            data_path: the path of the data, type: str
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.funcs_per_layer = None
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.input_size = config.input_size  # number of operators
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.n_layers = config.n_layers
        self.num_func_layer = config.num_func_layer
        self.funcs_avail = config.funcs_avail
        self.optimizer = config.optimizer
        self.auto = False
        self.add_bias = config.add_bias
        self.threshold = config.threshold

        self.clip_grad = config.clip_grad
        self.max_norm = config.max_norm
        self.window_size = config.window_size
        self.refine_constants = config.refine_constants
        self.n_restarts = config.n_restarts
        self.reward_type = config.reward_type

        if config.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("Use Device:", self.device)

        # Standard deviation of random distribution for weight initializations.
        self.init_sd_first = 0.1
        self.init_sd_last = 1.0
        self.init_sd_middle = 0.5

        self.config = config

        self.func = func
        self.func_name = func_name

        # generate data or load data from file
        if self.func is not None:
            # add noise
            if config.NOISE > 0:
                self.X, self.y = generate_data(func, self.config.N_TRAIN, self.config.DOMAIN[0], self.config.DOMAIN[1])  # y shape is (N, 1)
                y_rms = torch.sqrt(torch.mean(self.y ** 2))
                scale = config.NOISE * y_rms
                self.y += torch.empty(self.y.shape[-1]).normal_(mean=0, std=scale)
                self.x_test, self.y_test = generate_data(func, self.config.N_TRAIN, range_min=self.config.DOMAIN_TEST[0],
                                                         range_max=self.config.DOMAIN_TEST[1])

            else:
                self.X, self.y = generate_data(func, self.config.N_TRAIN, self.config.DOMAIN[0], self.config.DOMAIN[1])  # y shape is (N, 1)
                self.x_test, self.y_test = generate_data(func, self.config.N_TRAIN, range_min=self.config.DOMAIN_TEST[0],
                                                         range_max=self.config.DOMAIN_TEST[1])
        else:
            self.X, self.y = self.load_data(self.data_path)
            self.x_test, self.y_test = self.X, self.y

        self.dtype = self.X.dtype  # obtain the data type, which determines the parameter type of the model

        if isinstance(self.n_layers, list) or isinstance(self.num_func_layer, list):
            print('*' * 16, 'Sampling...', '*' * 16)
            self.auto = True

        self.agent = Agent(auto=self.auto, input_size=self.input_size, hidden_size=self.hidden_size,
                           num_funcs_avail=len(self.funcs_avail), n_layers=self.n_layers,
                           num_funcs_layer=self.num_func_layer, device=self.device, dtype=self.dtype)

        self.agent = self.agent.to(self.dtype)

        if not os.path.exists(self.config.results_dir):
            os.makedirs(self.config.results_dir)

        func_dir = os.path.join(self.config.results_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)
        self.results_dir = func_dir

        self.now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # save hyperparameters
        args = {
            "date": self.now_time,
            "add_bias": config.add_bias,
            "train_domain": config.DOMAIN,
            "test_domain": config.DOMAIN_TEST,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "risk_factor": config.risk_factor,
            "n_layers": config.n_layers,
            "num_func_layer": config.num_func_layer,
            "funcs_avail": str([func.name for func in config.funcs_avail]),
            "init_sd_first": 0.1,
            "init_sd_last": 1.0,
            "init_sd_middle": 0.5,
            "noise_level": config.NOISE
        }
        with open(os.path.join(self.results_dir, 'args_{}.txt'.format(self.func_name)), 'a') as f:
            f.write(json.dumps(args))
            f.write("\n")
        f.close()

    def solve_environment(self):
        epoch_best_expressions = []
        epoch_best_rewards = []
        epoch_mean_rewards = []
        epoch_mean_r2 = []
        epoch_best_r2 = []
        epoch_best_relative_error = []
        epoch_mean_relative_error = []
        best_expression, best_performance, best_relative_error = None, float('-inf'), float('inf')
        early_stopping = False

        # log the expressions of all epochs
        f1 = open(os.path.join(self.results_dir, 'eq_{}_all.txt'.format(self.func_name)), 'a')
        f1.write('\n{}\t\t{}\n'.format(self.now_time, self.func_name))
        f1.write('{}\t\tReward\t\tR2\t\tExpression\t\tnum_layers\t\tnum_funcs_layer\t\tfuncs_per_layer\n'.format(self.reward_type))

        # log the best expressions of each epoch
        f2 = open(os.path.join(self.results_dir, 'eq_{}_summary.txt'.format(self.func_name)), 'a')
        f2.write('\n{}\t\t{}\n'.format(self.now_time, self.func_name))
        f2.write('Epoch\t\tReward\t\tR2\t\tExpression\n')

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.config.learning_rate1)
        else:
            optimizer = torch.optim.RMSprop(self.agent.parameters(), lr=self.config.learning_rate1)

        for i in range(self.num_epochs):
            print("******************** Epoch {:02d} ********************".format(i))
            expressions = []
            rewards = []
            r2 = []
            relative_error_list = []
            batch_log_probs = torch.zeros([self.batch_size], device=self.device)
            batch_entropies = torch.zeros([self.batch_size], device=self.device)

            j = 0
            while j < self.batch_size:
                error, R2, eq, log_probs, entropies, num_layers, num_func_layer, funcs_per_layer_name = self.play_episodes()  # play an episode
                # if the expression is invalid, e.g. a constant or None, resample the structure of the symbolic network
                if 'x_1' not in str(eq) or eq is None:
                    R2 = 0.0
                if 'x_1' in str(eq) and self.refine_constants:
                    res = self.bfgs(eq, self.X, self.y, self.n_restarts)
                    eq = res['best expression']
                    R2 = res['R2']
                    error = res['error']
                    relative_error = res['relative error']
                else:
                    relative_error = 100

                reward = 1 / (1 + error)
                print("Final expression: ", eq)
                print("Test R2: ", R2)
                print("Test error: ", error)
                print("Relative error: ", relative_error)
                print("Reward: ", reward)
                print('\n')

                f1.write('{:.8f}\t\t{:.8f}\t\t{:.8f}\t\t{:.8f}\t\t{}\t\t{}\t\t{}\t\t{}\n'.format(error, relative_error, reward, R2, eq, num_layers,
                                                                                                 num_func_layer,
                                                                                                 funcs_per_layer_name))

                if R2 > 0.99:
                    print("~ Early Stopping Met ~")
                    print("Best expression: ", eq)
                    print("Best reward:     ", reward)
                    print(f"{self.config.reward_type} error:      ", error)
                    print("Relative error:  ", relative_error)
                    early_stopping = True
                    break

                batch_log_probs[j] = log_probs
                batch_entropies[j] = entropies
                expressions.append(eq)
                rewards.append(reward)
                r2.append(R2)
                relative_error_list.append(relative_error)
                j += 1

            if early_stopping:
                f2.write('{}\t\t{:.8f}\t\t{:.8f}\t\t{}\n'.format(i, reward, R2, eq))
                break

            # a batch expressions
            ## reward
            rewards = torch.tensor(rewards, device=self.device)
            best_epoch_expression = expressions[np.argmax(rewards.cpu())]
            epoch_best_expressions.append(best_epoch_expression)
            epoch_best_rewards.append(max(rewards).item())
            epoch_mean_rewards.append(torch.mean(rewards).item())

            ## R2
            r2 = torch.tensor(r2, device=self.device)
            best_r2_expression = expressions[np.argmax(r2.cpu())]
            epoch_best_r2.append(max(r2).item())
            epoch_mean_r2.append(torch.mean(r2).item())

            epoch_best_relative_error.append(relative_error_list[np.argmax(r2.cpu())])

            # log the best expression of a batch
            f2.write(
                '{}\t\t{:.8f}\t\t{:.8f}\t\t{:.8f}\t\t{}\n'.format(i, relative_error_list[np.argmax(r2.cpu())], max(rewards).item(), max(r2).item(),
                                                                  best_r2_expression))

            # save the best expression from the beginning to now
            if max(r2) > best_performance:
                best_performance = max(r2)
                best_expression = best_r2_expression
                best_relative_error = min(epoch_best_relative_error)

            if self.config.risk_seeking:
                threshold = np.quantile(rewards.cpu(), self.config.risk_factor)
            indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] > threshold], device=self.device)
            if len(indices_to_keep) == 0:
                print("Threshold removes all expressions. Terminating.")
                break

            # Select corresponding subset of rewards, log_probabilities, and entropies
            sub_rewards = torch.index_select(rewards, 0, indices_to_keep)
            sub_log_probs = torch.index_select(batch_log_probs, 0, indices_to_keep)
            sub_entropies = torch.index_select(batch_entropies, 0, indices_to_keep)

            # Compute risk seeking and entropy gradient
            risk_seeking_grad = torch.sum((sub_rewards - threshold) * sub_log_probs, dim=0)
            entropy_grad = torch.sum(sub_entropies, dim=0)

            # Mean reduction and clip to limit exploding gradients
            risk_seeking_grad = torch.clip(risk_seeking_grad / (self.config.risk_factor * len(sub_rewards)), min=-1e6, max=1e6)
            entropy_grad = self.config.entropy_weight * torch.clip(entropy_grad / (self.config.risk_factor * len(sub_rewards)), min=-1e6, max=1e6)

            # compute loss and update parameters
            loss = -1 * (risk_seeking_grad + entropy_grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1.close()
        f2.close()

        # save the rewards
        f3 = open(os.path.join(self.results_dir, "reward_{}_{}.txt".format(self.func_name, self.now_time)), 'w')
        for i in range(len(epoch_mean_rewards)):
            f3.write("{} {:.8f}\n".format(i + 1, epoch_mean_rewards[i]))
        f3.close()

        # plot reward curve
        if self.config.plot_reward:
            # plt.plot([i + 1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)  # best reward of full epoch
            plt.plot([i + 1 for i in range(len(epoch_mean_rewards))], epoch_mean_rewards)  # mean reward of full epoch
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.title('Reward over Time ' + self.now_time)
            plt.show()
            plt.savefig(os.path.join(self.results_dir, "reward_{}_{}.png".format(self.func_name, self.now_time)))

        if early_stopping:
            return eq, R2, error, relative_error
        else:
            return best_expression, best_performance.item(), 1 / max(rewards).item() - 1, best_relative_error

    def bfgs(self, eq, X, y, n_restarts):
        variable = self.vars_name

        # Parse the expression and get all the constants
        expr = eq
        c = symbols('c0:10000')  # Suppose we have at most n constants, c0, c1, ..., cn-1
        consts = list(expr.atoms(Float))  # Only floating-point coefficients are counted, not power exponents
        consts_dict = {c[i]: const for i, const in enumerate(consts)}  # map between c_i and unoptimized constants

        for c_i, val in consts_dict.items():
            expr = expr.subs(val, c_i)

        def loss(expr, X):
            diffs = []
            for i in range(X.shape[0]):
                curr_expr = expr
                for idx, j in enumerate(variable):
                    curr_expr = sp.sympify(curr_expr).subs(j, X[i, idx])
                diff = curr_expr - y[i]
                diffs.append(diff)
            return np.mean(np.square(diffs))

        # Lists where all restarted will be appended
        F_loss = []
        RE_list = []
        R2_list = []
        consts_ = []
        funcs = []

        print('Constructing BFGS loss...')
        loss_func = loss(expr, X)

        for i in range(n_restarts):
            x0 = np.array(consts, dtype=float)
            s = list(consts_dict.keys())
            # bfgs optimization
            fun_timed = TimedFun(fun=sp.lambdify(s, loss_func, modules=['numpy']), stop_after=int(1e10))
            if len(x0):
                minimize(fun_timed.fun, x0, method='BFGS')  # check consts interval and if they are int
                consts_.append(fun_timed.x)
            else:
                consts_.append([])

            final = expr
            for i in range(len(s)):
                final = sp.sympify(final).replace(s[i], fun_timed.x[i])

            funcs.append(final)

            values = {x: X[:, idx] for idx, x in enumerate(variable)}
            y_pred = sp.lambdify(variable, final)(**values)
            if isinstance(y_pred, float):
                print('y_pred is float: ', y_pred, type(y_pred))
                R2 = 0.0
                loss_eq = 10000
            else:
                y_pred = torch.where(torch.isinf(y_pred), 10000, y_pred)  # check if there is inf
                y_pred = torch.where(y_pred.clone().detach() > 10000, 10000, y_pred)  # check if there is large number
                R2 = max(0.0, R_Square(y.squeeze(), y_pred))
                loss_eq = torch.mean(torch.square(y.squeeze() - y_pred)).item()
                relative_error = torch.mean(torch.abs((y.squeeze() - y_pred) / y.squeeze())).item()
            R2_list.append(R2)
            F_loss.append(loss_eq)
            RE_list.append(relative_error)
        best_R2_id = np.nanargmax(R2_list)
        best_consts = consts_[best_R2_id]
        best_expr = funcs[best_R2_id]
        best_R2 = R2_list[best_R2_id]
        best_error = F_loss[best_R2_id]
        best_re = RE_list[best_R2_id]

        return {'best expression': best_expr,
                'constants': best_consts,
                'R2': best_R2,
                'error': best_error,
                'relative error': best_re}

    def play_episodes(self):
        ############################### Sample a symbolic network ##############################
        init_state = torch.rand((1, self.input_size), device=self.device, dtype=self.dtype)  # initial the input state

        if self.auto:
            num_layers, num_funcs_layer, action_index, log_probs, entropies = self.agent(
                init_state)  # output the symbolic network structure parameters
            self.n_layers = num_layers
            self.num_func_layer = num_funcs_layer
        else:
            action_index, log_probs, entropies = self.agent(init_state)

        self.funcs_per_layer = {}
        self.funcs_per_layer_name = {}

        for i in range(self.n_layers):
            layer_funcs_list = list()
            layer_funcs_list_name = list()
            for j in range(self.num_func_layer):
                layer_funcs_list.append(self.funcs_avail[action_index[i, j]])
                layer_funcs_list_name.append(self.funcs_avail[action_index[i, j]].name)
            self.funcs_per_layer.update({i + 1: layer_funcs_list})
            self.funcs_per_layer_name.update({i + 1: layer_funcs_list_name})

        # let binary functions follow unary functions
        for layer, funcs in self.funcs_per_layer.items():
            unary_funcs = [func for func in funcs if isinstance(func, BaseFunction)]
            binary_funcs = [func for func in funcs if isinstance(func, BaseFunction2)]
            sorted_funcs = unary_funcs + binary_funcs
            self.funcs_per_layer[layer] = sorted_funcs

        print("Operators of each layer obtained by sampling: ", self.funcs_per_layer_name)

        ############################### Start training ##############################
        error_test, r2_test, eq = self.train(self.config.trials)

        return error_test, r2_test, eq, log_probs, entropies, self.n_layers, self.num_func_layer, self.funcs_per_layer_name

    def train(self, trials=1):
        """Train the network to find a given function"""

        data, target = self.X.to(self.device), self.y.to(self.device)
        test_data, test_target = self.x_test.to(self.device), self.y_test.to(self.device)

        self.x_dim = data.shape[-1]

        self.vars_name = [f'x_{i}' for i in range(1, self.x_dim + 1)]  # Variable names

        width_per_layer = [len(f) for f in self.funcs_per_layer.values()]
        n_double_per_layer = [functions.count_double(f) for f in self.funcs_per_layer.values()]

        if self.auto:
            init_stddev = [self.init_sd_first] + [self.init_sd_middle] * (self.n_layers - 2) + [self.init_sd_last]

        # Arrays to keep track of various quantities as a function of epoch
        loss_list = []  # Total loss (MSE + regularization)
        error_list = []  # MSE
        reg_list = []  # Regularization
        error_test_list = []  # Test error
        r2_test_list = []  # Test R2

        error_test_final = []
        r2_test_final = []
        eq_list = []

        def log_grad_norm(net):
            sqsum = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    sqsum += (p.grad ** 2).sum().item()
            return np.sqrt(sqsum)

        # for trial in range(trials):
        retrain_num = 0
        trial = 0
        while 0 <= trial < trials:
            print("Training on function " + self.func_name + " Trial " + str(trial + 1) + " out of " + str(trials))

            #  reinitialize for each trial
            if self.auto:
                net = SymbolicNet(self.n_layers,
                                  x_dim=self.x_dim,
                                  funcs=self.funcs_per_layer,
                                  initial_weights=None,
                                  init_stddev=init_stddev,
                                  add_bias=self.add_bias).to(self.device)

            else:
                net = SymbolicNet(self.n_layers,
                                  x_dim=self.x_dim,
                                  funcs=self.funcs_per_layer,
                                  initial_weights=[
                                      # kind of a hack for truncated normal
                                      torch.fmod(torch.normal(0, self.init_sd_first, size=(self.x_dim, width_per_layer[0] + n_double_per_layer[0])),
                                                 2),
                                      # binary operator has two inputs
                                      torch.fmod(
                                          torch.normal(0, self.init_sd_middle, size=(width_per_layer[0], width_per_layer[1] + n_double_per_layer[1])),
                                          2),
                                      torch.fmod(
                                          torch.normal(0, self.init_sd_middle, size=(width_per_layer[1], width_per_layer[2] + n_double_per_layer[2])),
                                          2),
                                      torch.fmod(torch.normal(0, self.init_sd_last, size=(width_per_layer[-1], 1)), 2)
                                  ]).to(self.device)

            net.to(self.dtype)

            loss_val = np.nan
            restart_flag = False
            while np.isnan(loss_val):
                # training restarts if gradients blow up
                criterion = nn.MSELoss()
                optimizer = optim.RMSprop(net.parameters(),
                                          lr=self.config.learning_rate2,
                                          alpha=0.9,  # smoothing constant
                                          eps=1e-10,
                                          momentum=0.0,
                                          centered=False)

                # adaptive learning rate
                lmbda = lambda epoch: 0.1
                scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
                # for param_group in optimizer.param_groups:
                #     print("Learning rate: %f" % param_group['lr'])

                # t0 = time.time()

                if self.clip_grad:
                    que = collections.deque()

                net.train()  # Set model to training mode

                # First stage of training, preceded by 0th warmup stage
                for epoch in range(self.config.n_epochs1 + 2000):
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(data)  # forward pass
                    regularization = L12Smooth(a=0.01)
                    mse_loss = criterion(outputs, target)

                    reg_loss = regularization(net.get_weights_tensor())
                    # loss = mse_loss + self.config.reg_weight * reg_loss
                    loss = mse_loss
                    loss.backward()

                    if self.clip_grad:
                        grad_norm = log_grad_norm(net)
                        que.append(grad_norm)
                        if len(que) > self.window_size:
                            que.popleft()
                            clip_threshold = 0.1 * sum(que) / len(que)
                            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=clip_threshold, norm_type=2)
                        else:
                            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm, norm_type=2)

                    optimizer.step()

                    # summary
                    if epoch % self.config.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()

                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)

                        with torch.no_grad():  # test error
                            test_outputs = net(test_data)  # [num_points, 1] as same as test_target
                            if self.reward_type == 'mse':
                                test_loss = F.mse_loss(test_outputs, test_target)
                            elif self.reward_type == 'nrmse':
                                test_loss = nrmse(test_target, test_outputs)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            test_outputs = torch.where(torch.isnan(test_outputs), torch.full_like(test_outputs, 100),
                                                       test_outputs)
                            r2 = R_Square(test_target, test_outputs)
                            r2_test_list.append(r2)

                        if self.config.verbose:
                            print("Epoch: {}\tTotal training loss: {}\tTest {}: {}".format(epoch, loss_val, self.reward_type, error_test_val))

                        if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                            restart_flag = True
                            break

                    if epoch == 2000:
                        scheduler.step()  # lr /= 10

                if restart_flag:
                    break

                scheduler.step()  # lr /= 10 again

                for epoch in range(self.config.n_epochs2):
                    optimizer.zero_grad()  # zero the parameter gradients
                    outputs = net(data)
                    regularization = L12Smooth(a=0.01)
                    mse_loss = criterion(outputs, target)
                    reg_loss = regularization(net.get_weights_tensor())
                    loss = mse_loss + self.config.reg_weight * reg_loss
                    loss.backward()

                    if self.clip_grad:
                        grad_norm = log_grad_norm(net)
                        que.append(grad_norm)
                        if len(que) > self.window_size:
                            que.popleft()
                            clip_threshold = 0.1 * sum(que) / len(que)
                            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=clip_threshold, norm_type=2)
                        else:
                            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm, norm_type=2)

                    optimizer.step()

                    if epoch % self.config.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)

                        with torch.no_grad():  # test error
                            test_outputs = net(test_data)
                            if self.reward_type == 'mse':
                                test_loss = F.mse_loss(test_outputs, test_target)
                            elif self.reward_type == 'nrmse':
                                test_loss = nrmse(test_target, test_outputs)
                            error_test_val = test_loss.item()
                            error_test_list.append(error_test_val)
                            test_outputs = torch.where(torch.isnan(test_outputs), torch.full_like(test_outputs, 100),
                                                       test_outputs)
                            r2 = R_Square(test_target, test_outputs)
                            r2_test_list.append(r2)
                        if self.config.verbose:
                            print("Epoch: {}\tTotal training loss: {}\tTest {}: {}".format(epoch, loss_val, self.reward_type, error_test_val))

                        if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                            break

                # t1 = time.time()

            if restart_flag:
                # self.play_episodes()
                retrain_num += 1
                if retrain_num == 5:  # only allow 5 restarts
                    return 10000, None, None
                continue

            # After the training, the symbolic network was transformed into an expression by pruning
            with torch.no_grad():
                weights = net.get_weights()
                if self.add_bias:
                    biases = net.get_biases()
                else:
                    biases = None
                expr = pretty_print.network(weights, self.funcs_per_layer, self.vars_name, self.threshold, self.add_bias, biases)

            # results of training trials
            error_test_final.append(error_test_list[-1])
            r2_test_final.append(r2_test_list[-1])
            eq_list.append(expr)

            trial += 1

        error_expr_sorted = sorted(zip(error_test_final, r2_test_final, eq_list), key=lambda x: x[0])  # List of (error, r2, expr)
        print('error_expr_sorted', error_expr_sorted)

        return error_expr_sorted[0]

    def load_data(self, path):
        '''Need to be implemented for specfic file'''
        pass


if __name__ == "__main__":
    config = Params()
    policy_gradient = PolicyGradient(config=config, func="x_1 ** 2 + x_2 ** 2", func_name="x_1**2+x_2**2")
    eq, R2, error, relative_error = policy_gradient.solve_environment()
    print('eq: ', eq)
    print('R2: ', R2)
    print('error: ', error)
    print('relative_error: ', relative_error)
    print('log(1 + MSE): ', np.log(1 + error))  # Grammar VAE

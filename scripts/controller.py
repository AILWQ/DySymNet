import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.functional import one_hot, log_softmax


class Agent(nn.Module):

    def __init__(self, auto, input_size, hidden_size, num_funcs_avail, n_layers, num_funcs_layer, device=None, dtype=torch.float32):
        super(Agent, self).__init__()
        self.auto = auto
        self.num_funcs_avail = num_funcs_avail  # Optional operator category per layer
        self.n_layers = n_layers  # Optional number of layers
        self.num_funcs_layer = num_funcs_layer  # Optional number of operators per layer
        self.dtype = dtype

        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'

        if self.auto:
            self.n_layer_decoder = nn.Linear(hidden_size, len(self.n_layers), device=device)
            self.num_funcs_layer_decoder = nn.Linear(hidden_size, len(self.num_funcs_layer), device=device)
            self.max_input_size = max(len(self.n_layers), len(self.num_funcs_layer))
            self.dynamic_lstm_cell = nn.LSTMCell(self.max_input_size, hidden_size, device=device)
            self.embedding = nn.Linear(self.num_funcs_avail, len(self.num_funcs_layer), device=device)

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, device=device)
        self.decoder = nn.Linear(hidden_size, self.num_funcs_avail, device=device)  # output probability distribution
        self.n_steps = n_layers
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=self.dtype, device=self.device)  # [batch_size, hidden_size]
        c_t = torch.zeros(1, self.hidden_size, dtype=self.dtype, device=self.device)  # [batch_size, hidden_size]

        return h_t, c_t

    def forward(self, input):

        if self.auto:
            if input.shape[-1] < self.max_input_size:
                input = nn.functional.pad(input, (0, self.max_input_size - input.shape[0]), 'constant', 0)

            assert input.shape[-1] == self.max_input_size, 'Error: the input dim of the first step is not equal to the max dim'

            h_t, c_t = self.hidden

            # Sample the number of layers first
            h_t, c_t = self.dynamic_lstm_cell(input, (h_t, c_t))  # [batch_size, hidden_size]
            n_layer_logits = self.n_layer_decoder(h_t)  # [batch_size, len(n_layers)]
            n_layer_probs = F.softmax(n_layer_logits, dim=-1)
            dist = Categorical(probs=n_layer_probs)
            action_index1 = dist.sample()
            log_prob1 = dist.log_prob(action_index1)
            entropy1 = dist.entropy()
            num_layers = self.n_layers[action_index1]

            # Sample the number of operators per layer
            input = n_layer_logits
            if input.shape[-1] < self.max_input_size:
                input = nn.functional.pad(input, (0, self.max_input_size - input.shape[-1]), 'constant', 0)
            h_t, c_t = self.dynamic_lstm_cell(input, (h_t, c_t))
            n_funcs_layer_logits = self.num_funcs_layer_decoder(h_t)  # [batch_size, len(num_funcs_layer)]
            n_funcs_layer_probs = F.softmax(n_funcs_layer_logits, dim=-1)
            dist = Categorical(probs=n_funcs_layer_probs)
            action_index2 = dist.sample()
            log_prob2 = dist.log_prob(action_index2)
            entropy2 = dist.entropy()
            num_funcs_layer = self.num_funcs_layer[action_index2]

            # Sample the operators
            input = n_funcs_layer_logits
            if input.shape[-1] < self.max_input_size:
                input = nn.functional.pad(input, (0, self.max_input_size - input.shape[0]), 'constant', 0)

            outputs = []
            for t in range(num_layers):
                h_t, c_t = self.dynamic_lstm_cell(input, (h_t, c_t))
                output = self.decoder(h_t)  # [batch_size, len(func_avail)]
                outputs.append(output)
                input = self.embedding(output)

            outputs = torch.stack(outputs).squeeze(1)  # [n_layers, len(funcs)]
            probs = F.softmax(outputs, dim=-1)
            dist = Categorical(probs=probs)
            action_index3 = dist.sample((num_funcs_layer,)).transpose(0, 1)  # [num_layers, num_func_layer]
            # print("action_index: ", action_index)
            log_probs = dist.log_prob(action_index3.transpose(0, 1)).transpose(0, 1)  # [num_layers, num_func_layer] compute the log probability of the sampled action
            entropies = dist.entropy()  # [num_layers] compute the entropy of the action distribution
            log_probs, entropies = torch.sum(log_probs), torch.sum(entropies)

            # another way to sample
            # probs = F.softmax(episode_logits, dim=-1)
            # action_index = torch.multinomial(probs, self.num_func_layer, replacement=True)

            # mask = one_hot(action_index, num_classes=self.input_size).squeeze(1)
            # log_probs = log_softmax(episode_logits, dim=-1)
            # episode_log_probs = torch.sum(mask.float() * log_probs)

            log_probs = log_probs + log_prob1 + log_prob2
            entropies = entropies + entropy1 + entropy2

            return num_layers, num_funcs_layer, action_index3, log_probs, entropies

        # Fix the number of layers and the number of operators per layer, only sample the operators, each layer is different
        else:
            outputs = []
            h_t, c_t = self.hidden

            for i in range(self.n_steps):
                h_t, c_t = self.lstm_cell(input, (h_t, c_t))
                output = self.decoder(h_t)  # [batch_size, num_choices]
                outputs.append(output)
                input = output

            outputs = torch.stack(outputs).squeeze(1)  # [num_steps, num_choices]
            probs = F.softmax(outputs, dim=-1)
            dist = Categorical(probs=probs)
            action_index = dist.sample((self.num_funcs_layer,)).transpose(0, 1)  # [num_layers, num_func_layer]
            # print("action_index: ", action_index)
            log_probs = dist.log_prob(action_index.transpose(0, 1)).transpose(0, 1)  # [num_layers, num_func_layer]
            entropies = dist.entropy()  # [num_layers]
            log_probs, entropies = torch.sum(log_probs), torch.sum(entropies)
            return action_index, log_probs, entropies

# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: Luo Xianzhen
#
# The code is based on the davidmrau's implementation:
# https://github.com/davidmrau/mixture-of-experts

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F


def get_norm_and_cos(a, b):
    if a.size(-1) != b.size(0):
        raise ValueError("Can't calculate!")
    a_norm = torch.norm(a, p=2, dim=-1, keepdim=True)  # [batch*mol*1]
    b_norm = torch.norm(b, p=2, dim=0, keepdim=True)  # [1*experts]
    ab = a @ b  # [batch*mol*experts]
    ab_norm = a_norm @ b_norm  # [batch*mol*experts]
    cos = ab / ab_norm  # [batch*mol*experts]
    a_output = a_norm.squeeze(-1).squeeze(-1).tolist()
    b_ouput = b_norm.squeeze(0).tolist()
    cos_ouput = cos.squeeze(1).tolist()
    logits_ouput = ab.squeeze(1).tolist()
    p = lambda x: [round(i, 2) for i in x]
    pp = lambda x: [p(i) for i in x]
    print("input=", p(a_output))
    print("gate=", p(b_ouput))
    print("cos=", pp(cos_ouput))
    print("logits=", pp(logits_ouput))


def print_gate(x):
    p = lambda x: [round(i, 2) for i in x]
    pp = lambda x: [p(i) for i in x]
    print(pp(x.tolist()))


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        out = self.fc2(hidden)
        return out


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=False, k=4, loss_coef=1e-2, dropout=0.1):
        super(MoE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.loss_coef = loss_coef
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        self.experts = nn.ModuleList([MLP(self.input_size, self.hidden_size, self.output_size) for i in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True)
        nn.init.xavier_uniform_(self.w_gate, gain=1.)

        # 激活函数，relu的平滑版本
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        assert (self.k <= self.num_experts)

    def forward(self, x, mask, train=True, _loss=True, _print=False):
        """Args:
        x: [batch_size, ****, input_size]
        train: a boolean scalar.


        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        # if _print:
        # get_norm_and_cos(x, self.w_gate)
        # print_gate(self.w_gate)

        logits = x @ self.w_gate
        top_k_logits, top_k_indices = logits.topk(min(self.k, self.num_experts), dim=-1)
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).to(self.device)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates).to(self.device)

        mask = mask.long().unsqueeze(-1).expand(mask.shape[0], mask.shape[1], gates.shape[-1])
        gates = torch.mul(gates, mask)

        if _loss:
            importance = gates.reshape(-1, self.num_experts).sum(0).float()
            eps = 1e-10
            moe_loss = importance.var() / (importance.mean()**2 + eps)
        else:
            moe_loss = 0

        moe_loss = moe_loss * self.loss_coef

        experts_index = torch.nonzero(gates, as_tuple=True)  # [num_used_experts, len(gates.shape)]
        experts_gate = gates[experts_index]  # [num_used_experts]
        experts_batch_from, experts_atom_from, experts_used = experts_index  # [num_used_experts]
        # experts_from = torch.stack(experts_index[:-1])  # [len(gates.shape), num_used_experts]
        experts_used, index_sorted_experts = experts_used.sort(stable=True)  # [num of used experts]
        experts_batch_from = experts_batch_from[index_sorted_experts]
        experts_atom_from = experts_atom_from[index_sorted_experts]
        # experts_from = list(experts_from[..., index_sorted_experts])  # [tensor(num_used_experts), tensor(num_used_experts)]
        experts_gate = experts_gate[index_sorted_experts]  # [nus]
        experts_count = list(gates.reshape(-1, self.num_experts).count_nonzero(0))  # [num_epxerts]
        if _print:
            print("train={},experts_count={}".format(train, [i.tolist() for i in experts_count]))
            print("moe_loss=", moe_loss)
            print("\n")
        experts_input = x[experts_batch_from, experts_atom_from]  # [nus, input_size]
        experts_input = torch.split(experts_input, experts_count, 0)
        experts_output = [self.experts[i](experts_input[i]) for i in range(self.num_experts)]
        experts_output = torch.cat(experts_output)  # [nus, output_size]
        experts_output = experts_output * experts_gate.unsqueeze(-1)  # [nus, output_size]
        # zeros = torch.zeros(x.shape[0], x.shape[1], self.output_size).cpu()  # [batch_size, ..., output_size]
        # zeros[experts_from] += experts_output.cpu()
        zeros = torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)  # [batch_size, ..., output_size]
        zeros[experts_batch_from, experts_atom_from] += experts_output
        zeros = self.dropout(self.sigmod(zeros)) + x
        return zeros, moe_loss

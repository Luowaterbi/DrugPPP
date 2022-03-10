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
        # print("noisy_gating=", noisy_gating)
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.loss_coef = loss_coef
        # instantiate experts
        # 就是普通的MLP
        self.experts = nn.ModuleList([MLP(self.input_size, self.hidden_size, self.output_size) for i in range(self.num_experts)])
        # 都初始化为0
        if self.noisy_gating:
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        else:
            self.w_gate = nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True)
            nn.init.xavier_uniform_(self.w_gate, gain=1.)

        # 激活函数，relu的平滑版本
        self.softplus = nn.Softplus()
        # 在dim=1维做softmax
        self.softmax = nn.Softmax(-1)
        self.sigmod = nn.Sigmoid()
        # 从均值为0，方差为1的离散正态分布随机采样
        self.normal = Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device))
        self.dropout = nn.Dropout(dropout)

        assert (self.k <= self.num_experts)

    def clean(self, features, mask):
        mask = mask.long().unsqueeze(-1).expand(mask.shape[0], mask.shape[1], features.shape[-1])
        return torch.mul(features, mask)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        x = x[torch.nonzero(x, as_tuple=True)]
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).reshape(-1, self.num_experts).sum(0)

    # 计算所有expert的prob
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        # 每个batch权值第k+1大expert的权值
        threshold_if_in = noisy_top_values.narrow(-1, -1, 1)
        # 与k+1权值比大小，每个batch权值前k大的experts True，其他False
        is_in = torch.gt(noisy_values, threshold_if_in)
        # 每个batch第k大的权值
        threshold_if_out = noisy_top_values.narrow(-1, -2, 1)
        # is each value currently in the top k.
        # cdf概率分布函数
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        # 让 top 1～k 跟 top k+1 的差距大，top k+1～num_experts 跟 top k 的差距大
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    # 只有在train的时候才加噪声，类似于dropout
    # 计算所有experts的权值和概率
    def noisy_top_k_gating(self, x, train, mask, _print, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # @ 就是矩阵相乘 ，logit就是权值
        # 输出模长和余弦相似度
        # if _print:
        #     get_norm_and_cos(x, self.w_gate)
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            # softplus激活函数，*train是说当不train的时候，不加noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            # 计算噪声，stddev只是随机数的权值
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(self.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        # k是每次用几个，num_experts是一共有几个
        # softmax之后的叫gates，之前的叫logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[..., :self.k]
        top_k_indices = top_indices[..., :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True).to(self.device)
        # 只有对应experts有权值，其他experts对应位置权值为0
        # 因为topk返回的是k个值和k个坐标，用这k个值构建gates矩阵，就要按坐标填回去，所以用scatter
        # 所以scatter与topk应该可以说配套使用了，在哪一维用的topk，scatter的时候就设置这一维
        gates = zeros.scatter(-1, top_k_indices, top_k_gates).to(self.device)

        # print("noisy_gating=",self.noisy_gating)
        # print("k=",self.k)
        # print("num_experts=",self.num_experts)
        if self.noisy_gating and self.k < self.num_experts and train:
            # clean logits是 *没加噪声的* *所有* experts权值
            # noisy logits是加噪声的所有experts的权值
            # noisy stddev是噪声的权值
            # top logit是前 *k+1* 个experts的权值
            prob = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)
            prob = self.clean(prob, mask)
            load = prob.sum(-2).sum(0)
        else:
            load = self._gates_to_load(gates)
        return self.clean(gates, mask), load

    def forward(self, x, mask, train=True, _print=False):
        """Args:
        x: [batch_size, ****, input_size]
        train: a boolean scalar.


        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if len(x.size()) < 3:
            x = torch.unsqueeze(x, 1)
        gates, load = self.noisy_top_k_gating(x, train, mask, _print)
        # calculate importance loss
        if self.noisy_gating:
            importance = gates.reshape(-1, self.num_experts).sum(-2)
            print("importance_loss=", self.cv_squared(importance))
            loss = self.cv_squared(load) + self.cv_squared(importance)
            loss *= self.loss_coef
        else:
            loss = 0

        experts_index = torch.nonzero(gates, as_tuple=True)  # [num_used_experts, len(gates.shape)]
        experts_gate = gates[experts_index]  # [num_used_experts]
        experts_batch_from, experts_atom_from, experts_used = experts_index  # [num_used_experts]
        experts_used, index_sorted_experts = experts_used.sort(stable=True)  # [num of used experts]
        experts_batch_from = experts_batch_from[index_sorted_experts]
        experts_atom_from = experts_atom_from[index_sorted_experts]
        experts_gate = experts_gate[index_sorted_experts]  # [nus]
        experts_count = list(gates.reshape(-1, self.num_experts).count_nonzero(0))  # [num_epxerts]
        if _print:
            print("train={},experts_count={}\n".format(train, [i.tolist() for i in experts_count]))
        experts_input = x[experts_batch_from, experts_atom_from]  # [nus, input_size]
        experts_input = torch.split(experts_input, experts_count, 0)
        experts_output = [self.experts[i](experts_input[i]) for i in range(self.num_experts)]
        experts_output = torch.cat(experts_output)  # [nus, output_size]
        experts_output = experts_output * experts_gate.unsqueeze(-1)  # [nus, output_size]
        zeros = torch.zeros(x.shape[0], x.shape[1], self.output_size, requires_grad=True).to(self.device)  # [batch_size, ..., output_size]
        zeros[experts_batch_from, experts_atom_from] += experts_output
        zeros = self.dropout(zeros) + x
        return zeros, loss

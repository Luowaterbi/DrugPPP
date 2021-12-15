import torch
from torch import nn
from MoE_mine import MoE

moe = MoE(input_size=2, output_size=2, num_experts=8, hidden_size=4 * 2, k=3)

inputs = torch.randn(2, 4, 2)
print("inputs=", inputs)
out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)

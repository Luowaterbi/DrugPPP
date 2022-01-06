import torch
from MoE_mine import MoE

moe = MoE(input_size=4, output_size=4, num_experts=4, hidden_size=4 * 2, k=2)

inputs = torch.randn(2, 3, 4)
mask = torch.ones(2, 3, dtype=int)
# inputs = torch.cat([a, a + 1, a + 2], 0)
print("inputs=", inputs)
out, aux_loss = moe(inputs, mask, train=False)

import torch
from MoE import MoE

moe = MoE(input_size=4, output_size=4, num_experts=4, hidden_size=4 * 2, k=2)

inputs = torch.randn(2, 4, 2, 4)
# inputs = torch.cat([a, a + 1, a + 2], 0)
print("inputs=", inputs)
out, aux_loss = moe(inputs, 6)

import torch
from Project.DrugPP.model.MoE_mine import MoE

moe = MoE(input_size=2, output_size=2, num_experts=4, hidden_size=4 * 2, k=2)

inputs = torch.randn(2, 3, 2)
print("inputs=", inputs)
out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)

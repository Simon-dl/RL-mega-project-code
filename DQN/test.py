
# Scratchpad

import torch
pred_frames = torch.tensor([.5,.3,.8,.5],dtype=torch.float)
pred_Q_value = torch.max(pred_frames).item()
print(pred_Q_value)
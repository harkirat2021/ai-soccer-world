"""
Models for soccer agents which maps state to action.
"""

import torch.nn as nn
import torch.nn.functional as F

"""
Soccer Agent Model 0

input - 1D tensor
output - tensor with 2 elements representing velocity vector.
"""
class SAM0(nn.Module):
    def __init__(self, input_dim):
      super(SAM0, self).__init__()
      self.fc1 = nn.Linear(input_dim, 32)
      self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
      h = F.relu(self.fc1(x))
      y = F.tanh(self.fc2(h))
      return y
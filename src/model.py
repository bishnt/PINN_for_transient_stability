import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class PINN(nn.Module):
  def __init__(
      self,
      n_hidden_layers: int = 6,
      n_neurons: int = 64,
      activation: str = 'tanh'
  ):
    super().__init__()
    
    self.register_buffer('t_scale',    torch.tensor(1.0))
    self.register_buffer('delta_mean', torch.tensor(0.0))
    self.register_buffer('delta_std',  torch.tensor(1.0))
    self.register_buffer('omega_mean', torch.tensor(0.0))
    self.register_buffer('omega_std',  torch.tensor(1.0))

    layers =[]

    #input layer
    layers.append(nn.Linear(1, n_neurons))
    layers.append(self._get_activation(activation))

    #hidden layers
    for _ in range(n_hidden_layers - 1):
      layers.append(nn.Linear(n_neurons, n_neurons))
      layers.append(self._get_activation(activation))

    #output layer
    layers.append(nn.Linear(n_neurons, 2))

    self.network = nn.Sequential(*layers)

    #Xavier initialization for stable training
    self._initialize_weights()

  def _get_activation(self, name: str) -> nn.Module:
    if name == 'relu':
      return nn.ReLU()
    elif name == 'tanh':
      return nn.Tanh()
    elif name == 'sigmoid':
      return nn.Sigmoid()

  def _initialize_weights(self):
    #Xavier uniform initialization for all linear layers.
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
        
        
  def set_normalizer(
        self,
        t_scale: float,
        delta_mean: float,
        delta_std: float,
        omega_mean: float,
        omega_std: float,
    ):
        self.t_scale.fill_(t_scale)
        self.delta_mean.fill_(delta_mean)
        self.delta_std.fill_(delta_std)
        self.omega_mean.fill_(omega_mean)
        self.omega_std.fill_(omega_std)
 
  def _normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.t_scale
 
  def _denormalize_delta(self, d_norm: torch.Tensor) -> torch.Tensor:
        return d_norm * self.delta_std + self.delta_mean
 
  def _denormalize_omega(self, w_norm: torch.Tensor) -> torch.Tensor:
        return w_norm * self.omega_std + self.omega_mean

  def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = self._normalize_t(t)           # map to [0, 1]
        out_norm = self.network(t_norm)          # network output in normalised space
        delta = self._denormalize_delta(out_norm[:, 0:1])
        omega = self._denormalize_omega(out_norm[:, 1:2])
        return torch.cat([delta, omega], dim=1)
      
  def predict_delta_omega(
    self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #Return delta and omega as separate tensors.
    out = self.forward(t)
    return out[:, 0:1], out[:, 1:2]

def count_parameters(model: nn.Module) -> int:
  #Count trainable parameters.
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
  model = PINN(n_hidden_layers=4, n_neurons=64)
  print(f'Parameters: {count_parameters(model):,}')
  t_test = torch.linspace(0, 2, 100).reshape(-1, 1)
  out = model(t_test)
  print(f'Output shape: {out.shape}') #[100, 2]


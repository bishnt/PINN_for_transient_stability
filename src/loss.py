
import torch
import torch.nn as nn
from typing import Dict, Tuple
from swing_equation import SMIBParameters

class PINNLoss:
    def __init__(
        self,
        params: SMIBParameters,
        lambda_phys: float = 1.0,
        lambda_data: float = 10.0,
        lambda_ic: float = 100.0,
        fault_factor: float = 1.0
    ):
        self.p = params
        self.lambda_phys = lambda_phys
        self.lambda_data = lambda_data
        self.lambda_ic = lambda_ic
        self.fault_factor = fault_factor
        self.mse = nn.MSELoss()

    def compute_physics_residual(
        self,
        model,
        t_colloc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = model(t_colloc)
        delta = out[:, 0:1]
        omega = out[:, 1:2]

        ones = torch.ones_like(delta)
        d_delta_dt = torch.autograd.grad(
            delta, t_colloc,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]
        d_omega_dt = torch.autograd.grad(
            omega, t_colloc,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True
        )[0]

        omega_dev = omega - self.p.omega0
        residual_delta = d_delta_dt - omega_dev

        Pe = self.fault_factor * self.p.Pmax * torch.sin(delta)
        residual_omega = d_omega_dt - (
            (self.p.omega0 / (2 * self.p.H)) *
            (self.p.Pm - Pe - self.p.D * omega_dev)
        )
        omega0 = self.p.omega0
        accel_scale = omega0 / (2.0 * self.p.H) 
        return residual_delta / omega0, residual_omega / accel_scale

    def __call__(
        self,
        model,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        #physics_loss
        res_d, res_w = self.compute_physics_residual(model, batch['t_colloc'])
        loss_phys = (
            self.mse(res_d, torch.zeros_like(res_d)) +
            self.mse(res_w, torch.zeros_like(res_w))
        )
        #data_loss
        pred_data = model(batch['t_data'])
        delta_pred  = pred_data[:, 0:1]
        omega_pred  = pred_data[:, 1:2]
 
        delta_std = model.delta_std
        omega_std = model.omega_std
 
        loss_data = (
            self.mse(delta_pred / delta_std, batch['delta_data'] / delta_std) +
            self.mse(omega_pred / omega_std, batch['omega_data'] / omega_std)
        )

        pred_ic = model(batch['t_ic'])
        loss_ic = (
            self.mse(pred_ic[:, 0:1], batch['delta_ic']) +
            self.mse(pred_ic[:, 1:2], batch['omega_ic'])
        )

        total = (self.lambda_phys * loss_phys +
                 self.lambda_data * loss_data +
                 self.lambda_ic * loss_ic)

        return {
            'total': total,
            'physics': loss_phys.detach(),
            'data': loss_data.detach(),
            'ic': loss_ic.detach(),
        }
"""
Multi-machine extension for power system transient stability.
Reuses existing modules: swing_equation, model, loss, data_generator patterns.
"""
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from swing_equation import SwingEquationSolver
from model import PINN


@dataclass
class MachineParams:
    """Parameters for a single synchronous machine (extends SMIBParameters pattern)."""
    H: float  # Inertia constant
    D: float  # Damping coefficient
    Pm: float  # Mechanical power input (pu)
    E: float  # Internal voltage magnitude (pu)


class MultiMachineSolver:
    """
    Multi-machine swing equation solver.
    Extends SwingEquationSolver pattern for coupled machines.
    """
    def __init__(
        self,
        machine_params: List[MachineParams],
        B_matrix: np.ndarray,  # Susceptance matrix (N x N)
        omega0: float = 2 * np.pi * 60
    ):
        self.machine_params = machine_params
        self.B_matrix = B_matrix
        self.omega0 = omega0
        self.n_machines = len(machine_params)

    def _electrical_powers(self, deltas: np.ndarray) -> np.ndarray:
        """Compute electrical power for each machine."""
        Pe = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            for j in range(self.n_machines):
                if i != j:
                    Pe[i] += (self.machine_params[i].E * self.machine_params[j].E * 
                             self.B_matrix[i, j] * np.sin(deltas[i] - deltas[j]))
        return Pe

    def swing_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Right-hand side of coupled swing equations."""
        # State: [delta1, omega1, delta2, omega2, ..., deltaN, omegaN]
        deltas = state[0::2]
        omegas = state[1::2]
        
        Pe = self._electrical_powers(deltas)
        
        derivatives = np.zeros_like(state)
        for i in range(self.n_machines):
            # d(delta_i)/dt = omega_i - omega0
            derivatives[2*i] = omegas[i] - self.omega0
            # d(omega_i)/dt = (omega0 / 2H) * (Pm - Pe - D*(omega_i - omega0))
            derivatives[2*i+1] = (self.omega0 / (2 * self.machine_params[i].H)) * (
                self.machine_params[i].Pm - Pe[i] - 
                self.machine_params[i].D * (omegas[i] - self.omega0))
        
        return derivatives

    def rk4_step(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        """Single RK4 integration step (reuse from swing_equation)."""
        k1 = self.swing_rhs(t, state)
        k2 = self.swing_rhs(t + dt / 2, state + dt * k1 / 2)
        k3 = self.swing_rhs(t + dt / 2, state + dt * k2 / 2)
        k4 = self.swing_rhs(t + dt, state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(
        self,
        t_span: Tuple[float, float],
        dt: float = 0.001,
        initial_state: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Simulate multi-machine system using RK4."""
        t0, tf = t_span
        n = int((tf - t0) / dt)
        
        if initial_state is None:
            # Default: small random angles, synchronous speed
            initial_state = []
            for i in range(self.n_machines):
                initial_state.extend([0.3 + 0.1 * i, self.omega0])
            initial_state = np.array(initial_state)
        
        states = np.zeros((n + 1, 2 * self.n_machines))
        times = np.zeros(n + 1)
        states[0] = initial_state
        
        for i in range(n):
            t_curr = t0 + i * dt
            states[i + 1] = self.rk4_step(t_curr, states[i], dt)
            times[i + 1] = t_curr + dt
        
        # Package results
        result = {'t': times}
        for i in range(self.n_machines):
            result[f'delta{i+1}'] = states[:, 2*i]
            result[f'omega{i+1}'] = states[:, 2*i+1]
        
        return result


class MultiMachinePINN(PINN):
    """
    Multi-machine PINN extending the base PINN class.
    Output: 2N variables (delta, omega per machine).
    """
    def __init__(
        self,
        n_machines: int = 2,
        n_hidden_layers: int = 6,
        n_neurons: int = 64,
        activation: str = 'tanh'
    ):
        # Initialize base PINN with custom output dimension
        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_neurons=n_neurons,
            activation=activation
        )
        self.n_machines = n_machines
        
        # Replace output layer for 2N outputs
        self.network = nn.Sequential(*list(self.network.children())[:-1])
        self.network.add_module('output', nn.Linear(n_neurons, 2 * n_machines))
        
        # Re-initialize weights
        self._initialize_weights()
        
        # Add normalization buffers for each machine
        for i in range(n_machines):
            if not hasattr(self, f'delta_mean_{i}'):
                self.register_buffer(f'delta_mean_{i}', torch.tensor(0.0))
                self.register_buffer(f'delta_std_{i}', torch.tensor(1.0))
                self.register_buffer(f'omega_mean_{i}', torch.tensor(0.0))
                self.register_buffer(f'omega_std_{i}', torch.tensor(1.0))

    def set_normalizer(
        self,
        t_scale: float,
        delta_means: List[float],
        delta_stds: List[float],
        omega_means: List[float],
        omega_stds: List[float]
    ):
        """Set normalization parameters for all machines."""
        self.t_scale.fill_(t_scale)
        for i in range(self.n_machines):
            getattr(self, f'delta_mean_{i}').fill_(delta_means[i])
            getattr(self, f'delta_std_{i}').fill_(delta_stds[i])
            getattr(self, f'omega_mean_{i}').fill_(omega_means[i])
            getattr(self, f'omega_std_{i}').fill_(omega_stds[i])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = self._normalize_t(t)
        out_norm = self.network(t_norm)
        
        # Split and denormalize for each machine
        outputs = []
        for i in range(self.n_machines):
            delta_norm = out_norm[:, 2*i:2*i+1]
            omega_norm = out_norm[:, 2*i+1:2*i+2]
            
            delta = self._denormalize_delta(delta_norm)
            omega = self._denormalize_omega(omega_norm)
            
            # Apply machine-specific normalization
            delta = delta * getattr(self, f'delta_std_{i}') + getattr(self, f'delta_mean_{i}')
            omega = omega * getattr(self, f'omega_std_{i}') + getattr(self, f'omega_mean_{i}')
            
            outputs.append(delta)
            outputs.append(omega)
        
        return torch.cat(outputs, dim=1)


class MultiMachineLoss:
    """
    Loss function for multi-machine PINN.
    Extends PINNLoss pattern for coupled machines.
    """
    def __init__(
        self,
        machine_params: List[MachineParams],
        B_matrix: np.ndarray,
        omega0: float = 2 * np.pi * 60,
        lambda_phys: float = 1.0,
        lambda_data: float = 10.0,
        lambda_ic: float = 100.0
    ):
        self.machine_params = machine_params
        self.B_matrix = torch.FloatTensor(B_matrix)
        self.omega0 = omega0
        self.lambda_phys = lambda_phys
        self.lambda_data = lambda_data
        self.lambda_ic = lambda_ic
        self.n_machines = len(machine_params)
        self.mse = nn.MSELoss()

    def _compute_electrical_powers(self, deltas: torch.Tensor) -> torch.Tensor:
        """Compute electrical power for each machine."""
        n = self.n_machines
        E = torch.FloatTensor([p.E for p in self.machine_params]).view(1, n)
        B = self.B_matrix
        
        Pe = torch.zeros_like(deltas)
        for i in range(n):
            for j in range(n):
                if i != j:
                    Pe[:, i:i+1] += E[0, i] * E[0, j] * B[i, j] * torch.sin(deltas[:, i:i+1] - deltas[:, j:j+1])
        return Pe

    def compute_physics_residual(
        self,
        model: nn.Module,
        t_colloc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute physics residual for all machines."""
        out = model(t_colloc)
        
        # Split into deltas and omegas
        deltas = []
        omegas = []
        for i in range(self.n_machines):
            deltas.append(out[:, 2*i:2*i+1])
            omegas.append(out[:, 2*i+1:2*i+2])
        
        deltas = torch.cat(deltas, dim=1)
        omegas = torch.cat(omegas, dim=1)
        
        # Compute time derivatives
        ones = torch.ones_like(deltas)
        d_deltas_dt = []
        d_omegas_dt = []
        
        for i in range(self.n_machines):
            d_delta_dt = torch.autograd.grad(
                deltas[:, i:i+1], t_colloc,
                grad_outputs=ones[:, i:i+1],
                create_graph=True,
                retain_graph=True
            )[0]
            d_omegas_dt.append(torch.autograd.grad(
                omegas[:, i:i+1], t_colloc,
                grad_outputs=ones[:, i:i+1],
                create_graph=True,
                retain_graph=True
            )[0])
            d_deltas_dt.append(d_delta_dt)
        
        d_deltas_dt = torch.cat(d_deltas_dt, dim=1)
        d_omegas_dt = torch.cat(d_omegas_dt, dim=1)
        
        # Compute electrical powers
        Pe = self._compute_electrical_powers(deltas)
        
        # Compute residuals
        residual_delta = d_deltas_dt - (omegas - self.omega0)
        
        residual_omega = torch.zeros_like(d_omegas_dt)
        for i in range(self.n_machines):
            omega_dev = omegas[:, i:i+1] - self.omega0
            residual_omega[:, i:i+1] = d_omegas_dt[:, i:i+1] - (
                (self.omega0 / (2.0 * self.machine_params[i].H)) *
                (self.machine_params[i].Pm - Pe[:, i:i+1] - self.machine_params[i].D * omega_dev)
            )
        
        return residual_delta, residual_omega

    def __call__(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss."""
        t_colloc = batch['t_colloc'].clone().detach().requires_grad_(True)
        
        # Physics loss
        res_d, res_w = self.compute_physics_residual(model, t_colloc)
        loss_phys = self.mse(res_d, torch.zeros_like(res_d)) + self.mse(res_w, torch.zeros_like(res_w))
        
        # Data loss
        pred_data = model(batch['t_data'])
        loss_data = torch.tensor(0.0)
        
        for i in range(self.n_machines):
            delta_pred = pred_data[:, 2*i:2*i+1]
            omega_pred = pred_data[:, 2*i+1:2*i+2]
            delta_data = batch[f'delta{i+1}_data']
            omega_data = batch[f'omega{i+1}_data']
            
            delta_std = getattr(model, f'delta_std_{i}')
            omega_std = getattr(model, f'omega_std_{i}')
            
            loss_data += (
                self.mse(delta_pred / delta_std, delta_data / delta_std) +
                self.mse(omega_pred / omega_std, omega_data / omega_std)
            )
        
        # IC loss
        pred_ic = model(batch['t_ic'])
        loss_ic = torch.tensor(0.0)
        
        for i in range(self.n_machines):
            delta_pred = pred_ic[:, 2*i:2*i+1]
            omega_pred = pred_ic[:, 2*i+1:2*i+2]
            delta_ic = batch[f'delta{i+1}_ic']
            omega_ic = batch[f'omega{i+1}_ic']
            
            delta_std = getattr(model, f'delta_std_{i}')
            omega_std = getattr(model, f'omega_std_{i}')
            
            loss_ic += (
                self.mse(delta_pred / delta_std, delta_ic / delta_std) +
                self.mse(omega_pred / omega_std, omega_ic / omega_std)
            )
        
        total = self.lambda_phys * loss_phys + self.lambda_data * loss_data + self.lambda_ic * loss_ic
        
        return {
            'total': total,
            'physics': loss_phys,
            'data': loss_data,
            'ic': loss_ic
        }


class MultiMachineDataGenerator:
    """Generate training data for multi-machine systems (extends PINNDataGenerator pattern)."""
    def __init__(
        self,
        machine_params: List[MachineParams],
        B_matrix: np.ndarray,
        omega0: float = 2 * np.pi * 60,
        seed: int = 42
    ):
        self.machine_params = machine_params
        self.B_matrix = B_matrix
        self.omega0 = omega0
        self.n_machines = len(machine_params)
        self.solver = MultiMachineSolver(machine_params, B_matrix, omega0)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_reference_trajectory(
        self,
        t_span: Tuple[float, float] = (0.0, 2.0),
        dt: float = 0.001,
        initial_state: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Generate reference trajectory using RK4."""
        return self.solver.simulate(t_span, dt, initial_state)

    def sample_training_data(
        self,
        trajectory: Dict[str, np.ndarray],
        n_collocation: int = 2000,
        n_data_points: int = 50,
        t_total: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """Sample training data from reference trajectory."""
        # Collocation points (Latin Hypercube)
        intervals = np.linspace(0, t_total, n_collocation + 1)
        t_colloc = np.array([
            np.random.uniform(intervals[i], intervals[i+1])
            for i in range(n_collocation)
        ])
        
        # Data points
        idx = np.linspace(0, len(trajectory['t']) - 1, n_data_points, dtype=int)
        t_data = trajectory['t'][idx]
        
        # Initial condition
        t_ic = np.array([0.0])
        
        def to_tensor(arr, grad=False):
            t = torch.FloatTensor(arr).reshape(-1, 1)
            t.requires_grad_(grad)
            return t
        
        batch = {
            't_colloc': to_tensor(t_colloc, grad=True),
            't_data': to_tensor(t_data),
            't_ic': to_tensor(t_ic)
        }
        
        # Add data for each machine
        for i in range(self.n_machines):
            batch[f'delta{i+1}_data'] = to_tensor(trajectory[f'delta{i+1}'][idx])
            batch[f'omega{i+1}_data'] = to_tensor(trajectory[f'omega{i+1}'][idx])
            batch[f'delta{i+1}_ic'] = to_tensor(np.array([trajectory[f'delta{i+1}'][0]]))
            batch[f'omega{i+1}_ic'] = to_tensor(np.array([trajectory[f'omega{i+1}'][0]]))
        
        return batch

    def compute_normalization_params(
        self,
        trajectory: Dict[str, np.ndarray]
    ) -> Tuple[float, List[float], List[float], List[float], List[float]]:
        """Compute normalization parameters from trajectory."""
        t_scale = trajectory['t'][-1]
        
        delta_means = []
        delta_stds = []
        omega_means = []
        omega_stds = []
        
        for i in range(self.n_machines):
            delta_means.append(float(np.mean(trajectory[f'delta{i+1}'])))
            delta_stds.append(float(np.std(trajectory[f'delta{i+1}'])))
            omega_means.append(float(np.mean(trajectory[f'omega{i+1}'])))
            omega_stds.append(float(np.std(trajectory[f'omega{i+1}'])))
        
        return t_scale, delta_means, delta_stds, omega_means, omega_stds


if __name__ == '__main__':
    # Test the multi-machine system
    m1 = MachineParams(H=5.0, D=0.05, Pm=0.8, E=1.05)
    m2 = MachineParams(H=4.0, D=0.05, Pm=0.6, E=1.02)
    
    B_matrix = np.array([[0.0, 2.0], [2.0, 0.0]])
    
    solver = MultiMachineSolver([m1, m2], B_matrix)
    trajectory = solver.simulate((0.0, 2.0), dt=0.001)
    
    print("Multi-Machine System Simulation:")
    print(f"  Time range: {trajectory['t'][0]:.3f}s - {trajectory['t'][-1]:.3f}s")
    print(f"  Machine 1 delta range: {np.degrees(trajectory['delta1'].min()):.1f}° - {np.degrees(trajectory['delta1'].max()):.1f}°")
    print(f"  Machine 2 delta range: {np.degrees(trajectory['delta2'].min()):.1f}° - {np.degrees(trajectory['delta2'].max()):.1f}°")
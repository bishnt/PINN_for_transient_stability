
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from swing_equation import SMIBParameters, SwingEquationSolver



# 1. Initialize SMIB Parameters
smib_params = SMIBParameters(
    H=5.0,
    D=0.05,
    omega0=2 * np.pi * 60,
    Pm=0.8,
    Pmax=2.1
)

# 2. Instantiate the SwingEquationSolver
solver = SwingEquationSolver(smib_params)

print(f"SMIB Parameters Initialized: {smib_params}")
print(f"Solver Initialized. Equilibrium Delta: {np.degrees(smib_params.delta_eq):.2f} degrees")

import torch

class PINNDataGenerator:
  """
  Generates all required datasets for PINN training.
  Three types of points:
  1. Collocation points: random t values where physics residual is enforced
  2. Data points: sparse measurements from RK4 simulation
  3. IC points: t=0 initial condition enforcement
  """
  def __init__(self, params: SMIBParameters, seed: int = 42):
    self.params = params
    self.solver = SwingEquationSolver(params) # Instantiate solver here
    np.random.seed(seed)
    torch.manual_seed(seed)

  def generate_reference_trajectory(
      self,
      fault_start: float = 0.1,
      fault_end: float = 0.2,
      t_total: float = 2.0,
      dt: float = 0.001
  ) -> dict:
    """Generate high-resolution reference trajectory using RK4."""
    t_span = (0.0, t_total)
    simulation_results = self.solver.simulate(
        t_span=t_span,
        dt=dt,
        delta0=self.params.delta_eq,
        omega_deviated0=0.0, # Starts at synchronous speed, so deviation is 0
        fault_start=fault_start,
        fault_end=fault_end,
        fault_factor_pre=1.0,
        fault_factor_fault=0.0, # Short circuit during fault
        fault_factor_post=1.0
    )

    # Convert omega_deviated to absolute omega (omega = omega_deviated + omega0)
    absolute_omega_trajectory = simulation_results['omega_deviated'] + self.params.omega0

    return {
        't': simulation_results['time'],
        'delta': simulation_results['delta'],
        'omega': absolute_omega_trajectory # Return absolute omega
    }

  def sample_training_data(
      self,
      trajectory: dict,
      n_collocation: int = 2000,
      n_data_points: int = 50,
      t_total: float = 2.0
  ) -> Dict[str, torch.Tensor]:
    """
    Sample training data from the reference trajectory.
    - Collocation points: Latin Hypercube sampling over [0, t_total]
    - Data points: sparse, uniformly sampled from trajectory
    """
    # -- Collocation points (no labels needed, only t) -------------
    # Latin Hypercube Sampling gives better coverage than uniform random
    intervals = np.linspace(0, t_total, n_collocation + 1)
    t_colloc = np.array([
        np.random.uniform(intervals[i], intervals[i+1])
        for i in range(n_collocation)
    ])

    # -- Data points (sparse measurements) -------------------------
    idx = np.linspace(0, len(trajectory['t'])-1, n_data_points, dtype=int)
    t_data = trajectory['t'][idx]
    d_data = trajectory['delta'][idx]
    w_data = trajectory['omega'][idx]

    # -- Initial condition ------------------------------------------
    t_ic = np.array([0.0])
    delta_ic = np.array([self.params.delta_eq])
    omega_ic = np.array([self.params.omega0]) # Absolute omega at t=0

    # Convert to tensors (requires_grad=True for collocation points)
    def to_tensor(arr, grad=False):
      t = torch.FloatTensor(arr).reshape(-1, 1)
      t.requires_grad_(grad)
      return t

    return {
        # Physics enforcement points
        't_colloc': to_tensor(t_colloc, grad=True),
        # Sparse measurement points
        't_data': to_tensor(t_data),
        'delta_data': to_tensor(d_data),
        'omega_data': to_tensor(w_data),
        # Initial condition point
        't_ic': to_tensor(t_ic),
        'delta_ic': to_tensor(delta_ic),
        'omega_ic': to_tensor(omega_ic),
    }

  def generate_test_grid(
      self, t_total: float = 2.0, n_points: int = 1000
  ) -> torch.Tensor:
    """Dense time grid for evaluation."""
    t = np.linspace(0, t_total, n_points)
    return torch.FloatTensor(t).reshape(-1, 1)

# 1. Instantiate the PINNDataGenerator
data_generator = PINNDataGenerator(params=smib_params, seed=42)

# Define simulation parameters for the reference trajectory
REF_FAULT_START = 1.0
REF_FAULT_END = 1.2
REF_T_TOTAL = 5.0
REF_DT = 0.0001 # Even finer resolution for reference

print(f"\nGenerating reference trajectory for t_total={REF_T_TOTAL}s, dt={REF_DT}s...")
reference_trajectory = data_generator.generate_reference_trajectory(
    fault_start=REF_FAULT_START,
    fault_end=REF_FAULT_END,
    t_total=REF_T_TOTAL,
    dt=REF_DT
)

print(f"Reference trajectory generated. Time points: {len(reference_trajectory['t'])}")
print(f"Sample (t, delta, omega) from reference:")
for i in [0, len(reference_trajectory['t']) // 2, -1]:
    t_val = reference_trajectory['t'][i]
    delta_val = np.degrees(reference_trajectory['delta'][i])
    omega_val = reference_trajectory['omega'][i]
    print(f"  t={t_val:.3f}s, delta={delta_val:.2f} deg, omega={omega_val:.2f} rad/s")

# 2. Sample Training Data
N_COLLOCATION = 5000
N_DATA_POINTS = 100
TRAINING_T_TOTAL = 5.0

print(f"\nSampling training data: {N_COLLOCATION} collocation, {N_DATA_POINTS} data points...")
training_data = data_generator.sample_training_data(
    trajectory=reference_trajectory,
    n_collocation=N_COLLOCATION,
    n_data_points=N_DATA_POINTS,
    t_total=TRAINING_T_TOTAL
)

print("--- Training Data Shapes ---")
for key, value in training_data.items():
    print(f"{key}: {value.shape} (requires_grad={value.requires_grad})")

print("\n--- Initial Conditions Sample ---")
print(f"t_ic: {training_data['t_ic'].numpy().flatten()}")
print(f"delta_ic: {np.degrees(training_data['delta_ic'].numpy().flatten())[0]:.2f} degrees")
print(f"omega_ic: {training_data['omega_ic'].numpy().flatten()[0]:.2f} rad/s")

print("\n--- Sample Collocation Points ---")
print(f"t_colloc (first 5): {training_data['t_colloc'][:5].detach().numpy().flatten()}")

print("\n--- Sample Data Points (t, delta, omega) ---")
for i in range(5):
    t_val = training_data['t_data'][i].item()
    delta_val = np.degrees(training_data['delta_data'][i].item())
    omega_val = training_data['omega_data'][i].item()
    print(f"  t={t_val:.3f}s, delta={delta_val:.2f} deg, omega={omega_val:.2f} rad/s")

# 3. Generate Test Grid
N_TEST_POINTS = 1000
TEST_T_TOTAL = 5.0

print(f"\nGenerating test grid with {N_TEST_POINTS} points...")
test_grid = data_generator.generate_test_grid(t_total=TEST_T_TOTAL, n_points=N_TEST_POINTS)

print(f"Test grid shape: {test_grid.shape}")
print(f"Test grid (first 5): {test_grid[:5].numpy().flatten()}")
print(f"Test grid (last 5): {test_grid[-5:].numpy().flatten()}")

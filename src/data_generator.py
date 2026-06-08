
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from swing_equation import SMIBParameters, SwingEquationSolver
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
    
    # Keep delta as absolute value - the deviation approach didn't help
    # because delta_dev is still unbounded
    delta_trajectory = simulation_results['delta']

    return {
        't': simulation_results['time'],
        'delta': delta_trajectory,  # Return absolute delta
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
    delta_ic = np.array([self.params.delta_eq])  # Absolute delta at equilibrium
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

  def generate_multiple_trajectories(
      self,
      n_trajectories: int = 10,
      fault_start_range: Tuple[float, float] = (0.05, 0.3),
      fault_duration_range: Tuple[float, float] = (0.05, 0.15),
      t_total: float = 2.0,
      dt: float = 0.001,
      include_unstable: bool = True,
      unstable_fraction: float = 0.2,
      seed: Optional[int] = None
  ) -> List[dict]:
    """
    Generate multiple trajectories with varying fault scenarios.
    
    Args:
        n_trajectories: Number of trajectories to generate
        fault_start_range: (min, max) range for fault start times
        fault_duration_range: (min, max) range for fault durations (for stable cases)
        t_total: Total simulation time
        dt: Time step for simulation
        include_unstable: If True, include some long-duration faults that cause instability
        unstable_fraction: Fraction of trajectories that should be potentially unstable (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        List of trajectory dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = []
    n_unstable = int(n_trajectories * unstable_fraction) if include_unstable else 0
    
    for i in range(n_trajectories):
        # Sample fault start time
        fault_start = np.random.uniform(*fault_start_range)
        
        # Sample fault duration
        if i < n_trajectories - n_unstable:
            # First portion: short faults (stable)
            fault_duration = np.random.uniform(*fault_duration_range)
        else:
            # Last portion: longer faults (potentially unstable)
            fault_duration = np.random.uniform(0.20, 0.35)
        
        fault_end = fault_start + fault_duration
        
        # Vary Pm slightly to get different operating points
        Pm_variation = np.random.uniform(-0.1, 0.1)
        varied_params = SMIBParameters(
            H=self.params.H,
            D=self.params.D,
            omega0=self.params.omega0,
            Pm=self.params.Pm + Pm_variation,
            Pmax=self.params.Pmax
        )
        varied_solver = SwingEquationSolver(varied_params)
        
        # Generate trajectory
        simulation_results = varied_solver.simulate(
            t_span=(0.0, t_total),
            dt=dt,
            delta0=varied_params.delta_eq,
            omega_deviated0=0.0,
            fault_start=fault_start,
            fault_end=fault_end,
            fault_factor_pre=1.0,
            fault_factor_fault=0.0,
            fault_factor_post=1.0
        )
        
        absolute_omega = simulation_results['omega_deviated'] + varied_params.omega0
        delta_trajectory = simulation_results['delta']
        
        trajectories.append({
            't': simulation_results['time'],
            'delta': delta_trajectory,
            'omega': absolute_omega,
            'fault_start': fault_start,
            'fault_end': fault_end,
            'Pm': varied_params.Pm,
            'is_stable': self._check_stability(delta_trajectory, simulation_results['time'])
        })
    
    return trajectories
  
  def _check_stability(self, delta: np.ndarray, time: np.ndarray, 
                       threshold_deg: float = 180.0) -> bool:
    """Check if trajectory remains stable (delta doesn't exceed threshold)."""
    max_delta_deg = np.max(np.abs(np.degrees(delta)))
    
    # A trajectory is unstable if:
    # 1. Delta exceeds threshold (loss of synchronism)
    # 2. Delta keeps growing monotonically at the end (diverging)
    
    if max_delta_deg > threshold_deg:
        return False
    
    # Check if delta keeps growing at the end (unstable)
    if len(time) > 100:
        # Look at last 50 points
        final_segment = delta[-50:]
        # If monotonic increase over 10 degrees in final segment, likely unstable
        delta_change = final_segment[-1] - final_segment[0]
        if delta_change > np.radians(10):  # More than 10 degrees increase
            return False
    
    return True

  def sample_from_multiple_trajectories(
      self,
      trajectories: List[dict],
      n_collocation_per_traj: int = 500,
      n_data_points_per_traj: int = 15,
      t_total: float = 2.0
  ) -> Dict[str, torch.Tensor]:
    """
    Sample training data from multiple trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        n_collocation_per_traj: Number of collocation points per trajectory
        n_data_points_per_traj: Number of data points per trajectory
        t_total: Total time for collocation sampling
    
    Returns:
        Dictionary with concatenated training data from all trajectories
    """
    all_t_colloc = []
    all_t_data = []
    all_delta_data = []
    all_omega_data = []
    
    for traj in trajectories:
        # Collocation points for this trajectory
        intervals = np.linspace(0, t_total, n_collocation_per_traj + 1)
        t_colloc = np.array([
            np.random.uniform(intervals[i], intervals[i+1])
            for i in range(n_collocation_per_traj)
        ])
        all_t_colloc.append(t_colloc)
        
        # Data points from this trajectory
        idx = np.linspace(0, len(traj['t'])-1, n_data_points_per_traj, dtype=int)
        all_t_data.append(traj['t'][idx])
        all_delta_data.append(traj['delta'][idx])
        all_omega_data.append(traj['omega'][idx])
    
    # Concatenate all data
    t_colloc = np.concatenate(all_t_colloc)
    t_data = np.concatenate(all_t_data)
    delta_data = np.concatenate(all_delta_data)
    omega_data = np.concatenate(all_omega_data)
    
    # Initial condition (use equilibrium from base params)
    t_ic = np.array([0.0])
    delta_ic = np.array([self.params.delta_eq])
    omega_ic = np.array([self.params.omega0])
    
    def to_tensor(arr, grad=False):
        t = torch.FloatTensor(arr).reshape(-1, 1)
        t.requires_grad_(grad)
        return t
    
    return {
        't_colloc': to_tensor(t_colloc, grad=True),
        't_data': to_tensor(t_data),
        'delta_data': to_tensor(delta_data),
        'omega_data': to_tensor(omega_data),
        't_ic': to_tensor(t_ic),
        'delta_ic': to_tensor(delta_ic),
        'omega_ic': to_tensor(omega_ic),
    }


if __name__ == '__main__':
    # Example usage demonstrating multiple trajectory generation
    smib_params = SMIBParameters(
        H=5.0,
        D=0.05,
        omega0=2 * np.pi * 60,
        Pm=0.8,
        Pmax=2.1
    )
    
    data_generator = PINNDataGenerator(params=smib_params, seed=42)
    
    print("=== Generating Multiple Trajectories ===\n")
    trajectories = data_generator.generate_multiple_trajectories(
        n_trajectories=10,
        fault_start_range=(0.05, 0.3),
        fault_duration_range=(0.05, 0.15),
        t_total=2.0,
        dt=0.001,
        include_unstable=True,
        seed=42
    )
    
    print(f"Generated {len(trajectories)} trajectories:")
    stable_count = sum(1 for t in trajectories if t['is_stable'])
    unstable_count = len(trajectories) - stable_count
    print(f"  Stable: {stable_count}, Unstable: {unstable_count}\n")
    
    for i, traj in enumerate(trajectories[:5]):  # Show first 5
        print(f"Trajectory {i+1}:")
        print(f"  Fault: {traj['fault_start']:.3f}s - {traj['fault_end']:.3f}s "
              f"(duration: {traj['fault_end']-traj['fault_start']:.3f}s)")
        print(f"  Pm: {traj['Pm']:.3f} pu")
        print(f"  Delta range: {np.degrees(traj['delta'].min()):.1f}° - "
              f"{np.degrees(traj['delta'].max()):.1f}°")
        print(f"  Stable: {traj['is_stable']}")
        print()
    
    # Sample training data from all trajectories
    print("=== Sampling Training Data ===")
    training_data = data_generator.sample_from_multiple_trajectories(
        trajectories,
        n_collocation_per_traj=500,
        n_data_points_per_traj=15,
        t_total=2.0
    )
    
    print(f"Total collocation points: {training_data['t_colloc'].shape[0]}")
    print(f"Total data points: {training_data['t_data'].shape[0]}")
    print(f"Delta data range: {np.degrees(training_data['delta_data'].min().item()):.1f}° - "
          f"{np.degrees(training_data['delta_data'].max().item()):.1f}°")

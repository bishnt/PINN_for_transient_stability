import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple
from swing_equation import SMIBParameters, SwingEquationSolver

class StabilityAnalyzer:
    """
    Uses the trained PINN to perform stability analysis.
    Much faster than running individual ODE simulations.
    """
    def __init__(self, model, params: SMIBParameters, device):
        self.model = model
        self.params = params
        self.device = device
        self.model.eval()
        self.solver = SwingEquationSolver(params)

    def predict(self, t_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run PINN inference on time array. Returns delta and omega (absolute)."""
        t_tensor = torch.FloatTensor(t_values).reshape(-1, 1).to(self.device)
        with torch.no_grad():
            out = self.model(t_tensor).cpu().numpy()
        return out[:, 0], out[:, 1]  # delta, omega (absolute)

    def evaluate_accuracy(
        self,
        trajectory: dict,
        n_points: int = 500
    ) -> dict:
        """Compare PINN predictions against RK4 reference."""
        # Use 't' key to match data_generator output
        t_eval = np.linspace(0, trajectory['t'][-1], n_points)
        # PINN predictions (returns delta and omega - both absolute)
        d_pred, w_pred = self.predict(t_eval)
        # Interpolate RK4 to same time points
        d_ref = np.interp(t_eval, trajectory['t'], trajectory['delta'])
        w_ref = np.interp(t_eval, trajectory['t'], trajectory['omega'])
        # Both are now absolute values - compare directly
        mae_delta = np.mean(np.abs(d_pred - d_ref))
        mae_omega = np.mean(np.abs(w_pred - w_ref))
        rmse_d = np.sqrt(np.mean((d_pred - d_ref)**2))
        rmse_w = np.sqrt(np.mean((w_pred - w_ref)**2))
        return {
            't': t_eval,
            'delta_pred': d_pred, 'delta_ref': d_ref,
            'omega_pred': w_pred, 'omega_ref': w_ref,
            'mae_delta': mae_delta, 'mae_omega': mae_omega,
            'rmse_delta': rmse_d, 'rmse_omega': rmse_w,
        }

    def plot_comparison(self, results: dict, save_path: str = None):
        """Plot PINN vs RK4 side by side."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        t = results['t']
        ax1.plot(t, np.degrees(results['delta_ref']), 'b-',
                 linewidth=2.5, label='RK4 Reference', alpha=0.9)
        ax1.plot(t, np.degrees(results['delta_pred']), 'r--',
                 linewidth=2, label='PINN Prediction', alpha=0.9)
        ax1.fill_between(t,
                         np.degrees(results['delta_ref']),
                         np.degrees(results['delta_pred']),
                         alpha=0.15, color='red', label='Error region')
        ax1.set_ylabel('Rotor Angle delta (degrees)', fontsize=11)
        ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
        ax1.set_title(f"PINN vs RK4 | RMSE delta: {np.degrees(results['rmse_delta']):.4f} deg",
                      fontsize=12, fontweight='bold')
        ax2.plot(t, results['omega_ref'], 'b-',
                 linewidth=2.5, label='RK4 Reference')
        ax2.plot(t, results['omega_pred'], 'r--',
                 linewidth=2, label='PINN Prediction')
        ax2.set_ylabel('Angular Velocity omega (rad/s)', fontsize=11)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def compute_stability_map(
        self,
        fault_durations: np.ndarray,
        initial_angles: np.ndarray,
        t_total: float = 2.0,
        delta_threshold_deg: float = 120.0,
        use_pinn: bool = True
    ) -> np.ndarray:
        """
        Build a 2D stability map over (fault duration, initial angle).
        Returns: stability_map[i,j] = 1 (stable) or 0 (unstable)

        Args:
            fault_durations: Array of fault durations to test (in seconds)
            initial_angles: Array of initial rotor angles to test (in radians)
            t_total: Total simulation time
            delta_threshold_deg: Angle threshold for stability (degrees)
            use_pinn: If True, use RK4 (PINN not suitable for parameter sweeps);
                      if False, also use RK4
        """
        threshold = np.radians(delta_threshold_deg)
        stability_map = np.zeros((len(initial_angles), len(fault_durations)))
        print('Computing stability map...')

        for i, delta0 in enumerate(initial_angles):
            for j, fault_dur in enumerate(fault_durations):
                # Use RK4 for stability assessment
                # PINN is trained on a single fault scenario and cannot generalize
                # to different fault durations or initial angles
                try:
                    traj = self.solver.simulate(
                        t_span=(0.0, t_total),
                        dt=0.001,
                        delta0=delta0,
                        omega_deviated0=0.0,
                        fault_start=0.0,
                        fault_end=fault_dur,
                        fault_factor_pre=1.0,
                        fault_factor_fault=0.0,
                        fault_factor_post=1.0
                    )
                    max_angle = np.max(np.abs(traj['delta']))
                    stability_map[i, j] = 1.0 if max_angle < threshold else 0.0
                except:
                    stability_map[i, j] = 0.0
        return stability_map

    def plot_stability_map(
        self,
        fault_durations: np.ndarray,
        initial_angles: np.ndarray,
        stability_map: np.ndarray,
        save_path: str = None
    ):
        """Plot the 2D stability boundary map."""
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = LinearSegmentedColormap.from_list(
            'stability', ['#FF4444', '#44BB44'], N=2
        )
        im = ax.contourf(
            fault_durations * 1000,  # convert to ms
            np.degrees(initial_angles),
            stability_map,
            levels=1, cmap=cmap, alpha=0.7
        )
        ax.contour(
            fault_durations * 1000,
            np.degrees(initial_angles),
            stability_map,
            levels=1, colors='black', linewidths=2
        )
        plt.colorbar(im, ax=ax, label='Stable (1) / Unstable (0)', ticks=[0, 1])
        ax.set_xlabel('Fault Duration (ms)', fontsize=12)
        ax.set_ylabel('Initial Rotor Angle (degrees)', fontsize=12)
        ax.set_title('Transient Stability Boundary Map', fontsize=14, fontweight='bold')
        ax.text(0.02, 0.95, '* Stable Region', transform=ax.transAxes,
                color='#44BB44', fontsize=11, fontweight='bold')
        ax.text(0.02, 0.89, '* Unstable Region', transform=ax.transAxes,
                color='#FF4444', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

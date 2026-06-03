# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SMIBParameters:
  H: float = 5.0 #inertia constant
  D: float = 0.05 #damping coeff
  omega0: float = 2 * np.pi * 60 #synchronous speed
  Pm: float = 0.8 #mechanical power input (pu)
  Pmax: float = 2.1 #Max electrical power (pu)
  E: float = 1.05 #generator internal voltage (pu)
  V: float = 1.0 #infinite bus voltage (pu)

@property
#defining the pre fault rotor angle
def delta_eq(self) -> float:
  return np.arcsin(self.Pm / self.Pmax)

from typing import Tuple, Optional

#equation solver usimg RK-4th order method.
class SwingEquationSolver:
  def __init__(self, params: SMIBParameters):
    self.params = params

#modeling power output according to fault scenarios where:
#when ff = 1.0 (normal), when ff = 0.0 (short circuit), when ff = 0-1 (partial faults)
  def electrical_power(self,
                       delta: float,
                       fault_factor: float = 1.0) -> float:
    return fault_factor * self.params.Pmax * np.sin(delta)

#definining the swing equation(rhs form)
  def swing_rhs(self, t: float,
                state: np.ndarray,
                fault_factor: float = 1.0) -> np.ndarray:
    delta, omega_deviated = state
    Pe = self.electrical_power(delta, fault_factor)
    d_delta = self.params.omega0 * omega_deviated # d_delta = omega, where omega is the absolute angular speed, and omega_deviated is (omega - omega0)
    d_omega_deviated = (self.params.omega0 / (2 * self.params.H)) * (self.params.Pm - Pe - self.params.D * omega_deviated)
    return np.array([d_delta, d_omega_deviated])

  def rk4_step(self, t: float,
               state: np.ndarray,
               dt: float,
               fault_factor: float = 1.0) -> np.ndarray:
    k1 = self.swing_rhs(t, state, fault_factor)
    k2 = self.swing_rhs(t + dt / 2, state + dt * k1 / 2, fault_factor)
    k3 = self.swing_rhs(t + dt / 2, state + dt * k2 / 2, fault_factor)
    k4 = self.swing_rhs(t + dt, state + dt * k3, fault_factor)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  def simulate(self, t_span: Tuple[float, float],
               dt: float = 0.001,
               delta0: Optional[float] = None,
               omega_deviated0: float = 0.0,
               fault_start: float = 0.0,
               fault_end: float = 0.0,
               fault_factor_pre: float = 1.0,
               fault_factor_fault: float = 0.0,
               fault_factor_post: float = 1.0) -> dict:
    t0, tf = t_span

    times = []
    delta_trajectory = []
    omega_deviated_trajectory = []

    if delta0 is None:
      delta0 = self.params.delta_eq

    current_state = np.array([delta0, omega_deviated0])

    t = t0
    while t <= tf:
      # Determine current fault factor based on time
      if t < fault_start:
        current_fault_factor = fault_factor_pre
      elif fault_start <= t < fault_end:
        current_fault_factor = fault_factor_fault
      else:
        current_fault_factor = fault_factor_post

      # Store current state
      times.append(t)
      delta_trajectory.append(current_state[0])
      omega_deviated_trajectory.append(current_state[1])

      # Take an RK4 step
      current_state = self.rk4_step(t, current_state, dt, current_fault_factor)
      t += dt

    # Append the last state as well, as the loop condition might cause it to be skipped
    if not times or abs(times[-1] - tf) > 1e-9: # Check if last point is missing
        times.append(tf)
        delta_trajectory.append(current_state[0])
        omega_deviated_trajectory.append(current_state[1])

    return {
        'time': np.array(times),
        'delta': np.array(delta_trajectory),
        'omega_deviated': np.array(omega_deviated_trajectory)
    }




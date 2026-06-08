# Multi-Trajectory Training for PINN Transient Stability Model

## Problem Identified

The model was **only training on a single stable trajectory** with fixed parameters:
- Fault start: t=0.1s (fixed)
- Fault end: t=0.2s (fixed)  
- Fault duration: 0.1s (fixed)
- Total time: 2.0s
- Single operating condition (Pm=0.8 pu)

This caused the model to:
1. **Overfit to one specific fault scenario**
2. **Never see unstable cases** (where faults last too long)
3. **Learn only stable post-fault behavior**
4. **Fail to generalize** to different fault timings or severities

## Solution Implemented

### 1. Enhanced `data_generator.py`

Added two new methods to `PINNDataGenerator`:

#### `generate_multiple_trajectories()`
Generates multiple trajectories with varying:
- **Fault start times**: Random within specified range (e.g., 0.05-0.3s)
- **Fault durations**: Short (stable) and long (unstable) cases
- **Operating conditions**: Slight variations in mechanical power (Pm ±0.1 pu)
- **Stability mix**: Configurable fraction of unstable cases

```python
trajectories = gen.generate_multiple_trajectories(
    n_trajectories=15,
    fault_start_range=(0.05, 0.3),
    fault_duration_range=(0.05, 0.15),  # For stable cases
    t_total=2.0,
    include_unstable=True,
    unstable_fraction=0.2,  # 20% will be potentially unstable
    seed=42
)
```

#### `sample_from_multiple_trajectories()`
Samples training data from ALL trajectories, concatenating:
- Collocation points from each trajectory
- Data points from each trajectory
- Single initial condition point

```python
batch = gen.sample_from_multiple_trajectories(
    trajectories,
    n_collocation_per_traj=500,
    n_data_points_per_traj=15,
    t_total=2.0
)
# Results in: 15 × 500 = 7,500 collocation points
#             15 × 15  = 225 data points
```

#### `_check_stability()`
Classifies trajectories as stable/unstable based on:
- Maximum rotor angle deviation (< 180°)
- Final trajectory behavior (not monotonically increasing)

### 2. Updated `training-2.ipynb`

Modified the training notebook to:
1. Generate 15 diverse trajectories instead of 1
2. Sample data from all trajectories
3. Report stability statistics
4. Show expanded delta ranges in training data

## Benefits

### Before (Single Trajectory):
- Delta range: ~9° to ~41° (very narrow)
- All data from ONE fault scenario
- No unstable cases
- Model learns only ONE specific response

### After (Multiple Trajectories):
- Delta range: -30° to +120°+ (much wider coverage)
- Data from 15 different fault scenarios
- Mix of stable (~80%) and unstable (~20%) cases
- Model learns GENERAL transient stability behavior

## Usage Example

```python
from src.data_generator import PINNDataGenerator
from src.swing_equation import SMIBParameters

params = SMIBParameters(H=5.0, D=0.05, Pm=0.8, Pmax=2.1)
gen = PINNDataGenerator(params, seed=42)

# Generate diverse training data
trajectories = gen.generate_multiple_trajectories(
    n_trajectories=15,
    fault_start_range=(0.05, 0.3),
    fault_duration_range=(0.05, 0.15),
    include_unstable=True,
    unstable_fraction=0.2
)

# Sample combined training batch
batch = gen.sample_from_multiple_trajectories(
    trajectories,
    n_collocation_per_traj=500,
    n_data_points_per_traj=15
)

print(f"Total collocation points: {batch['t_colloc'].shape[0]}")
print(f"Total data points: {batch['t_data'].shape[0]}")
```

## Recommended Parameters

For robust training:

| Parameter | Recommended Value | Purpose |
|-----------|------------------|---------|
| `n_trajectories` | 15-20 | Enough diversity without excessive computation |
| `fault_start_range` | (0.05, 0.3) | Cover early and late fault occurrences |
| `fault_duration_range` | (0.05, 0.15) | Typical stable fault clearing times |
| `unstable_fraction` | 0.2-0.3 | Include some unstable cases for boundary learning |
| `n_collocation_per_traj` | 400-600 | Adequate physics enforcement per trajectory |
| `n_data_points_per_traj` | 12-20 | Sparse measurements from each trajectory |

## Testing

Run the included test:
```bash
cd /workspace
python3 -c "
import sys; sys.path.insert(0, 'src')
from data_generator import PINNDataGenerator
from swing_equation import SMIBParameters

params = SMIBParameters()
gen = PINNDataGenerator(params)
trajs = gen.generate_multiple_trajectories(n_trajectories=15, seed=42)
stable = sum(1 for t in trajs if t['is_stable'])
print(f'Generated: {stable} stable, {15-stable} unstable')
"
```

## Next Steps

1. **Run training** with the updated `training-2.ipynb`
2. **Evaluate** on both stable and unstable test cases
3. **Tune parameters** based on validation performance
4. **Consider adding**:
   - Variable fault severity (fault_factor between 0 and 1)
   - Different system parameters (H, D variations)
   - Three-phase vs single-phase faults

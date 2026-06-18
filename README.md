# PINN for Transient Stability Analysis

Physics-Informed Neural Networks (PINNs) for power system transient stability analysis using the swing equation. This project implements deep learning models to solve generator rotor dynamics during and after electrical faults, providing fast alternatives to traditional numerical integration methods.

## 🎯 Features

- **Single-Machine Infinite-Bus (SMIB) System**: Full implementation of the classical swing equation with fault modeling
- **Multi-Machine Extension**: Coupled swing equations for multiple generator systems
- **Physics-Informed Training**: Combines data-driven learning with physics-based constraints
- **Adaptive Loss Weighting**: Neural Tangent Kernel (NTK) based adaptive weighting for balanced training
- **Stability Analysis**: Fast stability boundary mapping and trajectory prediction
- **Comprehensive Evaluation**: Comparison with RK4 numerical integration and accuracy metrics

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+
- Matplotlib 3.7+
- Pandas 2.0+
- Scikit-learn 1.2+
- Jupyter 1.0+

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/bishnt/PINN_for_transient_stability.git
cd PINN_for_transient_stability
```

2. Create a virtual environment (recommended):
```bash
python -m venv pinn_env
source pinn_env/bin/activate  # On Windows: pinn_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
PINN_for_transient_stability/
├── src/
│   ├── model.py              # PINN architecture and normalization
│   ├── swing_equation.py     # SMIB parameters and RK4 solver
│   ├── loss.py               # Physics-informed loss function
│   ├── trainer.py            # Training logic (Adam + L-BFGS)
│   ├── data_generator.py     # Training data generation
│   ├── stability_analysis.py # Stability analysis and evaluation
│   └── multi_machine.py      # Multi-machine extension
├── notebook/
│   ├── training-1.ipynb      # Single-machine training
│   ├── training-2.ipynb      # Advanced training configurations
│   ├── training_3.ipynb      # Multi-machine training
│   ├── data_simulation.ipynb # Data generation and visualization
│   ├── evaluations.ipynb     # Model evaluation and comparison
│   ├── stability_analytics.ipynb  # Stability boundary analysis
│   └── multi-machine.ipynb   # Multi-machine system analysis
├── models/                   # Saved model checkpoints
├── checkpoints/              # Training checkpoints
├── plots/                    # Generated plots and figures
└── requirements.txt
```

## 🔬 Key Components

### 1. Swing Equation Solver
The classical swing equation for generator rotor dynamics:
```
dδ/dt = ω - ω₀
dω/dt = (ω₀/2H) × (Pm - Pe - D×(ω-ω₀))
```
Where:
- δ: Rotor angle
- ω: Angular velocity
- H: Inertia constant
- D: Damping coefficient
- Pm: Mechanical power
- Pe: Electrical power

### 2. PINN Architecture
- Feedforward neural network with configurable hidden layers and neurons
- Input: Time (t)
- Output: Rotor angle (δ) and angular velocity (ω)
- Normalization for stable training
- Xavier initialization

### 3. Loss Function
Combines three components:
- **Physics Loss**: Residual of the swing equation at collocation points
- **Data Loss**: MSE against sparse measurements
- **Initial Condition Loss**: Enforces t=0 conditions

### 4. Training Strategy
- **Phase 1**: Adam optimizer with adaptive learning rate
- **Phase 2**: L-BFGS fine-tuning for high precision
- Optional adaptive loss weighting based on NTK analysis

## 💻 Usage

### Single-Machine Training

```python
from swing_equation import SMIBParameters
from data_generator import PINNDataGenerator
from model import PINN
from loss import PINNLoss
from trainer import PINNTrainer

# Define system parameters
params = SMIBParameters(H=5.0, D=0.05, Pm=0.8, Pmax=2.1)

# Generate training data
gen = PINNDataGenerator(params, seed=42)
trajectory = gen.generate_reference_trajectory(
    fault_start=0.1, fault_end=0.2, t_total=2.0
)
batch = gen.sample_training_data(
    trajectory, n_collocation=5000, n_data_points=150
)

# Build and train model
model = PINN(n_hidden_layers=4, n_neurons=64)
loss_fn = PINNLoss(params, fault_start=0.1, fault_end=0.2)
trainer = PINNTrainer(model, loss_fn, device='cuda')

# Train with adaptive weighting
trainer.train_adam(batch, n_epochs=12000, lr=1e-3, use_adaptive_weights=True)
trainer.train_lbfgs(batch, n_epochs=400, lr=0.05, use_fixed_weights=True)

# Save model
trainer.save_model('models/pinn_trained.pt')
```

### Multi-Machine System

```python
from multi_machine import MachineParams, MultiMachineSolver, MultiMachinePINN

# Define machine parameters
machines = [
    MachineParams(H=5.0, D=0.05, Pm=0.8, E=1.05),
    MachineParams(H=4.0, D=0.06, Pm=0.7, E=1.02)
]

# Susceptance matrix
B_matrix = np.array([[0, -5.0], [-5.0, 0]])

# Create solver and simulate
solver = MultiMachineSolver(machines, B_matrix)
results = solver.simulate(t_span=(0, 2.0), dt=0.001)
```

### Stability Analysis

```python
from stability_analysis import StabilityAnalyzer

analyzer = StabilityAnalyzer(model, params, device='cuda')

# Evaluate accuracy
results = analyzer.evaluate_accuracy(trajectory)
analyzer.plot_comparison(results, save_path='plots/comparison.png')

# Generate stability map
fault_durations = np.linspace(0.05, 0.3, 20)
initial_angles = np.linspace(0, np.pi/2, 20)
stability_map = analyzer.compute_stability_map(
    fault_durations, initial_angles
)
analyzer.plot_stability_map(
    fault_durations, initial_angles, stability_map
)
```

## 📊 Results and Evaluation

The project includes comprehensive evaluation notebooks that demonstrate:

- **Trajectory Accuracy**: PINN predictions vs RK4 reference solutions
- **Training Convergence**: Loss history for both Adam and L-BFGS phases
- **Stability Boundaries**: 2D stability maps in fault duration vs initial angle space
- **Phase Portraits**: Rotor angle vs angular velocity trajectories
- **Multi-Machine Dynamics**: Coupled oscillator behavior

Typical accuracy metrics:
- RMSE (δ): < 0.01 radians for stable trajectories
- RMSE (ω): < 0.1 rad/s for stable trajectories
- Training time: ~5-10 minutes on GPU for single-machine system

## 📓 Notebooks

The `notebook/` directory contains interactive Jupyter notebooks:

1. **training-1.ipynb**: Basic single-machine training pipeline
2. **training-2.ipynb**: Advanced training with different configurations
3. **training_3.ipynb**: Multi-machine system training
4. **data_simulation.ipynb**: Data generation and visualization tools
5. **evaluations.ipynb**: Model evaluation and accuracy metrics
6. **stability_analytics.ipynb**: Stability boundary analysis
7. **multi-machine.ipynb**: Multi-machine system dynamics

## 🔧 Configuration

### System Parameters
Default SMIB parameters can be modified in `swing_equation.py`:
- `H`: Inertia constant (default: 5.0)
- `D`: Damping coefficient (default: 0.05)
- `Pm`: Mechanical power (default: 0.8 pu)
- `Pmax`: Maximum electrical power (default: 2.1 pu)

### Training Hyperparameters
Key training parameters in notebooks:
- `n_hidden_layers`: Number of hidden layers (default: 4-6)
- `n_neurons`: Neurons per layer (default: 64)
- `n_collocation`: Physics enforcement points (default: 5000)
- `n_data_points`: Sparse measurements (default: 150)
- `ADAM_EPOCHS`: Adam training iterations (default: 12000)
- `LBFGS_EPOCHS`: L-BFGS fine-tuning iterations (default: 400)

## 🎓 Applications

This research enables:
- **Fast Stability Assessment**: Real-time transient stability evaluation
- **Parametric Studies**: Quick exploration of different fault scenarios
- **Security Analysis**: Integration with power system security tools
- **Educational Tools**: Interactive learning of power system dynamics

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional machine models and network topologies
- Advanced PINN architectures (attention mechanisms, residual connections)
- Integration with real power system data
- Extended fault models and protection systems
- GPU optimization and parallel processing

## 📝 License

This project is provided for research and educational purposes.

## 🙏 Acknowledgments

- Built on the physics-informed neural networks framework
- Implements classical power system stability analysis methods
- Uses PyTorch for deep learning implementation

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: This implementation focuses on the classical swing equation model. For practical power system applications, additional modeling details and validation against real systems are recommended.
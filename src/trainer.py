import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, List

class PINNTrainer:
    def __init__(self, model, loss_fn, device):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.history = {'total': [], 'physics': [], 'data': [], 'ic': []}

    def move_batch_to_device(self, batch: Dict) -> Dict:
        GRAD_KEYS = {'t_colloc', 'x_colloc'}
        return {
            k: v.to(self.device).requires_grad_(k in GRAD_KEYS)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_adaptive_weights(self, batch: Dict):
          self.model.zero_grad()
          losses = self.loss_fn(self.model, batch)
    
          grad_norms = {}
          for key in ['physics', 'data', 'ic']:
                if losses[key].requires_grad:
                          grads = torch.autograd.grad(
                                          losses[key],
                                          self.model.parameters(),
                                          retain_graph=True,
                                          allow_unused=True
                          )
                          norm = sum(
                                          g.norm() ** 2 for g in grads if g is not None
                          ).sqrt()
                          grad_norms[key] = norm.item() + 1e-8
                else:
                          grad_norms[key] = 1e-8
    
          mean_norm = np.mean(list(grad_norms.values()))
          weights = {k: mean_norm / v for k, v in grad_norms.items()}
          w_sum = sum(weights.values())
          weights = {k: 3.0 * v / w_sum for k, v in weights.items()}
          return weights
    
    def train_adam(
          self,
          batch: Dict,
          n_epochs: int = 5000,
          lr: float = 1e-3,
          log_every: int = 500,
          rebalance_every: int = 1000,
    ):
          batch = self.move_batch_to_device(batch)          # ← MUST be first
          weights = self._compute_adaptive_weights(batch)   # ← now batch is ready
          print(f'  Initial weights → physics: {weights["physics"]:.3f} | '
                          f'data: {weights["data"]:.3f} | ic: {weights["ic"]:.3f}')
    
          optimizer = optim.Adam(self.model.parameters(), lr=lr)
          scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=500, factor=0.5
          )
    
          print(' Phase 1: Adam Optimization')
          pbar = tqdm(range(n_epochs), desc='Adam')
          for epoch in pbar:
                if epoch > 0 and epoch % rebalance_every == 0:
                          weights = self._compute_adaptive_weights(batch)
    
                optimizer.zero_grad()
                losses = self.loss_fn(self.model, batch)
                total = (
                          weights['physics'] * losses['physics']
                          + weights['data']  * losses['data']
                          + weights['ic']    * losses['ic']
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(total.detach())
    
                self.history['total'].append(total.item())
                for key in ['physics', 'data', 'ic']:
                          self.history[key].append(losses[key].item())
    
                if epoch % log_every == 0:
                          pbar.set_postfix({
                                          'total':   f"{total.item():.2e}",
                                          'physics': f"{losses['physics'].item():.2e}",
                                          'data':    f"{losses['data'].item():.2e}",
                          })
    
          print(f'Adam done. Final loss: {self.history["total"][-1]:.2e}')
          print(f'  Final weights → physics: {weights["physics"]:.3f} | '
                          f'data: {weights["data"]:.3f} | ic: {weights["ic"]:.3f}')

    def train_lbfgs(                           
        self,
        batch: Dict,
        n_epochs: int = 400,
        lr: float = 0.05,
    ):
        batch = self.move_batch_to_device(batch)

        # Freeze weights at whatever Adam converged to
        weights = self._compute_adaptive_weights(batch)

        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=20,
            history_size=100,
            line_search_fn='strong_wolfe'
        )

        print(' Phase 2: L-BFGS Fine-Tuning')
        pbar = tqdm(range(n_epochs), desc='L-BFGS')

        for epoch in pbar:
            def closure():
                optimizer.zero_grad()
                losses = self.loss_fn(self.model, batch)
                total = (
                    weights['physics'] * losses['physics']
                    + weights['data']   * losses['data']
                    + weights['ic']     * losses['ic']
                )
                total.backward()
                return total

            optimizer.step(closure)

            with torch.no_grad():
                losses = self.loss_fn(self.model, batch)
                total = (
                    weights['physics'] * losses['physics']
                    + weights['data']   * losses['data']
                    + weights['ic']     * losses['ic']
                )

            self.history['total'].append(total.item())
            for key in ['physics', 'data', 'ic']:
                self.history[key].append(losses[key].item())

            if epoch % 100 == 0:
                pbar.set_postfix({'total': f"{total.item():.2e}"})

        print(f'L-BFGS done. Final loss: {self.history["total"][-1]:.2e}')

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f'Model loaded from {path}')
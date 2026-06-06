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
        self.weights = {'physics': 1.0, 'data': 1.0, 'ic': 1.0}
        self.history = {'total': [], 'physics': [], 'data': [], 'ic': []}

    def move_batch_to_device(self, batch: Dict) -> Dict:
        GRAD_KEYS = {'t_colloc', 'x_colloc'}
        return {
            k: v.to(self.device).requires_grad_(k in GRAD_KEYS)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_adaptive_weights(self, batch, alpha=0.1, min_weight=0.05):
        """
        Computes stabilized NTK-based adaptive weights for PINN losses.
        """
        # Calculate losses first so the 'losses' variable exists in this scope
        losses = self.loss_fn(self.model, batch)
        
        # Calculate gradient norms for each loss component
        grad_norms = {}
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        for key in ['physics', 'data', 'ic']:
            grads = torch.autograd.grad(
                losses[key], 
                params, 
                retain_graph=True, 
                allow_unused=True
            )
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
            grad_norms[key] = grad_norm.item()

        # Compute NTK traces safely
        traces = {}
        for key in ['physics', 'data', 'ic']:
            val = losses[key]
            loss_val = val.item() if hasattr(val, 'item') else float(val)
            loss_val = max(loss_val, 1e-6)
            traces[key] = (grad_norms[key] ** 2) / (4 * loss_val)

        # Anchor-Based Scaling relative to 'data' loss
        anchor_trace = traces['data'] + 1e-8
        target_w_physics = anchor_trace / (traces['physics'] + 1e-8)
        target_w_ic = anchor_trace / (traces['ic'] + 1e-8)

        # Smooth updates across epochs using Exponential Moving Average (EMA)
        self.weights['physics'] = (1 - alpha) * self.weights['physics'] + alpha * target_w_physics
        self.weights['ic'] = (1 - alpha) * self.weights['ic'] + alpha * target_w_ic
        self.weights['data'] = 1.0 

        # Guardrail floor
        self.weights['physics'] = max(self.weights['physics'], min_weight)
        self.weights['ic'] = max(self.weights['ic'], min_weight)

        return self.weights

    def train_adam(
        self,
        batch: Dict,
        n_epochs: int = 5000,
        lr: float = 1e-3,
        log_every: int = 500,
        rebalance_every: int = 50,  # FIX 1: Update weights much more frequently
    ):
        batch = self.move_batch_to_device(batch)
        weights = self._compute_adaptive_weights(batch)
        print(f'  Initial weights → physics: {weights["physics"]:.3f} | '
              f'data: {weights["data"]:.3f} | ic: {weights["ic"]:.3f}')
    
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)
    
        print(' Phase 1: Adam Optimization')
        pbar = tqdm(range(n_epochs), desc='Adam')
        for epoch in pbar:
            if epoch > 0 and epoch % rebalance_every == 0:
                weights = self._compute_adaptive_weights(batch, alpha=0.2) 
    
            optimizer.zero_grad()
            losses = self.loss_fn(self.model, batch)
            
            total = (
                weights['physics'] * losses['physics']
                + weights['data']  * losses['data']
                + weights['ic']    * losses['ic']
            )
            
            total.backward()
            
            # Gradient clipping protects against exploding gradients during weight shifts
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()  # Step every epoch based on schedule, not plateau
    
            # Log history
            self.history['total'].append(total.item())
            for key in ['physics', 'data', 'ic']:
                self.history[key].append(losses[key].item())
    
            if epoch % log_every == 0:
                pbar.set_postfix({
                    'total':   f"{total.item():.2e}",
                    'physics': f"{losses['physics'].item():.2e}",
                    'data':    f"{losses['data'].item():.2e}",
                    'lr':      f"{optimizer.param_groups[0]['lr']:.1e}"
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
    
                # Freeze adaptive loss weights at whatever Adam converged to
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
                          # Placeholders to capture the latest metrics from inside the closure
                          epoch_losses = {}
                          epoch_total = None
    
                          def closure():
                                          nonlocal epoch_losses, epoch_total
                                          optimizer.zero_grad()
                                          # Ensure t_colloc has requires_grad=True for physics residual
                                          batch['t_colloc'] = batch['t_colloc'].detach().requires_grad_(True)
                                          losses = self.loss_fn(self.model, batch)
                                          total = (
                                          weights['physics'] * losses['physics']
                                          + weights['data']   * losses['data']
                                          + weights['ic']     * losses['ic']
                                          )
                                          total.backward()
                                          
                                          # Safely capture detached copies for logging
                                          epoch_losses = {k: v.detach() for k, v in losses.items()}
                                          epoch_total = total.detach()
                                          
                                          return total
    
                          # L-BFGS evaluates the closure multiple times per step during line search
                          optimizer.step(closure)
    
                          # Log metrics using the values captured from the final closure evaluation
                          self.history['total'].append(epoch_total.item())
                          for key in ['physics', 'data', 'ic']:
                                          self.history[key].append(epoch_losses[key].item())
    
                          if epoch % 100 == 0:
                                          pbar.set_postfix({'total': f"{epoch_total.item():.2e}"})
    
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
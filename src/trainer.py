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
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_adam(
        self,
        batch: Dict,
        n_epochs: int = 5000,
        lr: float = 1e-3,
        log_every: int = 500
    ):
        batch = self.move_batch_to_device(batch)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=500, factor=0.5
        )

        print(' Phase 1: Adam Optimization')
        pbar = tqdm(range(n_epochs), desc='Adam')
        for epoch in pbar:
            optimizer.zero_grad()
            losses = self.loss_fn(self.model, batch)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(losses['total'])

            for key in self.history:
                self.history[key].append(losses[key].item())

            if epoch % log_every == 0:
                pbar.set_postfix({
                    'total': f"{losses['total'].item():.2e}",
                    'physics': f"{losses['physics'].item():.2e}",
                    'data': f"{losses['data'].item():.2e}",
                })
        print(f'Adam done. Final loss: {self.history["total"][-1]:.2e}')

    def train_lbfgs(
        self,
        batch: Dict,
        n_epochs: int = 1000,
        lr: float = 0.01
    ):
        batch = self.move_batch_to_device(batch)
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lr, max_iter=20,
            history_size=100,
            line_search_fn='strong_wolfe'
        )

        print(' Phase 2: L-BFGS Fine-Tuning')
        pbar = tqdm(range(n_epochs), desc='L-BFGS')
        for epoch in pbar:
            def closure():
                optimizer.zero_grad()
                losses = self.loss_fn(self.model, batch)
                losses['total'].backward()
                return losses['total']
            optimizer.step(closure)

            with torch.no_grad():
                losses = self.loss_fn(self.model, batch)
            for key in self.history:
                self.history[key].append(losses[key].item())
            if epoch % 100 == 0:
                pbar.set_postfix({'total': f"{losses['total'].item():.2e}"})
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
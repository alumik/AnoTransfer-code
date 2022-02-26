import os
import torch
import numpy as np
import anomalytransfer as at

from typing import Optional
from torch.backends import cudnn
from torch.utils.data import TensorDataset, DataLoader


class Encoder(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self._lstm = torch.nn.LSTM(input_dim, hidden_dim)
        self._hidden_to_mean = torch.nn.Linear(hidden_dim, latent_dim)
        self._hidden_to_log_std = torch.nn.Linear(hidden_dim, latent_dim)

        torch.nn.init.xavier_uniform_(self._hidden_to_mean.weight)
        torch.nn.init.xavier_uniform_(self._hidden_to_log_std.weight)

    def forward(self, x):
        _, (h_end, c_end) = self._lstm(x)
        hidden = h_end[-1, :, :]
        self.mean = self._hidden_to_mean(hidden)
        self.log_std = self._hidden_to_log_std(hidden)
        if self.training:
            std = torch.exp(0.5 * self.log_std)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(self.mean)
        else:
            return self.mean


class Decoder(torch.nn.Module):

    def __init__(self,
                 seq_length: int,
                 latent_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 batch_size: int,
                 device: str):
        super(Decoder, self).__init__()

        self._lstm = torch.nn.LSTM(1, hidden_dim)
        self._latent_to_hidden = torch.nn.Linear(latent_dim, hidden_dim)
        self._hidden_to_output = torch.nn.Linear(hidden_dim, output_dim)

        self._model_input = torch.zeros(seq_length, batch_size, 1).to(device)
        self._c_0 = torch.zeros(1, batch_size, hidden_dim).to(device)

        torch.nn.init.xavier_uniform_(self._latent_to_hidden.weight)
        torch.nn.init.xavier_uniform_(self._hidden_to_output.weight)

    def forward(self, x):
        hidden = self._latent_to_hidden(x)
        h_0 = torch.stack([hidden])
        hidden, _ = self._lstm(self._model_input, (h_0, self._c_0))
        return self._hidden_to_output(hidden)


class LatentTransformer(torch.nn.Module):

    def __init__(self,
                 seq_length: int,
                 input_dim: int,
                 hidden_dim: int = 90,
                 latent_dim: int = 20,
                 batch_size: int = 32,
                 max_grad_norm: int = 5,
                 device: Optional[str] = None):
        super().__init__()

        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        cudnn.benchmark = True
        if device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        self._encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self._decoder = Decoder(seq_length=seq_length, latent_dim=latent_dim, hidden_dim=hidden_dim,
                                output_dim=input_dim, batch_size=self._batch_size, device=self._device)
        self._is_fitted = False
        self.to(self._device)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=20, gamma=0.9)
        self._loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        return self._decoder(self._encoder(x))

    def _loss(self, x) -> torch.Tensor:
        x_recon = self(x)
        mean, log_std = self._encoder.mean, self._encoder.log_std
        kl_loss = -0.5 * torch.mean(1 + log_std - mean.pow(2) - log_std.exp())
        reconstruction_loss = self._loss_fn(x_recon, x)
        return kl_loss + reconstruction_loss

    def fit(self, data: torch.Tensor, epochs: int, verbose=1):
        self.train()
        dataset = TensorDataset(data.to(self._device))
        train_loader = DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=True, drop_last=True)
        print('Training Epochs')
        if verbose:
            progbar = at.utils.ProgBar(epochs, interval=0.5, stateful_metrics=['loss'], unit_name='epoch')
        for i in range(epochs):
            epoch_losses = []
            for x in train_loader:
                x = x[0].permute(1, 0, 2)
                self._optimizer.zero_grad()
                loss = self._loss(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self._max_grad_norm)
                self._optimizer.step()
                epoch_losses.append(loss)
            self._lr_scheduler.step()
            epoch_loss = torch.mean(torch.as_tensor(epoch_losses)).numpy()
            if verbose:
                progbar.add(1, values=[('loss', epoch_loss)])
        self._is_fitted = True

    def transform(self, data: torch.Tensor) -> np.ndarray:
        self.eval()
        dataset = TensorDataset(data.to(self._device))
        test_loader = DataLoader(dataset=dataset, batch_size=self._batch_size)
        print('Transforming Steps')
        progbar = at.utils.ProgBar(len(test_loader), interval=0.5)
        if self._is_fitted:
            with torch.no_grad():
                latent = []
                for x in test_loader:
                    x = x[0].permute(1, 0, 2)
                    x = self._encoder(x).cpu().numpy()
                    latent.append(x)
                    progbar.add(1)
                return np.concatenate(latent, axis=0)
        raise RuntimeError('Model needs to be fitted')

    def save(self, path: str):
        if self._is_fitted:
            torch.save(self.state_dict(), path)
        else:
            raise RuntimeError('Model needs to be fitted')

    def load(self, path: str):
        self._is_fitted = True
        self.load_state_dict(torch.load(path))

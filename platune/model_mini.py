import torch
import pytorch_lightning as pl

from typing import Dict

from platune.networks.transformerv2 import Denoiser


class PLaTuneMini(pl.LightningModule):

    DEFAULT_MODEL_CONFIG: Dict[str, object] = {
        "latent_dim": 64,
        "seq_len": 32,
        "embed_dim": 256,
        "noise_embed_dims": 128,
        "n_layers": 4,
        "mlp_multiplier": 3,
        "dropout": 0.1,
        "causal": False,
        "lr": 1e-4,
        "use_grad_clip": True,
        "nb_steps": 20,
        "n_audio_examples": 12,
        "sample_rate": 44100,
    }

    def __init__(
        self,
        latent_dim: int = 64,
        seq_len: int = 32,
        embed_dim: int = 256,
        noise_embed_dims: int = 128,
        n_layers: int = 4,
        mlp_multiplier: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
        lr: float = 1e-4,
        use_grad_clip: bool = True,
        nb_steps: int = 20,
        n_audio_examples: int = 12,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.flow = Denoiser(
            n_channels=latent_dim,
            seq_len=seq_len,
            embed_dim=embed_dim,
            noise_embed_dims=noise_embed_dims,
            n_layers=n_layers,
            mlp_multiplier=mlp_multiplier,
            dropout=dropout,
            causal=causal
        )

        self.latent_dim = latent_dim
        self.lr = lr
        self.use_grad_clip = use_grad_clip
        self.nb_steps = nb_steps

        self.automatic_optimization = False

        self._codec = None
        self.n_audio_examples = n_audio_examples
        self.sample_rate = sample_rate

    @classmethod
    def default_model_config(cls) -> Dict[str, object]:
        return dict(cls.DEFAULT_MODEL_CONFIG)

    @property
    def codec(self):
        if self._codec is None:
            from music2latent import EncoderDecoder
            self._codec = EncoderDecoder(device=self.device)
        return self._codec

    def decode_latent_to_audio(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            audio = self.codec.decode(z)
            if audio.dim() == 3:
                audio = audio[:, 0, :]
        return audio

    def configure_optimizers(self):
        return torch.optim.AdamW(self.flow.parameters(), self.lr, (.5, .999))

    @torch.no_grad()
    def s_to_z(self, s, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]
        x = s.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.flow(x, time=t) * dt
        return x

    @torch.no_grad()
    def z_to_s(self, z, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(1, 0, nb_steps + 1).to(self.device)[:-1]
        x = z.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x - self.flow(x, time=t) * dt
        return x

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        z = batch[0].to(self.device)
        s = torch.randn_like(z)

        target = z - s
        t = torch.rand(z.size(0), 1, 1, device=self.device)
        interpolant = (1 - t) * s + t * z
        model_output = self.flow(interpolant, time=t)
        loss = ((model_output - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.)
        opt.step()

        self.log("diffusion_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.logger is None or not hasattr(self.logger, "experiment") or batch_idx != 0:
            return

        z = batch[0].to(self.device)
        n_log = min(self.n_audio_examples, z.shape[0])
        z_orig_audio = self.decode_latent_to_audio(z[:n_log])
        z_synth = self.s_to_z(torch.randn_like(z[:n_log]), nb_steps=self.nb_steps)
        z_synth_audio = self.decode_latent_to_audio(z_synth)

        for i in range(n_log):
            self.logger.experiment.add_audio(
                f"val_audio/0_synth_{i}",
                z_synth_audio[i],
                self.global_step,
                sample_rate=self.sample_rate,
            )
            self.logger.experiment.add_audio(
                f"val_audio/1_original_{i}",
                z_orig_audio[i],
                self.global_step,
                sample_rate=self.sample_rate,
            )


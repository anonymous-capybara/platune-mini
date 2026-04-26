import gin
import torch
import torch.nn as nn
import torch.distributions as D
import pytorch_lightning as pl

from typing import Callable, Dict, List, Tuple

from platune.helpers.data_visualization import plot_features_extraction


@gin.configurable
class PLaTune(pl.LightningModule):

    def __init__(
            self,
            flow: Callable[[], nn.Module],
            latent_dim: int = 64,
            discrete_keys: List[str] = None,
            continuous_keys: List[str] = None,
            classes_attr_discrete: List[List[int]] = [],
            min_max_attr_continuous: List[Tuple[int]] = [],
            bins_values: List[List[int]] = [],
            sigma_init: float = 0.4,
            r: float = 0.25,
            sigma_decay: float = 0.995,
            sigma_target_continuous: float = 0.005,
            lr: int = 1e-4,
            use_grad_clip: bool = False,
            n_ex_val: int = 0,
            nb_steps: int = 20,
            n_alpha_channels: int = 0,
            sigma_target_alpha: float = 0.3,
            codec_name: str = "music2latent",
            latents_per_timestep: int = 8,
            aug_probs: List[float] = [],
        ):

        super().__init__()

        # model
        self.flow = flow()

        # discrete controls
        self.discrete_keys = discrete_keys or []
        self.n_attr_discrete = len(self.discrete_keys)
        self.classes_attr_discrete = classes_attr_discrete
        self.n_classes = [len(a) for a in self.classes_attr_discrete] if self.n_attr_discrete > 0 else []
        self.min_max_attr_discrete = [(0, v - 1) for v in self.n_classes] if self.n_attr_discrete > 0 else []

        # continuous controls
        self.continuous_keys = continuous_keys or []
        self.n_attr_continuous = len(self.continuous_keys)
        self.min_max_attr_continuous = min_max_attr_continuous

        self.bins_values = torch.tensor(bins_values)
        self.n_quantized_classes = None
        if len(bins_values) > 0:
            print(
                'using quantized continuous attributes (e.g. processed by the model as discrete attributes)'
            )
            self.n_quantized_classes = len(bins_values[0])
        
        self.all_keys = self.discrete_keys + self.continuous_keys

        # dims
        self.latent_dim = latent_dim
        self.control_dim = self.n_attr_discrete + self.n_attr_continuous

        self.channel_ranges: Dict[str, Tuple[int, int]] = {}
        for i, key in enumerate(self.discrete_keys):
            self.channel_ranges[key] = (i, i + 1)
        for i, key in enumerate(self.continuous_keys):
            self.channel_ranges[key] = (self.n_attr_discrete + i, self.n_attr_discrete + i + 1)

        if self.n_quantized_classes is not None:
            self.min_max_attr = self.min_max_attr_discrete + [
                (0, self.n_quantized_classes - 1)
                for _ in range(self.n_attr_continuous)
            ]
        else:
            self.min_max_attr = self.min_max_attr_discrete + self.min_max_attr_continuous

        self.style_dim = latent_dim - self.control_dim

        # Optional N-dimensional alpha channels (augmentation strength conditioning)
        self.n_alpha_channels = n_alpha_channels
        if self.n_alpha_channels > 0:
            self.control_dim += self.n_alpha_channels
            self.style_dim -= self.n_alpha_channels
            # NOTE: do NOT append to min_max_attr — alpha is normalized
            # separately in append_alpha_to_controls(), and normalize_attr()
            # is called before alpha is concatenated.
            self.channel_ranges["alpha"] = (self.control_dim - self.n_alpha_channels, self.control_dim)
            self.all_keys = self.all_keys + ["alpha"]
            print(f"Alpha channels enabled: n_alpha={self.n_alpha_channels}, "
                  f"control_dim={self.control_dim}, style_dim={self.style_dim}")

        assert self.style_dim > 0, (
            f"control_dim ({self.control_dim}) >= latent_dim ({latent_dim}). "
            f"Not enough room for style dimensions.")
        
        # hparams
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.r = r

        sigma_target = []
        if self.n_attr_discrete > 0:
            self.sigma_target_discrete = torch.tensor([
                (2 / self.n_classes[i]) * r
                for i in range(self.n_attr_discrete)
            ])
            sigma_target.append(self.sigma_target_discrete)

        if self.n_attr_continuous > 0:
            if len(bins_values) > 0:
                self.sigma_target_continuous = torch.tensor([
                    (2 / self.n_quantized_classes) * r
                    for i in range(self.n_attr_continuous)
                ])
            else:
                self.sigma_target_continuous = torch.full(
                    (self.n_attr_continuous, ), sigma_target_continuous)

            sigma_target.append(self.sigma_target_continuous)

        if self.n_alpha_channels > 0:
            sigma_target.append(torch.full((self.n_alpha_channels,), sigma_target_alpha))

        if len(sigma_target) > 0:
            self.sigma_target = torch.cat(sigma_target)
        else:
            self.sigma_target = torch.empty(0, dtype=torch.float32)

        self.lr = lr
        self.use_grad_clip = use_grad_clip

        self.automatic_optimization = False
        
        # for validation step
        self.n_examples = n_ex_val
        self.nb_steps = nb_steps

        self.validation_step_outputs = {}

        # Audio logging (codec loaded lazily)
        self._codec = None
        self.codec_name = codec_name
        self.latents_per_timestep = latents_per_timestep
        self.aug_probs = aug_probs
        # Precompute cumulative probabilities for efficient multi-loader sampling
        if len(aug_probs) > 0:
            self._aug_cumprobs = torch.cumsum(torch.tensor(aug_probs), dim=0)
        else:
            self._aug_cumprobs = torch.tensor([])
        self.n_audio_examples = 12  # Number of audio examples to log
        self.sample_rate = 44100

    @property
    def codec(self):
        """Lazily load codec for audio decoding."""
        if self._codec is None:
            if self.codec_name == "codicodec":
                from codicodec import EncoderDecoder
            else:
                from music2latent import EncoderDecoder
            self._codec = EncoderDecoder(device=self.device)
        return self._codec
    
    def decode_latent_to_audio(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor to audio waveform.
        
        Args:
            z: [B, latent_dim, T_flat] where T_flat = T_orig * latents_per_timestep
        """
        with torch.no_grad():
            if self.codec_name == "codicodec" and self.latents_per_timestep > 1:
                # Reshape from [B, 64, T_flat] back to [B, T_orig, latents_per_timestep, 64]
                B = z.shape[0]
                T_flat = z.shape[2]
                T_orig = T_flat // self.latents_per_timestep
                z = z.transpose(1, 2).reshape(B, T_orig, self.latents_per_timestep, -1)
            audio = self.codec.decode(z)
            # codicodec returns [B, 2, samples] (stereo) — take first channel
            if audio.dim() == 3:
                audio = audio[:, 0, :]
        return audio  # [B, samples]

    def configure_optimizers(self):
        params = list(self.flow.parameters())
        opt = torch.optim.AdamW(params, self.lr, (.5, .999))
        return opt

    def normalize_attr(self, x, invert=False):
        min_v = [t[0] for t in self.min_max_attr]
        min_values = torch.tensor(min_v, dtype=x.dtype).unsqueeze(0).unsqueeze(2).to(x.device)
        
        max_v = [t[1] for t in self.min_max_attr]
        max_values = torch.tensor(max_v, dtype=x.dtype).unsqueeze(0).unsqueeze(2).to(x.device)

        if not invert:
            # normalize between -1 and 1
            x = (x - min_values) / (max_values - min_values)
            x = 2 * x - 1
        else:
            x = (0.5 * (x + 1))
            x = (max_values - min_values) * x + min_values
        return x

    def append_alpha_to_controls(self, c, alpha):
        """
        Append normalized alpha channels to the control tensor.
        
        Args:
            c: Normalized controls [B, control_dim_without_alpha, T]
            alpha: Raw alpha tensor [B, N_alpha, T] with values in [0, 1]
            
        Returns:
            c_with_alpha: [B, control_dim, T] (control_dim includes N_alpha channels)
        """
        # Normalize alpha from [0, 1] to [-1, 1]
        alpha_norm = 2 * alpha.to(c.device) - 1
        return torch.cat([c, alpha_norm], dim=1)

    def build_control_tensor(self, z, ad, ac, alpha):
        """
        Build the normalized control tensor used by train/validation.

        Returns a tensor shaped [B, control_dim, T]. When there are no
        attribute channels, this creates an empty [B, 0, T] tensor and
        optionally appends alpha conditioning.
        """
        if self.n_attr_discrete > 0 or self.n_attr_continuous > 0:
            a = self.process_attributes(ad, ac)
            c = self.normalize_attr(a)
        else:
            c = z.new_empty((z.shape[0], 0, z.shape[-1]))

        if self.n_alpha_channels > 0:
            c = self.append_alpha_to_controls(c, alpha)

        return c
    
    def convert_class_values_to_indices(self, a):
        new_a = []
        for i in range(self.n_attr_discrete):
            classes_i = torch.tensor(self.classes_attr_discrete[i]).to(a.device)
            a_ids = torch.searchsorted(classes_i.contiguous(), a[:, i, :].contiguous())
            new_a.append(a_ids.unsqueeze(1))
        new_a = torch.cat(new_a, dim=1)
        return new_a
    
    def process_attributes(self, ad, ac):
        attr = []
        if ad.shape[-1] > 0:
            ad = ad.to(self.device)
            ad_ids = self.convert_class_values_to_indices(ad)
            attr.append(ad_ids)

        if ac.shape[-1] > 0:
            ac = ac.to(self.device)
            if len(self.bins_values) > 0:
                ac_quantized = torch.zeros_like(ac).to(self.device)
                self.bins_values = self.bins_values.to(ac.device)
                for i in range(self.n_attr_continuous):
                    data = ac[:, i, :].flatten()
                    classes = torch.bucketize(data, self.bins_values[i])
                    ac_quantized[:, i, :] = classes.reshape(-1, ac.shape[-1])
                attr.append(ac_quantized)
            else:
                attr.append(ac)

        if len(attr) == 0:
            raise ValueError(
                "process_attributes() requires at least one discrete or continuous "
                "attribute channel. Use build_control_tensor() for no-control models."
            )

        attr = torch.cat(attr, dim=1)
        return attr
    
    @torch.no_grad()
    def cs_to_z(self, cs, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]
        x = cs.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.flow(x, time=t) * dt
        return x

    @torch.no_grad()
    def z_to_cs(self, z, nb_steps=10):
        dt = 1 / nb_steps
        t_values = torch.linspace(1, 0, nb_steps + 1).to(self.device)[:-1]
        x = z.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x - self.flow(x, time=t) * dt
        return x

    def get_sigma(self, a, warmup=False):
        if warmup:
            # apply warmup on sigma_target
            progress = self.global_step / 50
            current_sigma_values = self.sigma_target + (self.sigma_init - self.sigma_target) * (self.sigma_decay**progress)
        else: 
            current_sigma_values = self.sigma_target

        if self.control_dim == 0:
            return torch.zeros((a.shape[0], 0, a.shape[-1]), dtype=a.dtype, device=a.device)

        current_sigma = torch.cat([
            torch.full((a.shape[0], 1, a.shape[-1]), current_sigma_values[i])
            for i in range(self.control_dim)
        ], dim=1).to(a.device)
        return current_sigma

    def get_cs_distributions(self, a, warmup=False, zero_var=False):
        # define control distribution
        current_sigma = self.get_sigma(a, warmup)
        if zero_var:
            c_dist = D.Normal(a, 0.001 * torch.ones_like(current_sigma))
        else:
            c_dist = D.Normal(a, current_sigma)

        # define style distribution
        s_dist = D.Normal(
            torch.zeros((a.shape[0], self.style_dim, a.shape[-1])).to(a.device),
            torch.ones((a.shape[0], self.style_dim, a.shape[-1])).to(a.device)
        )

        return c_dist, s_dist

    def get_cs_samples(self, c_dist, s_dist):
        """
        Sample from control and style distributions.
        """
        c_samples = c_dist.sample()
        s_samples = s_dist.sample()
        cs = torch.cat([c_samples, s_samples], dim=1)
        return cs

    def compute_nll(self, cs, c_dist, s_dist):
        c = cs[:, :self.control_dim]
        s = cs[:, self.control_dim:]

        if self.control_dim > 0:
            c_logp = c_dist.log_prob(c)
            nll_control = -c_logp.sum(1, keepdim=True).mean()
        else:
            nll_control = torch.zeros((), dtype=cs.dtype, device=cs.device)

        s_logp = s_dist.log_prob(s)

        nll_style = -s_logp.sum(1, keepdim=True).mean()

        return nll_control + nll_style, nll_control, nll_style

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        z, ad, ac, alpha = batch
        z = z.to(self.device)

        # Randomly substitute with augmented batch from one of the augmented loaders
        aug_loaders = getattr(self, '_aug_loaders', [])
        if len(aug_loaders) > 0:
            p = torch.rand(1).item()
            # Find which loader (if any) this draw selects
            selected_idx = None
            for i, cumprob in enumerate(self._aug_cumprobs):
                if p < cumprob.item():
                    selected_idx = i
                    break
            if selected_idx is not None:
                try:
                    if self._aug_iters[selected_idx] is None:
                        self._aug_iters[selected_idx] = iter(aug_loaders[selected_idx])
                    aug_batch = next(self._aug_iters[selected_idx])
                except StopIteration:
                    self._aug_iters[selected_idx] = iter(aug_loaders[selected_idx])
                    aug_batch = next(self._aug_iters[selected_idx])
                z, ad, ac, alpha = aug_batch
                z = z.to(self.device)

        c = self.build_control_tensor(z, ad, ac, alpha)

        # Get the distribution samples
        c_dist, s_dist = self.get_cs_distributions(c, warmup=True)
        cs = self.get_cs_samples(c_dist, s_dist)

        # diffusion loss
        target = z - cs
        t = torch.rand(z.size(0), 1, 1).to(self.device)
        interpolant = (1 - t) * cs + t * z
        model_output = self.flow(interpolant, time=t)
        loss = ((model_output - target)**2).mean()

        # optimization
        opt.zero_grad()
        loss.backward()
        # gradient clipping:
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.flow.parameters()), 1.)
        opt.step()

        # tensorboard
        self.log("diffusion_loss", loss)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.trainer is not None:

            z, ad, ac, alpha = batch
            z = z.to(self.device)

            c = self.build_control_tensor(z, ad, ac, alpha)

            c_dist, s_dist = self.get_cs_distributions(c, warmup=False)
            cs = self.get_cs_samples(c_dist, s_dist)

            cs_rec = self.z_to_cs(z, nb_steps=self.nb_steps)
            nll_loss, nll_control, nll_style = self.compute_nll(cs_rec, c_dist, s_dist)

            # Conditional synthesis: source controls + random Gaussian style
            B, _, T = c.shape
            style_random = torch.randn(B, self.style_dim, T, device=self.device)
            cs_synth = torch.cat([c, style_random], dim=1)
            z_synth = self.cs_to_z(cs_synth, nb_steps=self.nb_steps)

            self.validation_step_outputs["feat_extract_loss"] = self.validation_step_outputs.get("feat_extract_loss", []) + [nll_loss.item()]
            self.validation_step_outputs["feat_extract_loss_control"] = self.validation_step_outputs.get("feat_extract_loss_control", []) + [nll_control.item()]
            self.validation_step_outputs["feat_extract_loss_style"] = self.validation_step_outputs.get("feat_extract_loss_style", []) + [nll_style.item()]

            # Log audio: original + conditionally synthesized
            if self.logger is not None and hasattr(self.logger, "experiment"):
                if batch_idx == 0:
                    n_log = min(self.n_audio_examples, z.shape[0])
                    z_orig_audio = self.decode_latent_to_audio(z[:n_log])
                    z_synth_audio = self.decode_latent_to_audio(z_synth[:n_log])

                    for i in range(n_log):
                        self.logger.experiment.add_audio(
                            f"val_audio/original_{i}",
                            z_orig_audio[i],
                            self.global_step,
                            sample_rate=self.sample_rate,
                        )
                        self.logger.experiment.add_audio(
                            f"val_audio/cond_synth_{i}",
                            z_synth_audio[i],
                            self.global_step,
                            sample_rate=self.sample_rate,
                        )
            
            if self.n_examples > 0:
                ex_c_gt = self.validation_step_outputs.get("c_gt", [])
                if self.all_keys and (len(ex_c_gt) == 0 or len(ex_c_gt) < self.n_examples):
                    n_ex = self.n_examples - len(ex_c_gt)
                    curr_c_gt = [c[i] for i in range(n_ex)]
                    self.validation_step_outputs["c_gt"] =  ex_c_gt + curr_c_gt
                    curr_c_rec = [cs_rec[:,:self.control_dim,:][i] for i in range(n_ex)]
                    self.validation_step_outputs["c_rec"] =  self.validation_step_outputs.get("c_rec", []) + curr_c_rec
            
            self.log('validation', nll_loss.item())

            return nll_loss.item()

    def _control_to_per_attr_for_plot(self, c_tensor):
        """
        Convert a control tensor to per-attribute 1D trajectories for plotting.
        
        Each attribute occupies one scalar control channel.
        Alpha channels (if any) are plotted individually.
        
        Returns a list of (attr_name, 1D tensor) tuples.
        """
        result = []
        for key in self.all_keys:
            start, end = self.channel_ranges[key]
            if key == "alpha":
                # Plot each alpha dimension separately
                for ch in range(start, end):
                    result.append((f"alpha_{ch - start}", c_tensor[ch, :]))
            else:
                result.append((key, c_tensor[start, :]))
        return result

    def on_validation_epoch_end(self):

        for k, v in self.validation_step_outputs.items():
            if k not in ["c_gt", "c_rec"]:
                self.log(k, torch.mean(torch.tensor(v)))
        
        n_ex = len(self.validation_step_outputs.get("c_gt", []))
        if n_ex > 0:
            for i in range(n_ex):
                c_gt_i = self.validation_step_outputs["c_gt"][i]
                c_rec_i = self.validation_step_outputs["c_rec"][i]
                gt_attrs = self._control_to_per_attr_for_plot(c_gt_i)
                rec_attrs = self._control_to_per_attr_for_plot(c_rec_i)
                for k, (attr_name, gt_vals) in enumerate(gt_attrs):
                    _, rec_vals = rec_attrs[k]
                    f_ik = plot_features_extraction(
                        c_gt=gt_vals,
                        c_rec=rec_vals,
                        descriptor_name=attr_name,
                        figsize=(10, 5),
                    )
                    self.logger.experiment.add_figure(f"ex {i} - features extraction {attr_name}", f_ik, self.global_step)

        self.validation_step_outputs = {}

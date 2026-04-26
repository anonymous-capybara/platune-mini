import os
import gin
import lmdb
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional

from .audio_example import AudioExample
from .base import SimpleDataset


CONTINUOUS_ATTRIBUTES = [
    'rms', 'loudness1s', 'integrated_loudness', 'centroid', 'bandwidth',
    'booming', 'sharpness', 'arousal', 'valence', "dark", "epic", "retro",
    "fast", "loudness1s", "energetic", "melodic", "emotional", "dark",
    "average_duration", "note_density", "central_pitch", "pitch_range",
    "children", "energetic", "emotional", "dark"  # MIDI note (21-108), ordinal so treated as continuous
]
DISCRETE_ATTRIBUTES = [
    'onsets',
    'pitch',
    'octave',
    'pitch_processed',
    'octave_processed',
    'instrument',
    'velocity',
    'dynamics',
    'Mode',
    "key",
    "melody",
    "melody_processed",
    "playing_technique",
]
AE_RATIO = 4096
SAMPLE_RATE = 44100
WINDOW_SIZE = 12.
HOP_SIZE = 3.


class LatentsContinuousDiscreteAttritbutesDataset(SimpleDataset):

    def __init__(
        self,
        path,
        keys=['z'] + CONTINUOUS_ATTRIBUTES + DISCRETE_ATTRIBUTES,
        use_hardcodec_keys=False,
        lmdb_keys_file: str = None,
        dataset_name: str = None,
        crop=None,
        n_alpha_channels: int = 0,
    ):

        if use_hardcodec_keys:
            keys = keys + CONTINUOUS_ATTRIBUTES + DISCRETE_ATTRIBUTES
        else:
            if "z" not in keys:
                keys = ["z"] + keys
        super().__init__(path, keys)

        if lmdb_keys_file is not None:
            with open(os.path.join(path, f"{lmdb_keys_file}.pkl"), "rb") as f:
                lmdb_keys = pickle.load(f)
            self.keys = lmdb_keys

        self.dataset_name = dataset_name
        self.crop = crop
        self.n_alpha_channels = n_alpha_channels
        print(self.crop)

    def __getitem__(self, idx):
        if self.cache:
            return self.data[idx]

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[idx]))

            z = torch.from_numpy(ae.get("z"))

        # if z.shape[-1] not power of 2 replicate last frame
        z_length = z.shape[-1]
        is_pow_2 = (z_length > 0 and (z_length & (z_length - 1)) == 0)

        if not is_pow_2:
            z = torch.nn.functional.pad(z, (0, 1), mode='replicate')

        attr_discrete = []
        attr_continuous = []

        metadata = None
        w = None
        midi = None

        for key in self.buffer_keys:
            if key == 'z':
                continue

            if key in DISCRETE_ATTRIBUTES:
                try:
                    attr = torch.from_numpy(ae.get(key))
                except:
                    attr = torch.zeros(z_length)

                if attr.shape[-1] != z_length:
                    #     print(
                    #         "warning, you are using interpolation on discrete data, maybe try it first - not tested"
                    #     )
                    attr = torch.nn.functional.interpolate(
                        attr.reshape(1, 1, -1),
                        mode='nearest',
                        # align_corners=True,
                        size=z_length).reshape(-1).long()

                attr_discrete.append(attr)
            elif key in CONTINUOUS_ATTRIBUTES:
                if self.dataset_name == 'jamendo':
                    try:
                        attr = ae.get(key)

                        z_time_indices = np.linspace(0, (z_length * AE_RATIO) /
                                                     SAMPLE_RATE, z_length)
                        attr_time_indices = np.asarray([
                            (WINDOW_SIZE / 2) + i * HOP_SIZE
                            for i in range(attr.shape[-1])
                        ])
                        a = np.interp(x=z_time_indices,
                                      xp=attr_time_indices,
                                      fp=attr).astype(np.float32)
                        attr = torch.from_numpy(a)
                    except:
                        print("error in jamendo")
                        attr = torch.zeros(z_length)
                else:
                    try:
                        attr = torch.from_numpy(ae.get(key))
                    except:
                        attr = torch.zeros(z_length)
                if attr.shape[-1] != z_length:
                    attr = torch.nn.functional.interpolate(
                        attr.reshape(1, 1, -1),
                        mode='linear',
                        align_corners=True,
                        size=z_length).reshape(-1)

                attr_continuous.append(attr)
            elif key == 'metadata':
                metadata = ae.get_metadata()
                metadata['lmdb_key'] = self.keys[idx]
            elif key == 'waveform':
                w = torch.from_numpy(ae.get("waveform"))
            elif key == 'midi':
                midi = ae.get("midi")
            else:
                raise ValueError(
                    f'Need to specify if attribute is discrete or continuous for key={key}'
                )

        if len(attr_discrete) > 0:
            attr_discrete = torch.stack(attr_discrete)
            if not is_pow_2:
                attr_discrete = torch.nn.functional.pad(attr_discrete, (0, 1),
                                                        mode='replicate')
        else:
            # attr_discrete = np.array(attr_discrete)
            attr_discrete = torch.tensor(attr_discrete)

        if len(attr_continuous) > 0:
            attr_continuous = torch.stack(attr_continuous)
            if not is_pow_2:
                attr_continuous = torch.nn.functional.pad(attr_continuous,
                                                          (0, 1),
                                                          mode='replicate')
        else:
            # attr_continuous = np.array(attr_continuous)
            attr_continuous = torch.tensor(attr_continuous)

        # Build alpha tensor: [N_alpha, T] float tensor, all zeros for non-augmented data
        T_final = z.shape[-1]
        alpha_tensor = torch.zeros((self.n_alpha_channels, T_final))

        if metadata is not None and self.crop is None:
            if w is not None:
                if midi is not None:
                    return z, attr_discrete, attr_continuous, alpha_tensor, metadata, w, midi
                else:
                    return z, attr_discrete, attr_continuous, alpha_tensor, metadata, w
            elif midi is not None:
                return z, attr_discrete, attr_continuous, alpha_tensor, metadata, midi
            else:
                return z, attr_discrete, attr_continuous, alpha_tensor, metadata
        if self.crop is not None:
            id_crop = np.random.randint(0, z_length - self.crop)
            z = z[:, id_crop:id_crop + self.crop]
            if len(attr_continuous) > 0:
                attr_continuous = attr_continuous[:,
                                                  id_crop:id_crop + self.crop]
            if len(attr_discrete) > 0:
                attr_discrete = attr_discrete[:, id_crop:id_crop + self.crop]
            alpha_tensor = alpha_tensor[:, id_crop:id_crop + self.crop]
            if metadata is not None and midi is not None:
                return z, attr_discrete, attr_continuous, alpha_tensor, metadata, midi
        return z, attr_discrete, attr_continuous, alpha_tensor


class AugmentedLatentsDataset(SimpleDataset):
    """Dataset reading precomputed augmented entries from a separate LMDB.
    Each entry contains z, the anchor's attributes, and the alpha vector.

    When multiple augmented LMDBs with different enabled_augmentations are used,
    `alpha_mapping` remaps the LMDB's local alpha channels into a canonical
    union vector (zero-filled for missing channels).

    Returns the same 4-tuple as LatentsContinuousDiscreteAttritbutesDataset:
        (z, attr_discrete, attr_continuous, alpha_tensor)
    """

    def __init__(
        self,
        path,
        discrete_keys: List[str] = [],
        continuous_keys: List[str] = [],
        n_alpha_channels: int = 0,
        alpha_mapping: Optional[List[int]] = None,
        crop=None,
    ):
        # Register buffer keys: z + attributes
        keys_list = ["z"] + discrete_keys + continuous_keys
        super().__init__(path, keys_list)

        self.discrete_keys = discrete_keys
        self.continuous_keys = continuous_keys
        self.n_alpha_channels = n_alpha_channels
        self.alpha_mapping = alpha_mapping  # indices into canonical union vector
        self.crop = crop

    def __getitem__(self, idx):
        if self.cache:
            return self.data[idx]

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[idx]))

        z = torch.from_numpy(ae.get("z"))

        # Read stored alpha vector [N_alpha]
        try:
            alpha_vec = torch.from_numpy(ae.get("alpha")).float()
        except Exception:
            alpha_vec = torch.zeros(self.n_alpha_channels)

        z_length = z.shape[-1]
        is_pow_2 = (z_length > 0 and (z_length & (z_length - 1)) == 0)
        if not is_pow_2:
            z = torch.nn.functional.pad(z, (0, 1), mode='replicate')

        attr_discrete = []
        attr_continuous = []
        for key in self.discrete_keys:
            try:
                attr = torch.from_numpy(ae.get(key))
            except Exception:
                attr = torch.zeros(z_length)
            if attr.shape[-1] != z_length:
                attr = torch.nn.functional.interpolate(
                    attr.reshape(1, 1, -1), mode='nearest',
                    size=z_length).reshape(-1).long()
            attr_discrete.append(attr)

        for key in self.continuous_keys:
            try:
                attr = torch.from_numpy(ae.get(key))
            except Exception:
                attr = torch.zeros(z_length)
            if attr.shape[-1] != z_length:
                attr = torch.nn.functional.interpolate(
                    attr.reshape(1, 1, -1), mode='linear',
                    align_corners=True, size=z_length).reshape(-1)
            attr_continuous.append(attr)

        if len(attr_discrete) > 0:
            attr_discrete = torch.stack(attr_discrete)
            if not is_pow_2:
                attr_discrete = torch.nn.functional.pad(
                    attr_discrete, (0, 1), mode='replicate')
        else:
            attr_discrete = torch.tensor(attr_discrete)

        if len(attr_continuous) > 0:
            attr_continuous = torch.stack(attr_continuous)
            if not is_pow_2:
                attr_continuous = torch.nn.functional.pad(
                    attr_continuous, (0, 1), mode='replicate')
        else:
            attr_continuous = torch.tensor(attr_continuous)

        # Remap local alpha to canonical union vector [n_alpha_channels]
        if self.alpha_mapping is not None and self.n_alpha_channels > 0:
            canonical_alpha = torch.zeros(self.n_alpha_channels)
            # alpha_vec may be shorter than n_alpha_channels; scatter into mapped positions
            for local_idx, union_idx in enumerate(self.alpha_mapping):
                if local_idx < alpha_vec.shape[0]:
                    canonical_alpha[union_idx] = alpha_vec[local_idx]
            alpha_vec = canonical_alpha

        # Build alpha tensor: broadcast [N_alpha] -> [N_alpha, T_final]
        T_final = z.shape[-1]
        alpha_tensor = alpha_vec.unsqueeze(-1).expand(-1, T_final)

        if self.crop is not None:
            id_crop = np.random.randint(0, z_length - self.crop)
            z = z[:, id_crop:id_crop + self.crop]
            if len(attr_continuous) > 0:
                attr_continuous = attr_continuous[:, id_crop:id_crop + self.crop]
            if len(attr_discrete) > 0:
                attr_discrete = attr_discrete[:, id_crop:id_crop + self.crop]
            alpha_tensor = alpha_tensor[:, id_crop:id_crop + self.crop]

        return z, attr_discrete, attr_continuous, alpha_tensor


@gin.configurable
def load_data(data_path: str,
              discrete_keys: List[str] = [],
              continuous_keys: List[str] = [],
              batch_size: int = 8,
              n_workers: int = 0,
              cache: bool = False,
              lmdb_keys_file: str = None,
              dataset_name: str = None,
              crop: int = None,
              augmented_lmdb_paths: Optional[List[str]] = None,
              augmented_alpha_mappings: Optional[List[List[int]]] = None,
              n_alpha_channels: int = 0,
              val_data_path: str = None):

    # Common kwargs shared between train and val datasets
    common_kwargs = dict(
        path=data_path,
        keys=["z"] + discrete_keys + continuous_keys,
        lmdb_keys_file=lmdb_keys_file,
        dataset_name=dataset_name,
        crop=crop,
        n_alpha_channels=n_alpha_channels,
    )

    train_full = LatentsContinuousDiscreteAttritbutesDataset(**common_kwargs)

    if val_data_path is None:
        # Auto-discover validation LMDB at <data_path>_val
        auto_val_path = data_path.rstrip('/') + "_val"
        if os.path.isdir(auto_val_path):
            val_data_path = auto_val_path
            print(f"Auto-discovered validation LMDB: {val_data_path}")
        else:
            raise ValueError(
                f"No val_data_path provided and auto-discovery failed "
                f"(looked for {auto_val_path}). "
                f"Run prepare_dataset.py with --val_fraction to create a validation LMDB."
            )

    print(f"Using separate validation LMDB: {val_data_path}")
    print(f"  Training uses 100% of {data_path} ({len(train_full)} entries)")

    val_dataset = LatentsContinuousDiscreteAttritbutesDataset(
        path=val_data_path,
        keys=["z"] + discrete_keys + continuous_keys,
        n_alpha_channels=n_alpha_channels,
    )

    if cache:
        train_full.build_cache()
        val_dataset.build_cache()

    train_dataset = train_full

    print("dataset sizes : ", len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True,
                              persistent_workers=True,
                              num_workers=n_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size,
                            shuffle=False,
                            persistent_workers=True,
                            num_workers=n_workers)

    print("dataloader sizes : ", len(train_loader), len(val_loader))

    # Optional augmented LMDB DataLoaders (one per augmented path)
    aug_loaders = []
    if augmented_lmdb_paths is not None:
        if augmented_alpha_mappings is None:
            augmented_alpha_mappings = [None] * len(augmented_lmdb_paths)  # type: ignore[list-item]
        for i, aug_path in enumerate(augmented_lmdb_paths):
            alpha_mapping = augmented_alpha_mappings[i] if augmented_alpha_mappings else None
            aug_dataset = AugmentedLatentsDataset(
                path=aug_path,
                discrete_keys=discrete_keys,
                continuous_keys=continuous_keys,
                n_alpha_channels=n_alpha_channels,
                alpha_mapping=alpha_mapping,
                crop=crop,
            )
            if cache:
                aug_dataset.build_cache()
            print(f"  augmented LMDB {i}: {len(aug_dataset)} entries from {aug_path} "
                  f"(alpha_mapping={alpha_mapping})")

            aug_loader = DataLoader(aug_dataset,
                                    batch_size,
                                    shuffle=True,
                                    persistent_workers=True,
                                    num_workers=n_workers)
            print(f"  augmented dataloader {i}: {len(aug_loader)} batches")
            aug_loaders.append(aug_loader)

    return train_loader, val_loader, aug_loaders

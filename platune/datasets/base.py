import gin
import torch
import lmdb
import numpy as np
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset

from platune.datasets.audio_example import AudioExample


class SimpleDataset(Dataset):

    def __init__(
        self,
        path,
        keys="all",
        transforms=None,
        readonly=True,
        map_size=None,
        num_samples=None,
    ) -> None:
        super().__init__()

        # Store path and config for lazy LMDB initialization (fork-safe)
        self._lmdb_path = path
        self._lmdb_readonly = readonly
        self._lmdb_map_size = map_size
        self._env = None  # Will be lazily initialized per-worker

        # Open temporarily to get keys (will be closed and reopened lazily)
        with self._get_env().begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        if keys == "all":
            self.buffer_keys = self.get_keys()
        else:
            self.buffer_keys = keys
        self.transforms = transforms

        if num_samples is not None and num_samples < len(self.keys):
            np.random.seed(42)
            self.keys = list(
                np.random.choice(self.keys, num_samples, replace=False))

        self.cache = False
        
        # Close the env after init so workers can open their own
        self._close_env()

    @property
    def env(self):
        """Lazily get LMDB environment (fork-safe: each worker opens its own)."""
        return self._get_env()

    def _get_env(self):
        """Get or create LMDB environment for current process."""
        if self._env is None:
            if self._lmdb_map_size is not None:
                self._env = lmdb.open(
                    self._lmdb_path,
                    lock=False,
                    readonly=self._lmdb_readonly,
                    readahead=False,
                    map_async=False,
                    map_size=1024**3 * self._lmdb_map_size
                )
            else:
                self._env = lmdb.open(
                    self._lmdb_path,
                    lock=False,
                    readonly=self._lmdb_readonly,
                    readahead=False,
                    map_async=False,
                )
        return self._env

    def _close_env(self):
        """Close LMDB environment (called after init and before fork)."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def build_cache(self):
        self.cache = False
        print("building cache")
        self.data = []
        for i in tqdm(range(len(self))):
            self.data.append(self.__getitem__(i))
        self.cache = True

    def __len__(self):
        return len(self.keys)

    def get_keys(self):
        with self._get_env().begin() as txn:
            ae = AudioExample(txn.get(self.keys[0]))
        return ae.get_keys()

    def __getitem__(self, index=None, key=None):
        if self.cache == True:
            return self.data[index]

        with self.env.begin() as txn:
            if key is not None:
                ae = AudioExample(txn.get(key))
            else:
                ae = AudioExample(txn.get(self.keys[index]))
        out = {}
        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    pass
                    #print("key: ", key, " not found")
        return out

    def _filter_keys_by_instrument(self, max_chunks: int = None, 
                                     instrument_filter: List[str] = None,
                                     seed: int = 42) -> List:
        """
        Filter LMDB keys by instrument category and optionally limit chunks per instrument.
        
        Args:
            max_chunks: Maximum number of chunks to keep per instrument (None = no limit)
            instrument_filter: List of instrument names to include (None = all instruments)
            seed: Random seed for reproducible subset selection
            
        Returns:
            Filtered list of LMDB keys
        """
        from collections import defaultdict
        
        # Create isolated RNG for reproducibility
        rng = np.random.default_rng(seed)
        
        # Group keys by instrument
        instrument_to_keys = defaultdict(list)
        
        filter_msg = []
        if instrument_filter is not None:
            filter_msg.append(f"instruments={instrument_filter}")
        if max_chunks is not None:
            filter_msg.append(f"max {max_chunks} chunks each")
        filter_msg.append(f"seed={seed}")
        print(f"Filtering keys by instrument ({', '.join(filter_msg)})...")
        
        with self.env.begin() as txn:
            for key in self.keys:
                ae = AudioExample(txn.get(key))
                try:
                    metadata = ae.get_metadata()
                    instrument = metadata.get("instrument", "unknown")
                except:
                    instrument = "unknown"
                
                # Skip if instrument not in filter list
                if instrument_filter is not None and instrument not in instrument_filter:
                    continue
                    
                instrument_to_keys[instrument].append(key)
        
        # Limit each instrument category
        filtered_keys = []
        for instrument in sorted(instrument_to_keys.keys()):  # Sort for deterministic order
            keys = instrument_to_keys[instrument]
            # Shuffle with seeded RNG for reproducibility
            rng.shuffle(keys)
            if max_chunks is not None:
                selected = keys[:max_chunks]
            else:
                selected = keys
            filtered_keys.extend(selected)
            print(f"  {instrument}: {len(selected)}/{len(instrument_to_keys[instrument])} chunks")
        
        # Shuffle the final list with seeded RNG
        rng.shuffle(filtered_keys)
        print(f"Total chunks after filtering: {len(filtered_keys)}")
        
        return filtered_keys

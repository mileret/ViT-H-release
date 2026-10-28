import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset, Subset
from omegaconf import ListConfig, DictConfig
from hmr4d.utils.pylogger import Log
from numpy.random import choice
from torch.utils.data import default_collate

import random as _py_random
import numpy as _np
import torch as _torch


class SourceHomogeneousBatchSampler:
    """Build batches containing samples from exactly one ConcatDataset child.

    Besides keeping per-source schemas batchable (Bones Seed/SOMA versus SMPL),
    this sampler preserves sequential locality inside each source.  That matters
    for the lazy Bones Seed joblib cache: fully random indices would repeatedly
    decompress different multi-GB pkl files.

    DDP ranks consume disjoint contiguous slices from the same global source
    block.  Every rank receives the same source schedule and number of batches.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=42):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("SourceHomogeneousBatchSampler requires ConcatDataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    @staticmethod
    def _dist_info():
        if _torch.distributed.is_available() and _torch.distributed.is_initialized():
            return _torch.distributed.get_rank(), _torch.distributed.get_world_size()
        return 0, 1

    def __len__(self):
        _, world_size = self._dist_info()
        global_block = self.batch_size * world_size
        if self.drop_last:
            return sum(len(ds) // global_block for ds in self.dataset.datasets)
        return sum((len(ds) + global_block - 1) // global_block for ds in self.dataset.datasets)

    def __iter__(self):
        rank, world_size = self._dist_info()
        global_block = self.batch_size * world_size
        generator = _torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        source_queues = []
        source_offset = 0
        for ds in self.dataset.datasets:
            n = len(ds)
            if self.drop_last:
                n_blocks = n // global_block
            else:
                n_blocks = (n + global_block - 1) // global_block

            block_ids = list(range(n_blocks))
            # Circularly rotate rather than fully permuting blocks.  This gives
            # a different epoch start while retaining pkl/cache locality.
            if self.shuffle and n_blocks > 1:
                shift = int(_torch.randint(n_blocks, (1,), generator=generator).item())
                block_ids = block_ids[shift:] + block_ids[:shift]

            queue = []
            for block_id in block_ids:
                global_start = block_id * global_block
                local_start = global_start + rank * self.batch_size
                local_end = min(local_start + self.batch_size, n)
                local = list(range(source_offset + local_start, source_offset + local_end))
                if len(local) < self.batch_size:
                    if self.drop_last or not local:
                        continue
                    # DDP requires identical full batch shapes.  Repeat only
                    # inside the same source, never with another dataset.
                    while len(local) < self.batch_size:
                        local.append(local[len(local) % max(len(local), 1)])
                queue.append(local)
            source_queues.append(queue)
            source_offset += n

        # Randomly interleave source queues while preserving the order inside
        # each queue. Choosing proportional to remaining batches reproduces the
        # original ConcatDataset sample proportions without mixed-source batches.
        positions = [0 for _ in source_queues]
        remaining = sum(len(q) for q in source_queues)
        while remaining > 0:
            available = [len(q) - positions[i] for i, q in enumerate(source_queues)]
            draw = int(_torch.randint(remaining, (1,), generator=generator).item())
            cumulative = 0
            source_id = None
            for i, count in enumerate(available):
                cumulative += count
                if draw < cumulative:
                    source_id = i
                    break
            if source_id is None:
                raise RuntimeError("Failed to select a non-empty source queue")
            yield source_queues[source_id][positions[source_id]]
            positions[source_id] += 1
            remaining -= 1

        self.epoch += 1


class SourceMixedBatchSampler:
    """Build DDP-safe batches with samples from all ConcatDataset children.

    Every rank owns a contiguous, non-overlapping shard of each source and all
    ranks use the same shuffled source-label schedule.  Indices consumed inside
    each rank/source shard remain sequential (with an epoch-dependent circular
    start).

    This is important for lazy multi-GB stores such as Bones/SOMA.  The old
    implementation assigned source indices globally and then sliced each
    global batch by rank.  A rank therefore advanced through Bones by roughly
    ``world_size`` times its local consumption; DataLoader workers multiplied
    that stride again and repeatedly evicted/decompressed different pkl files.
    Per-rank contiguous shards retain true mixed-source batches while keeping
    each worker's accesses local.  Matching source schedules also guarantee
    that source-dependent preprocessing takes the same branch on every rank.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=42):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("SourceMixedBatchSampler requires ConcatDataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not self.drop_last:
            raise ValueError("SourceMixedBatchSampler currently requires drop_last=True")

    @staticmethod
    def _dist_info():
        if _torch.distributed.is_available() and _torch.distributed.is_initialized():
            return _torch.distributed.get_rank(), _torch.distributed.get_world_size()
        return 0, 1

    def __len__(self):
        _, world_size = self._dist_info()
        local_total = sum(len(ds) // world_size for ds in self.dataset.datasets)
        return local_total // self.batch_size

    def __iter__(self):
        rank, world_size = self._dist_info()
        lengths = _torch.tensor([len(ds) for ds in self.dataset.datasets], dtype=_torch.long)
        local_lengths = lengths // world_size
        local_total = int(local_lengths.sum().item())
        usable = (local_total // self.batch_size) * self.batch_size
        if usable == 0:
            self.epoch += 1
            return

        generator = _torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # This source schedule is deliberately identical on every rank.  The
        # samples behind each label are different because each rank owns a
        # disjoint contiguous shard of that source.
        source_ids = _torch.repeat_interleave(_torch.arange(len(lengths)), local_lengths)
        if self.shuffle:
            source_ids = source_ids[_torch.randperm(local_total, generator=generator)]
        source_ids = source_ids[:usable]

        # Assign each source occurrence its next sequential index inside this
        # rank's contiguous source shard.  At most ``world_size - 1`` samples
        # per source are dropped so shards never overlap.
        offsets = _torch.cat([_torch.zeros(1, dtype=_torch.long), lengths.cumsum(0)[:-1]])
        global_indices = _torch.empty(usable, dtype=_torch.long)
        for source_id, local_len in enumerate(local_lengths.tolist()):
            locations = _torch.nonzero(source_ids == source_id, as_tuple=False).flatten()
            if locations.numel() == 0:
                continue
            if local_len <= 0:
                raise RuntimeError(
                    f"Source {source_id} has fewer samples than world_size={world_size}."
                )
            start = 0
            if self.shuffle and local_len > 1:
                start = int(_torch.randint(local_len, (1,), generator=generator).item())
            sequential = (_torch.arange(locations.numel()) + start) % local_len
            shard_start = rank * local_len
            global_indices[locations] = sequential + shard_start + offsets[source_id]

        for local_batch in global_indices.reshape(-1, self.batch_size):
            yield local_batch.tolist()

        self.epoch += 1


def _worker_init_fn(worker_id: int):
    """Seed NumPy and Python RNGs from the PyTorch worker seed."""
    seed = (_torch.initial_seed() + worker_id) % (2 ** 32)
    _np.random.seed(seed)
    _py_random.seed(seed)


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


_OPTIONAL_SOMA_KEYS = {"soma_joints_w", "soma_obs_joints_w"}


def collate_fn(batch):
    """Collate SMPL and Bones/SOMA samples in one batch.

    Bones/SOMA has two extra top-level tensors.  Missing tensors cannot be
    represented by ``None`` because PyTorch's default collate rejects a
    tensor/None mixture.  Instead, fill the missing entries with zeros of the
    exact reference shape and expose ``source_is_soma`` so the training step
    can choose the observation builder independently for every sample.

    All non-optional keys must still match exactly; an unrelated schema drift
    remains an error rather than being silently hidden.
    """
    if not batch:
        raise ValueError("collate_fn received an empty batch")

    soma_flags = ["soma_obs_joints_w" in sample for sample in batch]
    expected_common_keys = set(batch[0].keys()) - _OPTIONAL_SOMA_KEYS
    for i, sample in enumerate(batch[1:], start=1):
        sample_common_keys = set(sample.keys()) - _OPTIONAL_SOMA_KEYS
        if sample_common_keys != expected_common_keys:
            raise ValueError(
                "Mixed dataset schemas differ outside the supported SOMA fields. "
                f"sample0_only={sorted(expected_common_keys - sample_common_keys)}, "
                f"sample{i}_only={sorted(sample_common_keys - expected_common_keys)}"
            )

    return_dict = {}
    for k in batch[0].keys():
        if k in _OPTIONAL_SOMA_KEYS:
            continue
        if k.startswith("meta"):  # data information, do not batch
            return_dict[k] = [d[k] for d in batch]
        else:
            return_dict[k] = default_collate([d[k] for d in batch])

    # Keep the old homogeneous-batch behaviour while also supporting a mixed
    # batch.  These tensors are only consumed for samples selected by the mask.
    if any(soma_flags):
        for key in _OPTIONAL_SOMA_KEYS:
            reference = next((sample[key] for sample in batch if key in sample), None)
            if reference is None:
                continue
            if not _torch.is_tensor(reference):
                raise TypeError(f"Optional SOMA field {key!r} must be a tensor")
            values = [sample[key] if key in sample else _torch.zeros_like(reference) for sample in batch]
            return_dict[key] = default_collate(values)

    return_dict["source_is_soma"] = _torch.tensor(soma_flags, dtype=_torch.bool)
    return_dict["B"] = len(batch)
    return return_dict


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_opts: DictConfig, loader_opts: DictConfig, limit_each_trainset=None, test_subset_indices=None):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)

        Args:
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.dataset_opts = dataset_opts
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset
        self.test_subset_indices = list(test_subset_indices) if test_subset_indices is not None else None
        self._setup_splits = set()

    def _setup_split(self, split: str):
        """Instantiate one split lazily after distributed rendezvous.

        Lightning's subprocess DDP launcher starts child ranks only when
        ``Trainer.fit/test`` is entered. Instantiating the large PTH-backed
        datasets in ``__init__`` therefore made rank 0 enter the process-group
        rendezvous while ranks 1-3 were still loading data, eventually hitting
        the 30-minute TCPStore timeout. ``setup`` runs after the strategy has
        initialized distributed, so all ranks rendezvous before heavy I/O.
        """
        if split in self._setup_splits or split not in self.dataset_opts:
            return

        # Train uses concat dataset
        if split == "train":
            assert "train" in self.loader_opts, "train not in loader_opts"
            split_opts = self.dataset_opts.get("train")
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                if self.limit_each_trainset:
                    dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                dataset.append(dataset_i)
                Log.info(f"[Train Dataset][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            dataset = ConcatDataset(dataset)
            self.trainset = dataset
            Log.info(f"[Train Dataset][All]: ConcatDataset size={len(dataset)}")
            Log.info(f"")
        else:
            # Val and Test use sequential datasets.
            assert split in self.loader_opts, f"split={split} not in loader_opts"
            split_opts = self.dataset_opts.get(split)
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                ds_i = instantiate(v)
                if split == "test" and self.test_subset_indices is not None:
                    ds_i = Subset(ds_i, self.test_subset_indices)
                dataset.append(ds_i)
                dataset_type = "Val Dataset" if split == "val" else "Test Dataset"
                Log.info(f"[{dataset_type}][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            setattr(self, f"{split}sets", dataset)
            Log.info(f"")

        self._setup_splits.add(split)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self._setup_split("train")
            # Do not deserialize every validation PTH before the first train
            # step. val_dataloader() constructs it only when Lightning actually
            # enters validation.
        elif stage == "validate":
            self._setup_split("val")
        elif stage == "test":
            self._setup_split("test")

    def train_dataloader(self):
        # Backward compatibility for standalone diagnostics which call this
        # method without first attaching the datamodule to a Trainer.
        self._setup_split("train")
        if hasattr(self, "trainset"):
            use_mixed_batches = bool(self.loader_opts.train.get("source_mixed_batches", False))
            use_homogeneous_batches = bool(
                self.loader_opts.train.get("source_homogeneous_batches", False)
            )
            if use_mixed_batches and use_homogeneous_batches:
                raise ValueError(
                    "source_mixed_batches and source_homogeneous_batches cannot both be enabled"
                )
            if use_mixed_batches:
                batch_sampler = SourceMixedBatchSampler(
                    self.trainset,
                    batch_size=self.loader_opts.train.batch_size,
                    shuffle=bool(self.loader_opts.train.get("shuffle", True)),
                    drop_last=True,
                    seed=int(self.loader_opts.train.get("seed", 42)),
                )
                Log.info(
                    "[Train Loader] source-mixed batches enabled: "
                    f"batch_size={self.loader_opts.train.batch_size}, batches/rank={len(batch_sampler)}"
                )
                return DataLoader(
                    self.trainset,
                    batch_sampler=batch_sampler,
                    num_workers=self.loader_opts.train.num_workers,
                    persistent_workers=True and self.loader_opts.train.num_workers > 0,
                    collate_fn=collate_fn,
                    worker_init_fn=_worker_init_fn,
                )
            if use_homogeneous_batches:
                batch_sampler = SourceHomogeneousBatchSampler(
                    self.trainset,
                    batch_size=self.loader_opts.train.batch_size,
                    shuffle=bool(self.loader_opts.train.get("shuffle", True)),
                    drop_last=True,
                    seed=int(self.loader_opts.train.get("seed", 42)),
                )
                Log.info(
                    "[Train Loader] source-homogeneous batches enabled: "
                    f"batch_size={self.loader_opts.train.batch_size}, batches/rank={len(batch_sampler)}"
                )
                return DataLoader(
                    self.trainset,
                    batch_sampler=batch_sampler,
                    num_workers=self.loader_opts.train.num_workers,
                    persistent_workers=True and self.loader_opts.train.num_workers > 0,
                    collate_fn=collate_fn,
                    worker_init_fn=_worker_init_fn,
                )
            return DataLoader(
                self.trainset,
                shuffle=bool(self.loader_opts.train.get("shuffle", True)),
                num_workers=self.loader_opts.train.num_workers,
                persistent_workers=True and self.loader_opts.train.num_workers > 0,
                batch_size=self.loader_opts.train.batch_size,
                drop_last=True,
                collate_fn=collate_fn,
                worker_init_fn=_worker_init_fn,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        self._setup_split("val")
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=collate_fn,
                        worker_init_fn=_worker_init_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return None

    def test_dataloader(self):
        self._setup_split("test")
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=collate_fn,
                        worker_init_fn=_worker_init_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()

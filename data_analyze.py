import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from matplotlib import pyplot as plt
import mne
import typing as tp
import math

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self) -> torch.Tensor:
        
        #info = mne.io.read_info("SPM_CTF_MEG_example_faces1_3D.ds")
        #raw = mne.io.read_raw_ctf(mne.datasets.spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds")
        raw = mne.io.read_raw_ctf("SPM_CTF_MEG_example_faces1_3D.ds")
        layout = mne.find_layout(raw.info)
        indexes: tp.List[int] = []
        valid_indexes: tp.List[int] = []
        for meg_index, name in enumerate(raw.info.ch_names):
            print(f"meg_index: {meg_index}, name: {name}")
            #name = name.rsplit("-", 1)[0]
            try:
                indexes.append(layout.names.index(name))
            except ValueError:
                if name not in self._invalid_names:
                    print(f"Invalid name: {name}")
            else:
                valid_indexes.append(meg_index)

            positions = torch.full((len(raw.info.ch_names), 2), self.INVALID)
            x, y = layout.pos[indexes, :2].T
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            positions[valid_indexes, 0] = x
            positions[valid_indexes, 1] = y
            return positions

    def get_positions(self, batch):
        meg = batch
        B, C, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(len(batch)):
            rec_pos = self.get_recording_layout()
            positions[idx, :len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb



@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Logdir: {logdir}")
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
    raw = mne.io.read_raw_ctf(
    mne.datasets.spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds")
    kwargs = dict(
    show_axes=True,
    coord_frame='head',
    meg=['helmet', 'sensors'],
    eeg=False,
    surfaces=['head', 'white'],
)
    fig = mne.viz.plot_alignment(raw.info, meg=["helmet", "sensors", "ref"])
    mne.viz.set_3d_title(figure=fig, title="CTF 275")
   
    position_getter = PositionGetter()
    embedding = FourierEmb()
    batch_meg = train_set[0][0].unsqueeze(0) # (B, C, T)
    positions = position_getter.get_positions(batch_meg)
    embedding = embedding(positions)
    print(f"Positions: {positions.shape}, Embedding: {embedding.shape}")
    

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    X, y, subject_idxs = train_set[20000]
    waveform=X
    print("Shape of waveform: {}".format(waveform.size()))

    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.show()
    
    print(f"X.shape: {X.shape}, y: {y}, subject_idxs: {subject_idxs}")
    # X.shape: torch.Size([271, 281])なのは、(channel, time)の順番であることを示している. 280は-100msから1300msまでのデータで、サンプリングレートは200Hzである
    print(f"X type: {type(X)}, y type: {type(y)}, subject_idxs type: {type(subject_idxs)}")
    #すべてtorch.Tensor型であることがわかる
   
    
if __name__ == "__main__":
    run()

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


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

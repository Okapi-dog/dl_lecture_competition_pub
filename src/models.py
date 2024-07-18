import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        reduced_time: int = 32,
        reduced_channel: int = 32,
        p_drop: float = 0.1,
        num_subjects: int = 4,
        time_dim: int = 281,
        num_img_emb: int = 512,
    ) -> None:
        super().__init__()
        self.num_subjects = num_subjects
        self.hid_dim = hid_dim
        #Head
        self.head = nn.Sequential(
            nn.Conv1d(in_channels , hid_dim, 3, padding="same"), #input channel = in_channels + num_subjects
        )
        
        #Blocks
        self.block1 = nn.Sequential(
            ConvBlock(num_subjects*time_dim, reduced_time*4, p_drop=p_drop),
            ConvBlock(reduced_time*4, reduced_time, p_drop=p_drop)
        )
        self.block2 = nn.Sequential(
            ConvBlock(hid_dim, reduced_channel*4, p_drop=p_drop),
            ConvBlock(reduced_channel*4, reduced_channel, p_drop=p_drop)
        )
        
        
        #End
        self.end = nn.Sequential(
            nn.Linear(reduced_time*reduced_channel, reduced_time*reduced_channel),
            nn.Linear(reduced_time*reduced_channel, 4*num_img_emb),
            nn.Linear(4*num_img_emb, num_img_emb),
        )

    def forward(self, X, subject_id: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        subject_id = subject_id.unsqueeze(-1).repeat(1, 1, self.hid_dim)  # (batch, num_subjects, channel)
        # Rest of the process
        X = self.head(X)  # (batch, channel, time)
        X = torch.einsum("bct,bsc->bstc", X, subject_id)  # (batch, subjects, time, channel)
        X = Rearrange("b s t c -> b (t s) c")(X)  # (batch , time * subjects, channel)

        # 畳み込みブロックでtimeを削減後、channelを削減
        X = self.block1(X)   # (time*subjects, reduced_time*subjects)
        X = Rearrange("b t c -> b c t")(X)  # (batch, channel, reduced_time)
        X = self.block2(X)   # (batch, reduced_time1 , reduced_time2)
        X = X.view(X.size(0), -1)  # (batch, reduced_time1 * reduced_time2)
        X = self.end(X)
        X = nn.BatchNorm1d(X.size(1))(X)  # バッチ正規化を追加
        X = nn.functional.normalize(X, p=2, dim=0)
        X = torch.sigmoid(X)
        return X


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
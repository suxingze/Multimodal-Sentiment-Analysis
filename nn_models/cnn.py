import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSubNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(CNNSubNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)  # (batch, out_channels)
        x = self.dropout(x)
        return x

class Fusion(nn.Module):
    def __init__(
        self,
        input_sizes,      # [text_dim, visual_dim, audio_dim]
        hidden_sizes,     # [text_hidden, visual_hidden, audio_hidden]
        dropouts,         # [text_dropout, visual_dropout, audio_dropout]
        post_fusion_dim,  # fusion_hidden
        output_size,
        fusion_type='concat'
    ):
        super(Fusion, self).__init__()
        assert len(input_sizes) == 3 and len(hidden_sizes) == 3 and len(dropouts) == 3, "input_sizes, hidden_sizes, dropouts must be lists of length 3"
        self.text_cnn = CNNSubNet(input_sizes[0], hidden_sizes[0], kernel_size=3, dropout=dropouts[0])
        self.visual_cnn = CNNSubNet(input_sizes[1], hidden_sizes[1], kernel_size=3, dropout=dropouts[1])
        self.audio_cnn = CNNSubNet(input_sizes[2], hidden_sizes[2], kernel_size=3, dropout=dropouts[2])
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            fusion_input_dim = hidden_sizes[0] + hidden_sizes[1] + hidden_sizes[2]
        elif fusion_type == 'sum' or fusion_type == 'mul':
            assert hidden_sizes[0] == hidden_sizes[1] == hidden_sizes[2], "For sum/mul fusion, hidden sizes must be equal"
            fusion_input_dim = hidden_sizes[0]
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.fc1 = nn.Linear(fusion_input_dim, post_fusion_dim)
        self.fc2 = nn.Linear(post_fusion_dim, output_size)
    
    def forward(self, audio_x, video_x, text_x, lengths_x):
        # audio_x, video_x, text_x: (batch, seq_len, feature_dim)
        # lengths_x: not used in this implementation, but kept for compatibility

        text_feat = self.text_cnn(text_x)
        visual_feat = self.visual_cnn(video_x)
        audio_feat = self.audio_cnn(audio_x)

        if self.fusion_type == 'concat':
            fused = torch.cat([text_feat, visual_feat, audio_feat], dim=1)
        elif self.fusion_type == 'sum':
            fused = text_feat + visual_feat + audio_feat
        elif self.fusion_type == 'mul':
            fused = text_feat * visual_feat * audio_feat
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        out = F.relu(self.fc1(fused))
        out = self.fc2(out)
        return out
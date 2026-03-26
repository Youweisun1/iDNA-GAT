import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from Transformer import *


class GraphCNN(nn.Module):
    def __init__(self, channels=32, r=4, num_node_features=3, aug_prob=0.5):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 16, heads=4, dropout=0.1)
        self.gat2 = GATConv(64, 32, heads=4, dropout=0.1)
        self.gat3 = GATConv(128, 32, heads=4, dropout=0.1)

        self.skip_proj1 = nn.Linear(64, 128)
        self.skip_proj2 = nn.Linear(128, 128)
        self.multi_scale_fusion = nn.Linear(640, 128)  # 输出统一为128维


        self.aug_prob = aug_prob
        self.noise_scale = 0.01
        self.scale_range = (0.9, 1.1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fusion_pre_bert = nn.Linear(32 * 41 + 128, 32 * 41)  # 注意：图特征维度是128，这里修正原代码注释错误（原注释32*4=128）

        self.trans = Encoder(1, 41, 4, 100, 400, 32)
        self.fusion = nn.Linear(1312, 1024)
        self.block1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x, graph_data=None):
        x_seq = x.permute(0, 2, 1)
        x_seq = self.conv1(x_seq)
        x_seq = self.conv2(x_seq)
        x_seq = self.conv3(x_seq)
        if graph_data is not None:
            x_graph, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
            x_graph1 = F.elu(self.gat1(x_graph, edge_index))
            x_graph2 = F.elu(self.gat2(x_graph1, edge_index)) + self.skip_proj1(x_graph1)
            x_graph3 = F.elu(self.gat3(x_graph2, edge_index)) + self.skip_proj2(x_graph2)
            mean_pool1, max_pool1 = global_mean_pool(x_graph1, batch), global_max_pool(x_graph1, batch)
            mean_pool2, max_pool2 = global_mean_pool(x_graph2, batch), global_max_pool(x_graph2, batch)
            mean_pool3, max_pool3 = global_mean_pool(x_graph3, batch), global_max_pool(x_graph3, batch)
            multi_scale_features = torch.cat([mean_pool1, max_pool1, mean_pool2, max_pool2, mean_pool3, max_pool3],
                                             dim=1)
            graph_feature = self.multi_scale_fusion(multi_scale_features)
            if self.training:
                if torch.rand(1).item() < self.aug_prob:
                    noise = torch.randn_like(graph_feature) * self.noise_scale
                    graph_feature = graph_feature + noise
                    scale = torch.empty(1).uniform_(*self.scale_range).to(graph_feature.device)
                    graph_feature = graph_feature * scale
                    graph_feature = F.dropout(graph_feature, p=0.1, training=True)
            x_seq_flat = x_seq.view(x_seq.size(0), -1)
            fused_features = torch.cat([x_seq_flat, graph_feature], dim=1)
            fused_features = self.fusion_pre_bert(fused_features)  # [batch, 1312]
            x_bert_input = fused_features.view(x_seq.size(0), 32, 41)
        else:
            x_bert_input = x_seq

        x_bert = self.trans(x_bert_input)
        self.attn = x_bert
        x_bert = x_bert.view(x_bert.size(0), -1)
        x = self.fusion(x_bert)
        return self.block1(x)

    def trainModel(self, x, graph_data):
        output = self.forward(x, graph_data)
        return self.block2(output)
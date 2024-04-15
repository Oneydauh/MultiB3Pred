import torch
from torch import nn
pre_norm = False
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv,TransformerConv,GATv2Conv,BatchNorm,GraphNorm,InstanceNorm, LayerNorm
from torch_geometric.nn.models import GIN, GAT, GraphSAGE
from torch_geometric.nn.pool import global_mean_pool
import pickle as pkl



class BnLiear(torch.nn.Module):

    def __init__(self, input_size, hidden_dim, dropout, activation):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.activation(self.bn(self.fc(x))))

class Feature(torch.nn.Module):

    def __init__(self, input_size, hidden_dim, dropout, n_layers):
        super().__init__()
        self.nn_list = nn.ModuleList()
        self.n_layers = n_layers
        self.in_layer = nn.Linear(input_size, hidden_dim)

        for _ in range(n_layers):
            self.nn_list.append(BnLiear(hidden_dim, hidden_dim, dropout=dropout,activation=F.relu))

    def forward(self, x):
        x = self.in_layer(x)
        for idx in range(self.n_layers):
            x = self.nn_list[idx](x) + x
        return x

class StandardNet(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, n_fc_layers,  dropout, feat_dim=1,  num_classes=1):
        super().__init__()
        self.symbol_emb = nn.Linear(input_size, hidden_dim)
        self.bond_emb=nn.Linear(19, hidden_dim)

        self.nn_list = nn.ModuleList()
        self.n_layers = n_layers
        for _ in range(n_layers):
            self.nn_list.append(GIN(hidden_dim, hidden_dim, 1, act = F.relu, dropout = dropout, norm = torch.nn.BatchNorm1d))

        #self.conv3 = GAT(hidden_dim, hidden_dim, 1, act=F.relu, dropout = dropout, norm = torch.nn.BatchNorm1d)
        # 处理molecule的整体性feature
        self.fc1_0 = Feature(feat_dim, hidden_dim, dropout = dropout, n_layers=n_fc_layers)
        self.fc1_1 = Feature(7, hidden_dim, dropout = dropout, n_layers=n_fc_layers)
        self.fc1_2 = Feature(1024, hidden_dim, dropout = dropout, n_layers=n_fc_layers)

        self.fc2 = Feature(hidden_dim * 4, hidden_dim, dropout = dropout, n_layers=n_fc_layers)
        #self.fc2 = Feature(hidden_dim + feat_dim + 7 + hidden_dim, 128, dropout = dropout, n_layers=n_fc_layers)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        #self.conv1 = nn.Conv1d(in_channels=2560, out_channels=128, kernel_size=1)

        '''
        encoder_layer = nn.TransformerEncoderLayer(input_size, 2, hidden_dim,dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        '''
        self.esmfc = nn.Linear(2560, hidden_dim)
        self.esmgnn = nn.Linear(1024, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*3+20, hidden_dim),  # 假设的隐藏层维度
            nn.ReLU(),
            nn.Dropout(),
            #nn.Linear(hidden_dim, 32),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        
        #x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x, edge_index, batch, edge_attr, esm = data.x, data.edge_index, data.batch, data.edge_attr, data.esmfold

        
        esm = torch.split(esm, 2560, dim=0)
        esm = torch.stack(esm)
        esm = self.esmfc(esm)
        #print(esm.shape)
        #pkl.dump({'esm':esm}, open('results/esmtsne.bin','wb'))
        x = self.fc3(esm)
        
        '''
        esm = esm.unsqueeze(1)
        esm = esm.permute(0, 2, 1)
        esm = self.conv1(esm)
        esm = esm.squeeze(-1)
        x = self.fc3(esm)
        '''
        '''
        esm = self.transformer_encoder(esm)
        esm = esm.squeeze(-1)
        esm = self.esmfc(esm)
        x = seed.fc3(esm)
        '''
        
        '''
        x = self.symbol_emb(x)
        edge_attr = self.bond_emb(edge_attr)

        for idx in range(self.n_layers):
            x = self.nn_list[idx](x, edge_index, edge_attr = edge_attr) + x
        #x = self.conv3(x, edge_index, edge_attr = edge_attr) + x
        feat = global_mean_pool(x, batch)
        #x = self.fc3(feat)

        
        b = len(data.y)
        feature = data.feature.view(b, -1)
        bc_feature = feature[:,:self.feat_dim]
        cf_feature = feature[:,self.feat_dim:-1024]
        fp_feature =  feature[:,-1024:]

        #feat1 = self.fc1_0(bc_feature)  # 对输入进行norm
        #feat2 = self.fc1_1(cf_feature)  # 对输入进行norm
        feat1 = bc_feature
        feat2 = cf_feature
        feat3 = self.fc1_2(fp_feature)  # 对输入进行norm

        feat = torch.cat([feat, feat1, feat2, feat3],dim = -1)
        #feat = self.fc2(feat)
        #x = self.fc3(feat)
        x = self.mlp(feat)
        '''
        
        '''
        feat = torch.cat([feat, esm], dim = -1)
        x = self.mlp(feat)
        '''
        if self.num_classes > 1:
            return x
        else:
            return x.flatten()

module = GATv2Conv

class ToplogyNet(torch.nn.Module):
    def __init__(self, input_size,  hidden_dim, n_fc_layers, n_layers, dropout, num_classes):
        super().__init__()
        self.feat_dim = input_size

        self.fc1_0 = Feature(self.feat_dim, hidden_dim, dropout = 0.1, n_layers=n_fc_layers)
        self.fc1_1 = Feature(7, hidden_dim, dropout = 0.5, n_layers=n_fc_layers)
        self.fc1_2 = Feature(1024, hidden_dim, dropout = 0.5, n_layers=n_fc_layers)
        
        self.nn_list = nn.ModuleList()
        self.n_layers = n_layers
        for _ in range(n_layers):
            self.nn_list.append(GIN(hidden_dim, hidden_dim, 1, act = F.relu, dropout = dropout, norm = torch.nn.BatchNorm1d))

        self.fc2 = Feature(hidden_dim * 3, hidden_dim, dropout = dropout, n_layers=n_fc_layers)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(x.size())
        #exit()
        bc_feature = x[:,:self.feat_dim]
        cf_feature = x[:,self.feat_dim:-1024]
        fp_feature =  x[:,-1024:]

        feat1 = self.fc1_0(bc_feature)  # 对输入进行norm
        feat2 = self.fc1_1(cf_feature)  # 对输入进行norm
        feat3 = self.fc1_2(fp_feature)  # 对输入进行norm

        x = torch.concat([feat1, feat2, feat3],dim = -1)
        x = self.fc2(x)
        for idx in range(self.n_layers):
            x = self.nn_list[idx](x, edge_index) + x
        x = global_mean_pool(x, data.batch)
        x = self.fc3(x)
        if self.num_classes > 1:
            return x
        else:
            return x.flatten()

class HeteroNet(nn.Module):
    
    def __init__(self, input_size,  hidden_dim, n_fc_layers, n_layers, dropout, feat_dim,  num_classes = 1):
        super().__init__()
        self.feat_dim = input_size
        self.num_classes = num_classes
        
        self.whole_model = StandardNet(input_size, hidden_dim, n_layers, n_fc_layers,  dropout, 5, num_classes = hidden_dim)
        self.top_model = ToplogyNet(feat_dim, hidden_dim, n_fc_layers, n_layers, dropout,  num_classes = hidden_dim)
        
        self.fc2 = Feature(hidden_dim * 2, hidden_dim, dropout = dropout, n_layers=n_fc_layers)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data, whole_graph):
        
        whole_feat = self.whole_model(whole_graph)
        #print(whole_feat.size())
        top_feat = self.top_model(data)
        #print(top_feat.size())
        x = F.relu(torch.concat([whole_feat, top_feat],dim = -1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x.flatten()
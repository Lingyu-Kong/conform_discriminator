import torch
import torch.nn as nn
import torch.optim as optim
from EGNN_model.ignn_layer import IGNN_Layer

class IGNN(nn.Module):
    """
    
    """
    def __init__(self,
                 device:torch.device,
                 lr:float,
                 num_nodes:int,
                 in_node_attr_dim:int,
                 node_attr_dim:int,
                 out_node_attr_dim:int,
                 egde_attr_dim:int,
                 message_dim:int,
                 global_layer_size:int,
                 num_layers:int=4,
                 activation:nn.Module=nn.SiLU(),
                 residual:bool=True,
                 attention:bool=True,
                 normalize:bool=False,
                 tanh:bool=False,
                 last_sigmoid:bool=False):
        super(IGNN,self).__init__()
        self.device=device
        self.num_nodes=num_nodes
        self.num_layers=num_layers
        self.node_attr_encoder=nn.Linear(in_node_attr_dim,node_attr_dim)
        self.node_attr_decoder=nn.Linear(node_attr_dim,out_node_attr_dim)
        self.module_list=nn.ModuleList()
        for i in range(num_layers):
            self.module_list.append(IGNN_Layer(device,
                                               node_attr_dim,
                                               egde_attr_dim,
                                               message_dim,
                                               node_attr_dim,
                                               activation,
                                               residual,
                                               attention,
                                               normalize,
                                               tanh))
        if not last_sigmoid:
            self.global_mlp=nn.Sequential(nn.Linear(num_nodes,global_layer_size),
                                        nn.ReLU(),
                                        nn.Linear(global_layer_size,global_layer_size),
                                        nn.ReLU(),
                                        nn.Linear(global_layer_size,1)).to(self.device)
        else:
            self.global_mlp=nn.Sequential(nn.Linear(num_nodes,global_layer_size),
                                        nn.ReLU(),
                                        nn.Linear(global_layer_size,global_layer_size),
                                        nn.ReLU(),
                                        nn.Linear(global_layer_size,1),
                                        nn.Sigmoid()).to(self.device)
        self.to(self.device)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)

    def forward(self,x,h,edge_index,edge_attr):
        """
        x: [batch_size*num_nodes,pos=3]
        h: [batch_size*num_nodes,in_node_attr_dim]
        edge_index: [2,batch_szie*num_edges]
        edge_attr: [batch_szie*num_edges,edge_attr_dim]

        return:
        h:[batch_size*num_nodes,out_node_attr_dim=1]
        global_r:[batch_size,1]
        """
        h=self.node_attr_encoder(h)
        for i in range(self.num_layers):
            h=self.module_list[i](x,h,edge_index,edge_attr)
        h=self.node_attr_decoder(h)
        h=h.view(-1,self.num_nodes)
        global_r=self.global_mlp(h)
        return global_r
    
    

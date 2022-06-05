from email import message
import torch
import torch.nn as nn

"""
h: [batch_size*num_nodes, node_attr_dim]
x: [batch_size*num_nodes, pos_dim=3]
source: [batch_size*num_edges, ?]
target: [batch_size*num_edges, ?]
edge_attr: [batch_size*num_edges, edge_attr_dim]
edge_index=[2,batch_size*num_edges]
"""

class IGNN_Layer(nn.Module):
    """
    singal message passing layer in IGNN
    """

    def __init__(self,
                 in_node_attr_dim:int,   ## dim(h_i)
                 edge_attr_dim:int,   ## dim(a_ij)
                 message_dim:int,     ## dim(m_ij)
                 out_node_attr_dim:int,  ## dim(h_i) (output of this layer)
                 activation:nn.Module=nn.SiLU(),
                 residual:bool=True,
                 attention:bool=True,
                 normalize:bool=False,
                 tanh:bool=False,):
        super(IGNN_Layer,self).__init__()
        self.residual=residual
        self.attention=attention
        self.normalize=normalize
        self.tanh=tanh
        self.epsilon=1e-8
        
        self.phi_e=nn.Sequential(
            nn.Linear(in_node_attr_dim*2+1+edge_attr_dim,message_dim),
            activation,
            nn.Linear(message_dim,message_dim),
            activation)

        self.phi_h=nn.Sequential(
            nn.Linear(in_node_attr_dim+message_dim,message_dim),
            activation,
            nn.Linear(message_dim,out_node_attr_dim))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(message_dim, 1),
                nn.Sigmoid())

    def phi_e_model_forward(self,source_h,target_h,radial,edge_attr):
        """
        m_ij <- phi_e(h_i,h_j,||x_i-x_j||^2,a_ij)
        """
        # print("source_h_shape:",source_h.shape)
        # print("target_h_shape:",target_h.shape)
        # print("radial_shape:",radial.shape)
        # print("edge_attr_shape:",edge_attr.shape)
        phi_e_input=torch.cat([source_h,target_h,radial,edge_attr],dim=1)
        phi_e_output=self.phi_e(phi_e_input)
        if self.attention:
            att_val=self.att_mlp(phi_e_output)
            message=phi_e_output*att_val
        else:
            message=phi_e_output
        return message

    def phi_h_model_forward(self,in_node_attr,message):
        """
        m_ij <- phi_h(h_i,m_ij)
        """
        phi_h_input=torch.cat([in_node_attr,message],dim=1)
        node_attr=self.phi_h(phi_h_input)
        if self.residual:
            node_attr=in_node_attr+node_attr
        return node_attr

    def get_radial(self,source_x,target_x):
        """
        get radial distance
        """
        radial=torch.norm(source_x-target_x,dim=1)
        return radial.unsqueeze(-1)

    def unsorted_segment_sum(self,data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def forward(self,x,h,edge_index,edge_attr):
        row=edge_index[0]
        col=edge_index[1]
        radial=self.get_radial(x[row],x[col])
        message=self.phi_e_model_forward(h[row],h[col],radial,edge_attr)
        message_sum=self.unsorted_segment_sum(message,row,h.size(0))
        node_attr=self.phi_h_model_forward(h,message_sum)
        return node_attr

from EGNN_model.ignn import IGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from utils.graph_utils import get_edges_batch


class EGNN_Discriminator():
    def __init__(self,
                 intrinsic_target_params,
                 intrinsic_model_params,
                 extrinsic_model_params,
                 num_nodes,
                 decay_steps,
                 decay_rate,
                 device):
        self.num_nodes = num_nodes
        self.device = device

        # intrinsic_target_model
        self.intrinsic_target_model = IGNN(**intrinsic_target_params)
        self.intrinsic_target_model.apply(weight_init)

        # intrinsic model
        self.intrinsic_model = IGNN(**intrinsic_model_params)
        self.intrinsic_model.apply(weight_init)

        # extrinsic model
        self.extrinsic_model = IGNN(**extrinsic_model_params)
        self.extrinsic_model.apply(weight_init)

        self.scheduler_ex = MultiStepLR(self.extrinsic_model.optimizer,milestones=[100,150,175,200],gamma=0.5)
        self.scheduler_in = MultiStepLR(self.intrinsic_model.optimizer,milestones=[100,150,175],gamma=0.5)

    def predict(self, conforms, batch_size):
        """
        conforms: [batch_size,num_atoms,3]
        x: [batch_size*num_atoms,3]

        return:
        intrinsic_predict  --> [batch_size,1]    error is intrinsic_reward
        extrinsic_predict  -->  [batch_size,1]   in [0,1]
        """
        x = conforms.view(-1, 3).to(self.device)
        h = torch.ones(x.shape[0], 1).to(self.device)
        edge_index, edge_attr = get_edges_batch(
            self.num_nodes, batch_size)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        intrinsic_predict = self.intrinsic_model(
            x, h, edge_index, edge_attr)  # [batch_size,1]
        extrinsic_predict = self.extrinsic_model(
            x, h, edge_index, edge_attr)  # [batch_size,1]

        return intrinsic_predict, extrinsic_predict

    def compute_loss_and_train(self, conforms, final_energy, batch_size):
        """
        conforms: [batch_size,num_atoms,3]
        final_energy: [batch_size,1]

        return:
        loss
        """
        conforms = conforms.to(self.device)
        final_energy = final_energy.to(self.device)
        intrinsic_predict, extrinsic_predict = self.predict(conforms, batch_size)

        # intrinsic loss and backward
        with torch.no_grad():
            x = conforms.view(-1, 3).to(self.device)
            h = torch.ones(x.shape[0], 1).to(self.device)
            edge_index, edge_attr = get_edges_batch(
                self.num_nodes, batch_size)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            intrinsic_target = self.intrinsic_target_model(x, h, edge_index, edge_attr)
        intrinsic_loss = F.mse_loss(intrinsic_predict, intrinsic_target)
        self.intrinsic_model.optimizer.zero_grad()
        intrinsic_loss.backward()
        self.intrinsic_model.optimizer.step()
        self.scheduler_in.step()

        # extrinsic loss and backward
        extrinsic_loss = F.mse_loss(extrinsic_predict, final_energy)
        self.extrinsic_model.optimizer.zero_grad()
        extrinsic_loss.backward()
        self.extrinsic_model.optimizer.step()
        self.scheduler_ex.step()

        intrinsic_reward=torch.abs((intrinsic_predict-intrinsic_target)).squeeze(-1).tolist()
        extrinsic_reward=(-extrinsic_predict).squeeze(-1).tolist()

        return intrinsic_loss.item(), extrinsic_loss.item(),intrinsic_reward, extrinsic_reward
        

    def filt(self, conforms, threshold,batch_size):
        """
        conforms: [batch_size,num_atoms,3]
        threshold: float

        return:
        conforms: [?,num_atoms,3]
        """
        conforms = conforms.to(self.device)
        _, extrinsic_predict = self.predict(conforms,batch_size)
        extrinsic = extrinsic_predict.view(-1)
        index = torch.arange(0, conforms.shape[0]).to(self.device)
        mask = index[extrinsic > threshold]
        conforms = conforms[mask, :]
        return conforms

    def load_model(self, path):
        self.intrinsic_model.load_state_dict(torch.load(path + 'intrinsic_model.pt'))
        self.intrinsic_target_model.load_state_dict(torch.load(path + 'intrinsic_target_model.pt'))
        self.extrinsic_model.load_state_dict(torch.load(path + 'extrinsic_model.pt'))

    def save_model(self, path):
        torch.save(self.intrinsic_model.state_dict(), path + 'intrinsic_model.pt')
        torch.save(self.intrinsic_target_model.state_dict(), path + 'intrinsic_target_model.pt')
        torch.save(self.extrinsic_model.state_dict(), path + 'extrinsic_model.pt')


class Intrinsic_Target(nn.Module):
    """
    input_size: num_atoms*3
    output_size: 1
    """

    def __init__(self,
                 device: torch.device,
                 input_size: int,
                 latent_size: int,
                 output_size: int,
                 num_layers: int = 4):
        super(Intrinsic_Target, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_size, latent_size))
        self.mlp.append(nn.ReLU())
        for i in range(num_layers):
            self.mlp.append(nn.Linear(latent_size, latent_size))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(latent_size, output_size))
        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        x: [batch_size,num_atoms,3]
        """
        # print(x.shape)
        x = x.to(self.device)
        x = x.view(-1, self.input_size)
        for i, module in enumerate(self.mlp):
            x = module(x)
        return x.detach()


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

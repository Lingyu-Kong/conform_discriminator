from model.ignn import IGNN
import torch
import torch.nn as nn
from utils.graph_utils import get_edges_batch


class Discriminator():
    def __init__(self,
                 intrinsic_target_params,
                 intrinsic_model_params,
                 score_model_params,
                 num_nodes,
                 batch_size,
                 device):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.device = device

        # intrinsic_target_model
        # self.intrinsic_target_model=Intrinsic_Target(**intrinsic_target_params)
        # self.intrinsic_target_model.apply(weight_init)
        self.intrinsic_target_model = IGNN(**intrinsic_target_params)
        self.intrinsic_target_model.apply(weight_init)

        # intrinsic model
        self.intrinsic_model = IGNN(**intrinsic_model_params)
        self.intrinsic_model.apply(weight_init)

        # score model
        self.score_model = IGNN(**score_model_params)
        self.score_model.apply(weight_init)

    def predict(self, conforms):
        """
        conforms: [batch_size,num_atoms,3]
        x: [batch_size*num_atoms,3]

        return:
        intrinsic_predict  --> [batch_size,1]    error is intrinsic_reward
        score_predict  -->  [batch_size,1]   in [0,1]
        """
        x = conforms.view(-1, 3).to(self.device)
        h = torch.ones(x.shape[0], 1).to(self.device)
        edge_index, edge_attr = get_edges_batch(
            self.num_nodes, self.batch_size)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        intrinsic_predict = self.intrinsic_model(
            x, h, edge_index, edge_attr)  # [batch_size,1]
        score_predict = self.score_model(
            x, h, edge_index, edge_attr)  # [batch_size,1]

        return intrinsic_predict, score_predict

    def compute_loss_and_train(self, conforms, potential):
        """
        conforms: [batch_size,num_atoms,3]
        potential: [batch_size,num_atoms]  -->  (-energy/steps) which can be modified

        return:
        loss
        """
        conforms = conforms.to(self.device)
        potential = potential.to(self.device)
        intrinsic_predict, score_predict = self.predict(conforms)
        with torch.no_grad():
            x = conforms.view(-1, 3).to(self.device)
            h = torch.ones(x.shape[0], 1).to(self.device)
            edge_index, edge_attr = get_edges_batch(
                self.num_nodes, self.batch_size)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            intrinsic_real = self.intrinsic_target_model(x, h, edge_index, edge_attr)
        intrinsic_reward = torch.square(intrinsic_real-intrinsic_predict)
        intrinsic_loss = torch.mean(intrinsic_reward)
        self.intrinsic_model.optimizer.zero_grad()
        intrinsic_loss.backward()
        self.intrinsic_model.optimizer.step()
        intrinsic_reward = intrinsic_reward.detach()
        reward = potential+intrinsic_reward
        # reward=potential
        score_loss = -torch.mean(score_predict*reward)
        self.score_model.optimizer.zero_grad()
        score_loss.backward()
        self.score_model.optimizer.step()
        return score_loss.item(), intrinsic_reward.mean().item(), potential.mean().item()

    def filt(self, conforms, threshold):
        """
        conforms: [batch_size,num_atoms,3]
        threshold: float

        return:
        conforms: [?,num_atoms,3]
        """
        conforms = conforms.to(self.device)
        _, score_predict = self.predict(conforms)
        score = score_predict.view(-1)
        index = torch.arange(0, conforms.shape[0]).to(self.device)
        mask = index[score > threshold]
        conforms = conforms[mask, :]
        return conforms


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

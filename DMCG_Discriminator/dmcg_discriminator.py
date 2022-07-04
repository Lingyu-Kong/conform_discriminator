import sys 
sys.path.append("..") 
from DMCG_model.dmcg_nn import GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR

class DMCG_Discriminator():
    def __init__(self,
                 intrinsic_model_params,
                 extrinsic_model_params,
                 num_nodes,
                 decay_steps,
                 decay_rate,
                 device):
        self.num_nodes = num_nodes
        self.device = device

        # intrinsic model
        self.intrinsic_model = GNN(**intrinsic_model_params)
        # self.intrinsic_model.apply(weight_init)

        # extrinsic model
        self.extrinsic_model = GNN(**extrinsic_model_params)
        # self.extrinsic_model.apply(weight_init)

        # self.scheduler_ex = MultiStepLR(self.extrinsic_model.optimizer,milestones=[100,150,175],gamma=decay_rate)
        # self.scheduler_in = MultiStepLR(self.intrinsic_model.optimizer,milestones=[100,150,175],gamma=decay_rate)
        self.scheduler_ex = StepLR(self.extrinsic_model.optimizer,step_size=decay_steps,gamma=decay_rate)
        self.scheduler_in = StepLR(self.intrinsic_model.optimizer,step_size=decay_steps,gamma=decay_rate)

    def predict(self, conforms):
        """
        conforms: [batch_size,num_atoms,3]

        return:
        intrinsic_predict  --> [batch_size,1]    error is intrinsic_reward
        extrinsic_predict  -->  [batch_size,1]   in [0,1]
        """
        intrinsic_predict=self.intrinsic_model(conforms)
        extrinsic_predict=self.extrinsic_model(conforms)
        return intrinsic_predict, extrinsic_predict

    def compute_loss_and_train(self, conforms, energy):
        """
        conforms: [batch_size,num_atoms,3]
        energy: [batch_size,1]

        return:
        loss
        """
        conforms = conforms.to(self.device)
        energy = energy.to(self.device)
        intrinsic_predict, extrinsic_predict = self.predict(conforms)

        ## extrinsic loss and backward
        extrinsic_loss = F.mse_loss(extrinsic_predict, energy)
        self.extrinsic_model.optimizer.zero_grad()
        extrinsic_loss.backward()
        self.extrinsic_model.optimizer.step()
        self.scheduler_ex.step()

        ## intrinsic loss and backward
        extrinsic_predict = extrinsic_predict.detach()
        intrinsic_loss = F.mse_loss(intrinsic_predict, extrinsic_predict-energy)
        self.intrinsic_model.optimizer.zero_grad()
        intrinsic_loss.backward()
        self.intrinsic_model.optimizer.step()
        self.scheduler_in.step()

        intrinsic_reward=intrinsic_predict.squeeze(-1).tolist()
        extrinsic_reward=extrinsic_predict.squeeze(-1).tolist()

        return intrinsic_loss.item(), extrinsic_loss.item(),intrinsic_reward, extrinsic_reward
        
    def load_model(self, path):
        self.intrinsic_model.load_state_dict(torch.load(path + 'intrinsic_model.pt'))
        self.extrinsic_model.load_state_dict(torch.load(path + 'extrinsic_model.pt'))

    def save_model(self, path):
        torch.save(self.intrinsic_model.state_dict(), path + 'intrinsic_model.pt')
        torch.save(self.extrinsic_model.state_dict(), path + 'extrinsic_model.pt')


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

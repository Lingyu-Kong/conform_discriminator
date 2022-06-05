import torch

class Sampler():
    def __init__(self,num_atoms,scale):
        self.num_atoms=num_atoms
        self.scale=scale

    def single_sample(self):
        pos=torch.rand(self.num_atoms,3)*(torch.rand(1).item()*self.scale)
        return pos.detach()
    
    def batch_sample(self,batch_size):
        pos_batch=torch.rand(batch_size,self.num_atoms,3)*(torch.rand(1).item()*self.scale)
        return pos_batch.detach()

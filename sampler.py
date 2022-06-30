import torch
from utils.graph_utils import get_edges_batch


def batch_filt(batch_data, batch_size,num_atoms,threshold):
    edge_index, _ = get_edges_batch(num_atoms, batch_data.shape[0])
    row = batch_data.view(-1, 3)[edge_index[0]]
    col = batch_data.view(-1, 3)[edge_index[1]]
    radial = torch.norm(row-col, dim=1)
    judge = radial > threshold
    judge.reshape(batch_size, -1)
    filt = []
    for i in range(batch_size):
        if False not in judge[i]:
            filt.append(i)
    filted_batch_size = len(filt)
    filt = torch.tensor(filt, dtype=torch.long)
    filted_batch_data = batch_data[filt]
    return filted_batch_data, filted_batch_size


class Sampler():
    def __init__(self, num_atoms, scale, threshold):
        self.num_atoms = num_atoms
        self.scale = scale
        self.threshold = threshold

    def single_sample(self):
        pos = torch.rand(self.num_atoms, 3)*(torch.rand(1).item()*self.scale)
        return pos.detach()

    def batch_sample(self, batch_size):
        pos_batch = torch.rand(batch_size, self.num_atoms, 3)
        scale_batch=(torch.rand(batch_size,self.num_atoms, 3)*self.scale)
        pos_batch = pos_batch*scale_batch
        pos_batch = pos_batch.detach()
        filted_batch_data, filted_batch_size = batch_filt(pos_batch, batch_size, self.num_atoms, self.threshold)
        return filted_batch_data, filted_batch_size

import torch

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edge_index = [rows, cols]
    return edge_index

def get_edges_batch(n_nodes, batch_size):
    edge_index = get_edges(n_nodes)
    edge_attr = torch.ones(len(edge_index[0]) * batch_size, 1)
    edge_index = [torch.LongTensor(edge_index[0]), torch.LongTensor(edge_index[1])]
    if batch_size == 1:
        return edge_index, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edge_index[0] + n_nodes * i)
            cols.append(edge_index[1] + n_nodes * i)
        edge_index = torch.cat((torch.cat(rows).unsqueeze(0), torch.cat(cols).unsqueeze(0)),dim=0)
    return edge_index, edge_attr
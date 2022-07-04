import torch

class Sampler():
    def __init__(self,num_atoms,pos_scale,threshold):
        self.num_atoms=num_atoms
        self.pos_scale=pos_scale
        self.threshold=threshold
    
    def single_sample(self):
        pos=torch.zeros(self.num_atoms,3)
        for i in range(self.num_atoms):
            if_continue=True
            while if_continue:
                new_pos=torch.rand(3)*2*self.pos_scale-self.pos_scale
                if_continue=False
                for j in range(i):
                    distance=torch.norm(new_pos-pos[j],p=2)
                    if distance<self.threshold:
                        if_continue=True
                        break
            pos[i,:]=new_pos
        return pos

    def batch_sample(self,batch_size):
        conforms=torch.zeros(batch_size,self.num_atoms,3)
        for i in range(batch_size):
            conforms[i,:,:]=self.single_sample()
            # print("sampled "+str(i)+"th")
        return conforms,batch_size
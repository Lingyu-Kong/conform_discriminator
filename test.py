from sampler import Sampler

sampler=Sampler(20,2,0.5)
pos_batch, filted_batch_size = sampler.batch_sample(30)
print(pos_batch)
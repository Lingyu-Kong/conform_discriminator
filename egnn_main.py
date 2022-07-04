from policy_sampler.sampler import Sampler
from EGNN_Discriminator.egnn_discriminator import EGNN_Discriminator
import argparse
import torch
import torch.nn as nn
from bfgs_env.bfgs_relax import batch_compute, batch_relax
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt

wandb.login()
wandb.init(project="crystall discriminator", entity="kly20")

parser = argparse.ArgumentParser()

parser.add_argument('--if_relax',type=bool,default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument("--decay_steps", type=int, default=20)
parser.add_argument("--decay_rate", type=float, default=0.5)
parser.add_argument('--pos_scale', type=int, default=2)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--num_atoms", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--in_node_attr_dim", type=int, default=1)
parser.add_argument("--node_attr_dim", type=int, default=128)
parser.add_argument("--out_node_attr_dim", type=int, default=1)
parser.add_argument("--edge_attr_dim", type=int, default=1)
parser.add_argument("--message_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=8)
parser.add_argument("--global_layer_size", type=int, default=128)
parser.add_argument("--residual", type=bool, default=True)
parser.add_argument("--attention", type=bool, default=True)
parser.add_argument("--normalize", type=bool, default=True)
parser.add_argument("--tanh", type=bool, default=True)

parser.add_argument("--target_latent_size", type=int, default=128)
parser.add_argument("--target_num_layers", type=int, default=4)
parser.add_argument("--target_out_put_size", type=int, default=1)

parser.add_argument("--max_relax_steps", type=int, default=200)
parser.add_argument("--threshold", type=float, default=1)
parser.add_argument("--potential_scale", type=int, default=100)

parser.add_argument("--cuda", action='store_true')
parser.add_argument("--performance_log", action='store_true')
parser.add_argument("--performance_log_size", type=int, default=128)

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

device = torch.device("cuda")

intrinsic_target_params={"device": device,
                          "lr": args.lr,
                          "num_nodes": args.num_atoms,
                          "in_node_attr_dim": args.in_node_attr_dim,
                          "node_attr_dim": args.node_attr_dim,
                          "out_node_attr_dim": args.out_node_attr_dim,
                          "egde_attr_dim": args.edge_attr_dim,
                          "message_dim": args.message_dim,
                          "global_layer_size": args.global_layer_size,
                          "num_layers": args.target_num_layers,
                          "activation": nn.SiLU(),
                          "residual": args.residual,
                          "attention": args.attention,
                          "normalize": args.normalize,
                          "tanh": args.tanh,
                          "last_sigmoid": False}

intrinsic_model_params = {"device": device,
                          "lr": args.lr,
                          "num_nodes": args.num_atoms,
                          "in_node_attr_dim": args.in_node_attr_dim,
                          "node_attr_dim": args.node_attr_dim,
                          "out_node_attr_dim": args.out_node_attr_dim,
                          "egde_attr_dim": args.edge_attr_dim,
                          "message_dim": args.message_dim,
                          "global_layer_size": args.global_layer_size,
                          "num_layers": args.num_layers,
                          "activation": nn.SiLU(),
                          "residual": args.residual,
                          "attention": args.attention,
                          "normalize": args.normalize,
                          "tanh": args.tanh,
                          "last_sigmoid": False}

extrinsic_model_params = {"device": device,
                      "lr": args.lr,
                      "num_nodes": args.num_atoms,
                      "in_node_attr_dim": args.in_node_attr_dim,
                      "node_attr_dim": args.node_attr_dim,
                      "out_node_attr_dim": args.out_node_attr_dim,
                      "egde_attr_dim": args.edge_attr_dim,
                      "message_dim": args.message_dim,
                      "global_layer_size": args.global_layer_size,
                      "num_layers": args.num_layers,
                      "activation": nn.SiLU(),
                      "residual": args.residual,
                      "attention": args.attention,
                      "normalize": args.normalize,
                      "tanh": args.tanh,
                      "last_sigmoid": False}


wandb.config = {
    "training_steps": args.num_steps,
    "batch_size": args.batch_size,
    "num_atoms": args.num_atoms,
    "pos_scale": args.pos_scale,
    "max_relax_steps": args.max_relax_steps}


if __name__ == "__egnn_main__":
    sampler = Sampler(args.num_atoms, args.pos_scale,args.threshold)
    discriminator = EGNN_Discriminator(intrinsic_target_params,
                                  intrinsic_model_params,
                                  extrinsic_model_params,
                                  args.num_atoms,
                                  args.decay_steps,
                                  args.decay_rate,
                                  device)

    wandb.watch(discriminator.intrinsic_target_model,log="all",log_freq=5)

    for i in range(args.num_steps):
        start_time = time.time()
        conforms,sampled_batch_size = sampler.batch_sample(args.batch_size)
        if  args.if_relax:
            _,relax_energy=batch_relax(conforms,args.max_relax_steps,sampled_batch_size)
        else:
            relax_energy=batch_compute(conforms,sampled_batch_size)
        relax_energy=torch.FloatTensor(relax_energy).unsqueeze(-1)
        intrinsic_loss,extrinsic_loss,intrinsic_reward,extrinsic_reward = discriminator.compute_loss_and_train(conforms, relax_energy,sampled_batch_size)
        end_time = time.time()

        wandb.log({"intrinsic_loss": intrinsic_loss,
                       "extrinsic_loss": extrinsic_loss,
                       "extrinsic reward": np.mean(extrinsic_reward).item(),
                       "intrinsic reward": np.mean(intrinsic_reward).item(), 
                       "sampled_batch_size": sampled_batch_size,})

        if i % 20  == 0:
            plt.plot((-relax_energy).tolist(),label="real energy")
            plt.plot(np.add(extrinsic_reward, intrinsic_reward).tolist(),label="prediction")
            wandb.log({"chart"+i.__str__():plt})
            # ucb=np.sum([intrinsic_reward,extrinsic_reward],axis=0).tolist())
            # relax_energy=relax_energy.tolist()
            # data = [[x, y] for (x, y) in zip(ucb, relax_energy)]
            # table = wandb.Table(data=data, columns=["x", "y"])
            # wandb.log({"plot_id "+i.__str__(): wandb.plot.scatter(table, "x", "y", title="Energy vs Reward Scatter Plot")})

        print("=======================================================================")
        print("step ", i, " : finished,    time cost : ",
              end_time-start_time, "s")
        print("=======================================================================")

    ## save the model
    discriminator.save_model("./model_save/multistep1_")

    # # 最后的测试
    # test_conforms = sampler.batch_sample(args.performance_log_size)
    # test_batch_steps, test_batch_energy = batch_relax(
    #     test_conforms, args.max_relax_steps, test_conforms.shape[0])
    # data = [[x, y] for (x, y) in zip(test_batch_steps, test_batch_energy)]
    # table = wandb.Table(data=data, columns=["x", "y"])
    # wandb.log({"test_plot: uniform_random": wandb.plot.scatter(
    #     table, "x", "y", title="Energy vs Steps Scatter Plot")})
    # filted_conforms = discriminator.filt(test_conforms, args.threshold)
    # test_batch_steps, test_batch_energy = batch_relax(
    #     filted_conforms, args.max_relax_steps, filted_conforms.shape[0])
    # data = [[x, y] for (x, y) in zip(test_batch_steps, test_batch_energy)]
    # table = wandb.Table(data=data, columns=["x", "y"])
    # wandb.log({"test_plot: uniform_filt": wandb.plot.scatter(
    #     table, "x", "y", title="Energy vs Steps Scatter Plot")})

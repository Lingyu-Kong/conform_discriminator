import sys 
sys.path.append("..") 
from policy_sampler.sampler import Sampler
from DMCG_Discriminator.dmcg_discriminator import DMCG_Discriminator
import argparse
import torch
from bfgs_env.bfgs_relax import batch_compute, batch_relax
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt

wandb.login()
wandb.init(project="crystall discriminator", entity="kly20")

parser = argparse.ArgumentParser()
## shared args
parser.add_argument('--if_relax',type=bool,default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument("--decay_steps", type=int, default=350)
parser.add_argument("--decay_rate", type=float, default=0.9)
parser.add_argument('--pos_scale', type=int, default=2)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--num_atoms", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--threshold", type=float, default=1)
parser.add_argument("--max_relax_steps", type=int, default=200)
parser.add_argument("--cuda", type=bool, default=True)
## DMCG args
parser.add_argument("--num_message_passing_steps", type=int, default=6)
parser.add_argument("--mlp_hidden_size",type=int,default=512)
parser.add_argument("--mlp_layers",type=int,default=2)
parser.add_argument("--latent_size",type=int,default=256)
parser.add_argument("--use_layer_norm",type=bool,default=False)
parser.add_argument("--global_reducer",type=str,default="sum")
parser.add_argument("--node_reducer",type=str,default="sum")
parser.add_argument("--dropedge_rate",type=float,default=0.1)
parser.add_argument("--dropnode_rate",type=float,default=0.1)
parser.add_argument("--dropout",type=float,default=0.1)
parser.add_argument("--layernorm_before",type=bool,default=False)
parser.add_argument("--use_bn",type=bool,default=False)
parser.add_argument("--cycle",type=int,default=1)
parser.add_argument("--node_attn",type=bool,default=True)
parser.add_argument("--global_attn",type=bool,default=True)

args = parser.parse_args()

device=torch.device("cuda" if args.cuda else "cpu")

intrinsic_params={
    "device":device,
    "lr":args.lr,
    "num_atoms":args.num_atoms,
    "mlp_hidden_size":args.mlp_hidden_size,
    "mlp_layers":args.mlp_layers,
    "latent_size":args.latent_size,
    "use_layer_norm":args.use_layer_norm,
    "num_message_passing_steps":args.num_message_passing_steps,
    "global_reducer":args.global_reducer,
    "node_reducer":args.node_reducer,
    "dropedge_rate":args.dropedge_rate,
    "dropnode_rate":args.dropnode_rate,
    "dropout":args.dropout,
    "layernorm_before":args.layernorm_before,
    "use_bn":args.use_bn,
    "cycle":args.cycle,
    "node_attn":args.node_attn,
    "global_attn":args.global_attn
}

extrinsic_params={
    "device":device,
    "lr":args.lr,
    "num_atoms":args.num_atoms,
    "mlp_hidden_size":args.mlp_hidden_size,
    "mlp_layers":args.mlp_layers,
    "latent_size":args.latent_size,
    "use_layer_norm":args.use_layer_norm,
    "num_message_passing_steps":args.num_message_passing_steps,
    "global_reducer":args.global_reducer,
    "node_reducer":args.node_reducer,
    "dropedge_rate":args.dropedge_rate,
    "dropnode_rate":args.dropnode_rate,
    "dropout":args.dropout,
    "layernorm_before":args.layernorm_before,
    "use_bn":args.use_bn,
    "cycle":args.cycle,
    "node_attn":args.node_attn,
    "global_attn":args.global_attn
}


wandb.config = {
    "network structure":"DMCG",
    "num_atoms":args.num_atoms,
    "pos_scale": args.pos_scale,
    "training_steps": args.num_steps,
    "batch_size": args.batch_size,
}

if __name__ == "__main__":
    sampler=Sampler(args.num_atoms,args.pos_scale, args.threshold)
    discriminator=DMCG_Discriminator(intrinsic_params,
                                extrinsic_params,
                                args.num_atoms,
                                args.decay_steps,
                                args.decay_rate,
                                device)

    for i in range(args.num_steps):
        start_time=time.time()
        conforms,sampled_batch_size=sampler.batch_sample(args.batch_size)
        if args.if_relax:
            _,energy=batch_relax(conforms,args.max_relax_steps,sampled_batch_size)
        else:
            energy=batch_compute(conforms,sampled_batch_size)
        energy=torch.FloatTensor(energy).unsqueeze(-1)
        intrinsic_loss,extrinsic_loss,intrinsic_rewards,extrinsic_rewards = discriminator.compute_loss_and_train(conforms, energy)
        end_time=time.time()
        wandb.log({"intrinsic_loss": intrinsic_loss,
                       "extrinsic_loss": extrinsic_loss,
                       "extrinsic reward": np.mean(extrinsic_rewards).item(),
                       "intrinsic reward": np.mean(intrinsic_rewards).item(), })
        if i%50==0:
            energy=energy.squeeze(-1).tolist()
            plt.figure()
            plt.plot(energy,label="real energy",color="red")
            plt.plot(extrinsic_rewards,label="prediction",color="blue")
            wandb.log({"chart_ex_and_real"+i.__str__():plt})
            plt.figure()
            plt.plot(np.subtract(extrinsic_rewards,energy).tolist(),label="real res",color="red")
            plt.plot(intrinsic_rewards,label="prediction",color="blue")
            wandb.log({"chart_in_and_res"+i.__str__():plt})

            print("=======================================================================")
            print("step ", i, " : finished,    time cost : ",
                end_time-start_time, "s")
            print("=======================================================================")

        discriminator.save_model("./dmcg_model_save/multistep_")
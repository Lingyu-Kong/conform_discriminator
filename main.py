from faulthandler import disable
from sampler import Sampler
from discriminator import Discriminator
import argparse
import torch
import torch.nn as nn
from bfgs_relax import get_conforms_potential, batch_relax
import wandb
import time

wandb.login()
wandb.init(project="crystall discriminator", entity="kly20")

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--pos_scale', type=int, default=10)
parser.add_argument("--num_steps", type=int, default=101)
parser.add_argument("--num_atoms", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--in_node_attr_dim", type=int, default=1)
parser.add_argument("--node_attr_dim", type=int, default=128)
parser.add_argument("--out_node_attr_dim", type=int, default=1)
parser.add_argument("--edge_attr_dim", type=int, default=1)
parser.add_argument("--message_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=8)
parser.add_argument("--global_layer_size", type=int, default=64)
parser.add_argument("--residual", type=bool, default=True)
parser.add_argument("--attention", type=bool, default=True)
parser.add_argument("--normalize", type=bool, default=True)
parser.add_argument("--tanh", type=bool, default=False)

parser.add_argument("--target_latent_size", type=int, default=128)
parser.add_argument("--target_num_layers", type=int, default=4)
parser.add_argument("--target_out_put_size", type=int, default=1)

parser.add_argument("--max_relax_steps", type=int, default=1500)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--potential_scale", type=int, default=100)

parser.add_argument("--cuda", action='store_true')
parser.add_argument("--performance_log", action='store_true')
parser.add_argument("--performance_log_size", type=int, default=128)

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

device = torch.device("cuda")

# intrinsic_target_params = {"device": device,
#                            "input_size": args.num_atoms*3,
#                            "latent_size": args.target_latent_size,
#                            "output_size": args.target_out_put_size,
#                            "num_layers": args.target_num_layers}

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

score_model_params = {"device": device,
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
                      "last_sigmoid": True}


wandb.config = {
    "training_steps": args.num_steps,
    "batch_size": args.batch_size,
    "num_atoms": args.num_atoms,
    "max_relax_steps": args.max_relax_steps}


def plot_smoother(batch_steps, batch_energy):
    energies = []
    steps = []
    for i in range(len(batch_steps)):
        if(batch_energy[i] <= 5):
            energies.append(batch_energy[i])
            steps.append(batch_steps[i])
    return steps, energies


if __name__ == "__main__":
    sampler = Sampler(args.num_atoms, args.pos_scale)
    discriminator = Discriminator(intrinsic_target_params,
                                  intrinsic_model_params,
                                  score_model_params,
                                  args.num_atoms,
                                  args.batch_size,
                                  device)

    # wandb.watch(discriminator.score_model,log="all",log_freq=1)

    for i in range(args.num_steps):
        start_time = time.time()
        conforms = sampler.batch_sample(args.batch_size)
        potentials = get_conforms_potential(
            conforms, args.max_relax_steps, args.batch_size, args.potential_scale)
        loss, intrinsic_reward, reward = discriminator.compute_loss_and_train(
            conforms, potentials)
        end_time = time.time()

        if i % 10 == 0:
            comforms = sampler.batch_sample(args.performance_log_size)
            filted_conforms = discriminator.filt(comforms, args.threshold)
            batch_steps, batch_energy = batch_relax(
                filted_conforms, args.max_relax_steps, filted_conforms.shape[0])
            batch_steps, batch_energy = plot_smoother(
                batch_steps, batch_energy)
            data = [[x, y] for (x, y) in zip(batch_steps, batch_energy)]
            table = wandb.Table(data=data, columns=["x", "y"])
            wandb.log({"plot_id "+i.__str__(): wandb.plot.scatter(table, "x", "y", title="Energy vs Steps Scatter Plot"),
                       "loss": loss,
                       "intrinsic_reward": intrinsic_reward,
                       "extrinsic reward": reward, })

        print("=======================================================================")
        print("step ", i, " : finished,    time cost : ",
              end_time-start_time, "s")
        print("loss : ", loss)
        print("=======================================================================")

    # 最后的测试
    test_conforms = sampler.batch_sample(args.performance_log_size)
    test_batch_steps, test_batch_energy = batch_relax(
        test_conforms, args.max_relax_steps, test_conforms.shape[0])
    data = [[x, y] for (x, y) in zip(test_batch_steps, test_batch_energy)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_plot: uniform_random": wandb.plot.scatter(
        table, "x", "y", title="Energy vs Steps Scatter Plot")})
    filted_conforms = discriminator.filt(comforms, args.threshold)
    test_batch_steps, test_batch_energy = batch_relax(
        filted_conforms, args.max_relax_steps, filted_conforms.shape[0])
    data = [[x, y] for (x, y) in zip(test_batch_steps, test_batch_energy)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({"test_plot: uniform_filt": wandb.plot.scatter(
        table, "x", "y", title="Energy vs Steps Scatter Plot")})

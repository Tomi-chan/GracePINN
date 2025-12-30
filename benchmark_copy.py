import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime, HeatND
from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
from src.pde.inverse import PoissonInv, HeatInv
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper

# It is recommended not to modify this example file.
# Please copy it as benchmark_xxx.py and make changes according to your own ideas.
pde_list = \
    [Heat2D_LongTime] + \
    [NS2D_LongTime] + \
    [Wave2D_LongTime] + \
    [GrayScottEquation]

# pde_list = \
#     [Burgers1D, Burgers2D] + \
#     [Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea] + \
#     [Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime] + \
#     [NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime] + \
#     [Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime] + \
#     [KuramotoSivashinskyEquation, GrayScottEquation] + \
#     [PoissonND, HeatND]

# pde_list += \
#     [(Burgers2D, {"datapath": "ref/burgers2d_1.dat", "icpath": ("ref/burgers2d_init_u_1.dat", "ref/burgers2d_init_v_1.dat")})] + \
#     [(Burgers2D, {"datapath": "ref/burgers2d_2.dat", "icpath": ("ref/burgers2d_init_u_2.dat", "ref/burgers2d_init_v_2.dat")})] + \
#     [(Burgers2D, {"datapath": "ref/burgers2d_3.dat", "icpath": ("ref/burgers2d_init_u_3.dat", "ref/burgers2d_init_v_3.dat")})] + \
#     [(Burgers2D, {"datapath": "ref/burgers2d_4.dat", "icpath": ("ref/burgers2d_init_u_4.dat", "ref/burgers2d_init_v_4.dat")})] + \
#     [(Poisson2D_Classic, {"scale": 2})] + \
#     [(Poisson2D_Classic, {"scale": 4})] + \
#     [(Poisson2D_Classic, {"scale": 8})] + \
#     [(Poisson2D_Classic, {"scale": 16})] + \
#     [(Heat2D_Multiscale, {"init_coef": (5 * np.pi, np.pi)})] + \
#     [(Heat2D_Multiscale, {"init_coef": (10 * np.pi, np.pi)})] + \
#     [(Heat2D_Multiscale, {"init_coef": (40 * np.pi, np.pi)})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a2.dat", "a": 2})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a8.dat", "a": 8})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a16.dat", "a": 16})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a32.dat", "a": 32})] + \
#     [(Wave1D, {"a": 2})] + \
#     [(Wave1D, {"a": 6})] + \
#     [(Wave1D, {"a": 8})] + \
#     [(Wave1D, {"a": 10})] + \
#     [(HeatND, {"dim": 4})] + \
#     [(HeatND, {"dim": 6})] + \
#     [(HeatND, {"dim": 8})] + \
#     [(HeatND, {"dim": 10})]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--device', type=str, default="0")  # set to "cpu" enables cpu training 
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--method', type=str, default="adam")

    command_args = parser.parse_args()

    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    for pde_config in pde_list:

        def get_model_dde():
            if isinstance(pde_config, tuple):
                pde = pde_config[0](**pde_config[1])
            else:
                pde = pde_config()

            # pde.training_points()
            valid = {"adam","gepinn","gracepinn","laaf","gaaf","multiadam","lra","ntk","lbfgs","rar"}
            if command_args.method not in valid:
                raise ValueError(f"Unknown --method={command_args.method}. Use one of: {sorted(valid)}")

            if command_args.method == "gepinn":
                pde.use_gepinn()
            # elif command_args.method == "gracepinn":
            #     percentiles = parse_float_pair(command_args.gracepinn_percentiles, (5.0, 95.0))
            #     weight_clip = parse_float_pair(command_args.gracepinn_weight_clip, (0.2, 0.8))
            #     time_dims = parse_int_list(command_args.gracepinn_time_dims)
            #     config = GracePINNConfig(
            #         total_iterations=max(command_args.iter, 1),
            #         k=command_args.gracepinn_k,
            #         alpha=command_args.gracepinn_alpha,
            #         sigma_scale=command_args.gracepinn_sigma_scale,
            #         percentiles=percentiles,
            #         weight_clip=weight_clip,
            #         time_dims=time_dims if time_dims else None,
            #     )
            #     pde.enable_gracepinn(GracePINNWeighting(config))

            # elif command_args.method == "gracepinn":
            #     percentiles = parse_float_pair(
            #         command_args.gracepinn_percentiles, (25.0, 75.0)
            #     )
            #     time_dims = parse_int_list(command_args.gracepinn_time_dims)
            #     config = GracePINNConfig(
            #         total_iterations=max(command_args.iter, 1),
            #         k=command_args.gracepinn_k,
            #         alpha=command_args.gracepinn_alpha,
            #         percentiles=percentiles,
            #         time_dims=time_dims if time_dims else None,
            #         radius=command_args.gracepinn_radius,
            #     )
            #     pde.enable_gracepinn(GracePINNWeighting(config))
            elif command_args.method == "gracepinn":
                pde.enable_gracepinn(total_iterations=command_args.iter)

            net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
            if command_args.method == "laaf":
                net = DNN_LAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], pde.input_dim, pde.output_dim)
            elif command_args.method == "gaaf":
                net = DNN_GAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], pde.input_dim, pde.output_dim)
            net = net.float()

            loss_weights = parse_loss_weight(command_args)
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)

            opt = torch.optim.Adam(net.parameters(), command_args.lr)
            if command_args.method == "multiadam":
                opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
            elif command_args.method == "lra":
                opt = LR_Adaptor(opt, loss_weights, pde.num_pde)
            elif command_args.method == "ntk":
                opt = LR_Adaptor_NTK(opt, loss_weights, pde)
            elif command_args.method == "lbfgs":
                opt = Adam_LBFGS(net.parameters(), switch_epoch=5000, adam_param={'lr':command_args.lr})

            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights)
            if command_args.method == "rar":
                model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})
            # the trainer calls model.train(**train_args)
            return model

        def get_model_others():
            model = None
            # create a model object which support .train() method, and param @model_save_path is required
            # create the object based on command_args and return it to be trained
            # schedule the task using trainer.add_task(get_model_other, {training args})
            return model

        callbacks = [
            TesterCallback(log_every=command_args.log_every),
            PlotCallback(log_every=command_args.plot_every, fast=True),
            LossCallback(verbose=True),
        ]
        # if command_args.method == "grace":
        #     callbacks.append(
        #         GraceCurriculumCallback(
        #             total_iterations=command_args.iter,
        #             alpha=command_args.grace_alpha,
        #             delta=command_args.grace_delta,
        #             radius=command_args.grace_radius,
        #             percentiles=(command_args.grace_clip_low, command_args.grace_clip_high),
        #             v_bounds=(command_args.grace_vmin, command_args.grace_vmax),
        #             k_neighbors=command_args.grace_knn,
        #             dump_debug=command_args.grace_debug,
        #         )
        #     )

        trainer.add_task(
            get_model_dde,
            {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": callbacks,
            },
        )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()

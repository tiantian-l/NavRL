import os
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN
):

    env.enable_render(True)
    env.eval()
    env.set_seed(seed)

    render_callback = RenderCallback(interval=2)
    
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,
            policy=policy,
            callback=render_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    # base_env.enable_render(not cfg.headless)
    env.enable_render(not cfg.headless)
    env.reset()
    
    done = trajs.get(("next", "done")) 
    first_done = torch.argmax(done.long(), dim=1).cpu() # idx of first done will be return for each trajs

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # log video
    info["recording"] = wandb.Video(
        render_callback.get_video_array(axes="t c h w"), 
        fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
        format="mp4"
    )

    # --- Velocity tracking plots (per env, first episode) ---
    try:
        # DEBUG: check what keys are available in trajs
        print("[DEBUG] trajs keys:", trajs.keys(include_nested=True))
        print("[DEBUG] trajs['next'] keys:", trajs["next"].keys(include_nested=True))

        vel_cmd_all = trajs[("next", "info", "vel_cmd")].cpu()    # (num_envs, T, 1, 3)
        drone_st_all = trajs[("next", "info", "drone_state")].cpu()  # (num_envs, T, 1, 13)
        vel_real_all = drone_st_all[..., 7:10]                      # (num_envs, T, 1, 3)
        num_envs = vel_cmd_all.shape[0]

        print(f"[DEBUG] vel_cmd_all shape: {vel_cmd_all.shape}")
        print(f"[DEBUG] vel_real_all shape: {vel_real_all.shape}")
        print(f"[DEBUG] first_done: {first_done}")

        for ei in range(num_envs):
            ep_len = first_done[ei].item() + 1
            v_cmd = vel_cmd_all[ei, :ep_len, 0, :]   # (ep_len, 3)
            v_real = vel_real_all[ei, :ep_len, 0, :]  # (ep_len, 3)

            print(f"[DEBUG] env{ei}: ep_len={ep_len}, v_cmd range=[{v_cmd.min():.4f}, {v_cmd.max():.4f}], v_real range=[{v_real.min():.4f}, {v_real.max():.4f}]")
            print(f"[DEBUG] env{ei}: v_cmd[:5]={v_cmd[:5].tolist()}")
            print(f"[DEBUG] env{ei}: v_real[:5]={v_real[:5].tolist()}")
            # check if all values are identical (clone issue)
            if ep_len > 1:
                print(f"[DEBUG] env{ei}: v_cmd all_same={torch.allclose(v_cmd[0], v_cmd[-1])}, v_real all_same={torch.allclose(v_real[0], v_real[-1])}")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(v_cmd[:, 0].numpy(), v_cmd[:, 1].numpy(),
                    color="tab:blue", linewidth=1.0, alpha=0.8, label="cmd")
            ax.plot(v_real[:, 0].numpy(), v_real[:, 1].numpy(),
                    color="tab:orange", linewidth=1.0, alpha=0.8, label="real")
            ax.set_xlabel("vx (m/s)")
            ax.set_ylabel("vy (m/s)")
            ax.set_title(f"Env {ei}  |  velocity trajectory ({ep_len} steps)")
            ax.legend(loc="upper right")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            info[f"eval/vel_tracking_env{ei}"] = wandb.Image(fig)
            plt.close(fig)

    except Exception as e:
        import traceback
        print(f"[eval] vel tracking plot skipped: {e}")
        traceback.print_exc()

    # --- Export nominal transition data for degradation model fitting ---
    # (Separate try block so plotting errors don't block data export)
    try:
        export_path = getattr(cfg, 'nominal_data_path', None)
        print(f"[eval] nominal_data_path = {export_path}")
        if export_path:
            vel_cmd_all = trajs[("next", "info", "vel_cmd")].cpu()       # (num_envs, T, 1, 3)
            drone_st_all = trajs[("next", "info", "drone_state")].cpu()  # (num_envs, T, 1, 13)
            vel_real_all = drone_st_all[..., 7:10]                       # (num_envs, T, 1, 3)
            num_envs = vel_cmd_all.shape[0]

            all_v_prev, all_u_prev, all_v_next = [], [], []
            for ei in range(num_envs):
                ep_len = first_done[ei].item() + 1
                if ep_len < 2:
                    print(f"[eval] env{ei}: ep_len={ep_len}, skipping (too short)")
                    continue
                v_cmd = vel_cmd_all[ei, :ep_len, 0, :]   # (ep_len, 3)
                v_real = vel_real_all[ei, :ep_len, 0, :]  # (ep_len, 3)
                # Transition tuples: (v_{t-1}, u_{t-1}, v_t)
                all_v_prev.append(v_real[:-1])
                all_u_prev.append(v_cmd[:-1])
                all_v_next.append(v_real[1:])
                print(f"[eval] env{ei}: ep_len={ep_len}, collected {ep_len-1} transitions")
            if all_v_prev:
                new_v_prev = torch.cat(all_v_prev, dim=0)
                new_u_prev = torch.cat(all_u_prev, dim=0)
                new_v_next = torch.cat(all_v_next, dim=0)
                # Append to existing file if it exists
                if os.path.exists(export_path):
                    existing = torch.load(export_path, map_location="cpu", weights_only=True)
                    new_v_prev = torch.cat([existing["v_prev"], new_v_prev], dim=0)
                    new_u_prev = torch.cat([existing["u_prev"], new_u_prev], dim=0)
                    new_v_next = torch.cat([existing["v_next"], new_v_next], dim=0)
                torch.save({
                    "v_prev": new_v_prev,
                    "u_prev": new_u_prev,
                    "v_next": new_v_next,
                }, export_path)
                print(f"[eval] Nominal transition data: {new_v_prev.shape[0]} total samples -> {export_path}")
            else:
                print("[eval] WARNING: no valid episodes found (all ep_len < 2)")
        else:
            print("[eval] nominal_data_path not set, skipping data export")
    except Exception as e:
        import traceback
        print(f"[eval] nominal data export failed: {e}")
        traceback.print_exc()

    env.train()
    # env.reset()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"


# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import minigrid
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

from tqdm import tqdm
from tensorboardX import SummaryWriter

from networks_jax import Network

from env import TILE_PIXELS, PartialAndTotalRecordVideo
from replay_buffer import ReplayBuffer
from gymnasium.envs.registration import register

from omegaconf import OmegaConf
import hydra

import matplotlib.pyplot as plt
import pickle

register(
    id="MiniGrid-SaltAndPepper-v0-custom",
    entry_point="env:SaltAndPepper",
    kwargs={"size": 15},
)


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    show_grid_lines,
    agent_view_size,
    show_walls_pov,
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                show_grid_lines=show_grid_lines,
                agent_view_size=agent_view_size,
                show_walls_pov=show_walls_pov,
                seed=seed,
            )
            env = PartialAndTotalRecordVideo(
                env,
                f"videos/{run_name}",
            )
        else:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                show_grid_lines=show_grid_lines,
                agent_view_size=agent_view_size,
                show_walls_pov=show_walls_pov,
                seed=seed,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.Autoreset(env)
        env.action_space.seed(seed)

        return env

    return thunk


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def log_metric(writer, metrics_dict, name, value, step, step_type="global_step"):
    writer.add_scalar(name, value, step)
    if "data" not in metrics_dict:
        metrics_dict["data"] = []
    metrics_dict["data"].append(
        {"metric": name, "step": step, "value": float(value), "step_type": step_type}
    )


def eval_env(cfg, envs):
    print("****Evaluating****")
    obs, _ = envs.reset(seed=cfg.seed)
    terminations = np.array([False])
    truncations = np.array([False])
    seed = cfg.seed

    hardcoded_actions = cfg.hardcoded_actions
    action_index = 0
    for global_step in tqdm(range(cfg.total_timesteps)):
        # ALGO LOGIC: put action logic here

        actions = jnp.array([hardcoded_actions[action_index]])
        action_index += 1
        action_index %= len(hardcoded_actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        if terminations or truncations:
            # key, new_key = jax.random.split(key)
            # seed = int(jax.random.randint(new_key, (), 0, 2**30))
            # rng = np.random.default_rng(np.asarray(new_key))
            next_obs, _ = envs.reset()
            terminations = np.array([False])
            truncations = np.array([False])
        else:

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # next_obs = np.expand_dims(obs["image"], axis=0)
        terminations = np.expand_dims(terminations, axis=0)
        truncations = np.expand_dims(truncations, axis=0)

        # plt.imshow(obs["image"], cmap="gray", vmin=0, vmax=255)
        # plt.savefig("previous_obs_image.png")
        # plt.close()

        # plt.imshow(next_obs["image"], cmap="gray", vmin=0, vmax=255)
        # plt.savefig("next_obs_image.png")
        # plt.close()


def train_env(cfg, envs, q_key, writer, run_name):
    metrics_dict = {}
    print("Default JAX device:", jax.devices()[0])
    print("All available devices:", jax.devices())

    obs, _ = envs.reset(seed=cfg.seed)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)

    terminations = np.array([False])
    truncations = np.array([False])
    q_network = Network(
        feature_dims=cfg.dense_features,
        action_dim=envs.action_space.n,
    )
    # plt.imshow(obs["image"], cmap="gray", vmin=0, vmax=255)
    # plt.savefig("reset_obs_image.png")
    # plt.close()

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(
            q_key,
            jnp.expand_dims(jnp.array(obs["image"]), 0),
        ),
        target_params=q_network.init(
            q_key,
            jnp.expand_dims(jnp.array(obs["image"]), 0),
        ),
        tx=optax.adam(learning_rate=cfg.learning_rate),
    )
    print(
        "params",
        sum(x.size for x in jax.tree.leaves(q_state.params)),
        "target_params",
        sum(x.size for x in jax.tree.leaves(q_state.target_params)),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, 1)
    )

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.observation_space["image"],
        envs.action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(
        q_state,
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    ):
        q_next_target = q_network.apply(
            q_state.target_params, next_observations
        )  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * cfg.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(q_pred.shape[0]), actions.squeeze()
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    for global_step in tqdm(range(cfg.total_timesteps)):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            cfg.start_e,
            cfg.end_e,
            cfg.exploration_fraction * cfg.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array([envs.action_space.sample() for _ in range(1)])
        else:
            q_values = q_network.apply(
                q_state.params,
                jnp.expand_dims(jnp.array(obs["image"]), 0),
            )
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        # TODO: Check if this adds an off by one error
        if terminations or truncations:
            # key, new_key = jax.random.split(key)
            # seed = int(jax.random.randint(new_key, (), 0, 2**30))
            # rng = np.random.default_rng(np.asarray(new_key))
            next_obs, _ = envs.reset()
            terminations = np.array([False])
            truncations = np.array([False])
        else:

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        log_metric(writer, metrics_dict, "reward_per_timestep", rewards, global_step)
        log_metric(writer, metrics_dict, "is_done", np.any(terminations), global_step)
        # next_obs = np.expand_dims(obs["image"], axis=0)
        terminations = np.expand_dims(terminations, axis=0)
        truncations = np.expand_dims(truncations, axis=0)

        # plt.imshow(obs["image"], cmap="gray", vmin=0, vmax=255)
        # plt.savefig("previous_obs_image.png")
        # plt.close()

        # plt.imshow(next_obs["image"], cmap="gray", vmin=0, vmax=255)
        # plt.savefig("next_obs_image.png")
        # plt.close()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if np.any(truncations) or np.any(terminations):
            log_metric(writer, metrics_dict, "epsilon", epsilon, global_step)
            log_metric(writer, metrics_dict, "charts/is_done", True, global_step)
            log_metric(
                writer,
                metrics_dict,
                "charts/success_rate",
                infos["episode"]["r"],
                global_step,
            )
            log_metric(
                writer,
                metrics_dict,
                "charts/average_episodic_reward",
                infos["episode"]["r"] / infos["episode"]["l"],
                envs.unwrapped.num_episodes,
                "episode",
            )

            log_metric(
                writer,
                metrics_dict,
                "charts/episodic_length",
                infos["episode"]["l"],
                global_step,
            )
            log_metric(
                writer,
                metrics_dict,
                "charts/episode_count",
                envs.unwrapped.num_episodes,
                global_step,
            )

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #         print(
            #             f"global_step={global_step}, episodic_return={info['episode']['r']}"
            #         )
            #         writer.add_scalar(
            #             "charts/episodic_return", info["episode"]["r"], global_step
            #         )
            #         writer.add_scalar(
            #             "charts/episodic_length", info["episode"]["l"], global_step
            #         )
        else:
            log_metric(writer, metrics_dict, "charts/is_done", False, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        rb.add(
            obs["image"],
            real_next_obs["image"],
            actions,
            rewards,
            terminations,
            infos,
        )
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                # perform a gradient-descent step

                loss, old_val, q_state = update(
                    q_state,
                    np.array(data.observations),
                    np.array(data.actions),
                    np.array(data.next_observations),
                    np.array(data.rewards).flatten(),
                    np.array(data.dones).flatten(),
                )

                if global_step % 100 == 0:
                    log_metric(
                        writer,
                        metrics_dict,
                        "losses/td_loss",
                        jax.device_get(loss),
                        global_step,
                    )
                    log_metric(
                        writer,
                        metrics_dict,
                        "losses/average_q_values",
                        jax.device_get(old_val).mean(),
                        global_step,
                    )
                    log_metric(
                        writer,
                        metrics_dict,
                        "losses/max_q_value",
                        jax.device_get(old_val).max(),
                        global_step,
                    )
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    log_metric(
                        writer,
                        metrics_dict,
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(
                        q_state.params, q_state.target_params, cfg.tau
                    )
                )

    if cfg.save_model:
        model_path = f"runs/{run_name}/{cfg.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            cfg.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            log_metric(
                writer, metrics_dict, "eval/episodic_return", episodic_return, idx
            )

        if cfg.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{cfg.env_id}-{cfg.exp_name}-seed{cfg.seed}"
            repo_id = f"{cfg.hf_entity}/{repo_name}" if cfg.hf_entity else repo_name
            push_to_hub(
                cfg,
                episodic_returns,
                repo_id,
                "DQN",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )
    metrics_path = f"runs/{run_name}/metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics_dict, f)
    print(f"Metrics saved to {metrics_path}")


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg):
    print("Hydra loaded config:")
    print(cfg)

    if cfg.dry_run:
        return
    assert cfg.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{cfg.env_id}__{cfg.exp_name}__seed_{cfg.seed}__{int(time.time())}__{cfg.experiment_description}__learning_rate_{cfg.learning_rate}__dense_features_{cfg.dense_features}__agent_view_size_{cfg.agent_view_size}"
    if cfg.track:
        import wandb
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
        if(cfg.track_offline):
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "online"
            wandb.login()
            
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("*", step_metric="global_step", step_sync=True)


    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = make_env(
        cfg.env_id,
        cfg.seed,
        0,
        cfg.capture_video,
        run_name,
        cfg.show_grid_lines,
        cfg.agent_view_size,
        cfg.show_walls_pov,
    )()
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    print(f"train: {cfg.train}, eval: {cfg.eval}")
    if cfg.eval:
        eval_env(cfg, envs)
    else:
        train_env(cfg, envs, q_key, writer, run_name)

    envs.close()
    writer.close()
    return envs


if __name__ == "__main__":
    main()

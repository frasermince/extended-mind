# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import json
import optax
from flax.training.train_state import TrainState

from tqdm import tqdm
from tensorboardX import SummaryWriter

from networks_jax import Network

from env import TILE_PIXELS, PathMode
from wrappers import PartialAndTotalRecordVideo
from replay_buffer import ReplayBuffer
from gymnasium.envs.registration import register

from omegaconf import OmegaConf
import hydra

import matplotlib.pyplot as plt
import pickle
import uuid
import pandas as pd

# Optional dependency: pyarrow for Parquet output
try:
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]
except Exception:
    pa = None
    pq = None

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
    show_optimal_path,
    path_mode,
    show_landmarks,
    nonstationary_path_decay_pixels,
    nonstationary_path_inclusion_pixels,
    nonstationary_path_decay_chance,
    nonstationary_visitations_before_path_appearance,
    nonstationary_steps_before_path_visible,
    nonstationary_only_optimal,
    tile_size,
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
                show_optimal_path=show_optimal_path,
                path_mode=path_mode,
                show_landmarks=show_landmarks,
                nonstationary_path_decay_pixels=nonstationary_path_decay_pixels,
                nonstationary_path_inclusion_pixels=nonstationary_path_inclusion_pixels,
                nonstationary_path_decay_chance=nonstationary_path_decay_chance,
                nonstationary_visitations_before_path_appearance=nonstationary_visitations_before_path_appearance,
                nonstationary_steps_before_path_visible=nonstationary_steps_before_path_visible,
                nonstationary_only_optimal=nonstationary_only_optimal,
                tile_size=tile_size,
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
                show_optimal_path=show_optimal_path,
                path_mode=path_mode,
                show_landmarks=show_landmarks,
                nonstationary_path_decay_pixels=nonstationary_path_decay_pixels,
                nonstationary_path_inclusion_pixels=nonstationary_path_inclusion_pixels,
                nonstationary_path_decay_chance=nonstationary_path_decay_chance,
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


def _get_metric_series(metrics_dict, metric_name):
    data = metrics_dict.get("data", [])
    return [item["value"] for item in data if item.get("metric") == metric_name]


def _get_last_metric_value(metrics_dict, metric_name, default_value=None):
    series = _get_metric_series(metrics_dict, metric_name)
    if len(series) == 0:
        return default_value
    return series[-1]


def write_parquet_result(cfg, run_name, metrics_dict, parquet_dir):
    """Write a single-row Parquet part file into a shared dataset directory.

    Each run calls this once at the end. Readers will scan the directory as one table.
    """
    if pa is None or pq is None:
        print("pyarrow not available; skipping Parquet result write.")
        return

    # Aggregate simple summaries used by plotting
    success_values = _get_metric_series(metrics_dict, "charts/success_rate")
    avg_ep_reward_values = _get_metric_series(
        metrics_dict, "charts/average_episodic_reward"
    )
    episode_lengths = _get_metric_series(metrics_dict, "charts/episodic_length")
    episode_counts = _get_metric_series(metrics_dict, "charts/episode_count")
    final_sps = _get_last_metric_value(metrics_dict, "charts/SPS", default_value=None)

    num_episodes = (
        int(episode_counts[-1]) if len(episode_counts) > 0 else len(success_values)
    )
    mean_success_rate = (
        float(np.mean(success_values)) if len(success_values) > 0 else None
    )
    mean_avg_episodic_reward = (
        float(np.mean(avg_ep_reward_values)) if len(avg_ep_reward_values) > 0 else None
    )
    mean_episodic_length = (
        float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else None
    )

    # Compose a single row
    row = {
        "timestamp_unix": float(time.time()),
        "date": time.strftime("%Y-%m-%d"),
        "run_name": run_name,
        "exp_name": cfg.exp_name,
        "exp_group_id": getattr(cfg, "exp_group_id", "default"),
        "seed": int(cfg.seed),
        "env_id": str(cfg.training.env_id),
        "learning_rate": float(cfg.training.learning_rate),
        "dense_features": str(list(cfg.training.dense_features)),
        "agent_view_size": int(cfg.agent_view_size),
        "path_mode": str(cfg.path_mode),
        "generate_optimal_path": bool(cfg.generate_optimal_path),
        "total_timesteps": int(cfg.training.total_timesteps),
        "num_episodes": int(num_episodes),
        "mean_success_rate": mean_success_rate,
        "mean_avg_episodic_reward": mean_avg_episodic_reward,
        "mean_episodic_length": mean_episodic_length,
        "final_SPS": float(final_sps) if final_sps is not None else None,
        "wandb_project_name": getattr(cfg, "wandb_project_name", None),
        "wandb_entity": getattr(cfg, "wandb_entity", None),
    }

    table = pa.Table.from_pylist([row])

    # Write to summary subdirectory under the hyperparameter-partitioned base dir
    summary_dir = os.path.join(parquet_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    part_path = os.path.join(summary_dir, f"part-{uuid.uuid4().hex}.parquet")
    pq.write_table(table, part_path, compression="zstd", compression_level=3)
    print(f"Parquet result saved to {part_path}")


def write_parquet_metrics(
    cfg,
    run_name,
    metrics_dict,
    metrics_parquet_dir,
    model_params_json,
    global_step,
    path_list,
):
    """Write all log_metric entries as a time-series Parquet part file.

    Each element in metrics_dict["data"] becomes one row with run metadata
    attached for efficient filtering across runs.
    """
    if pa is None or pq is None:
        print("pyarrow not available; skipping metrics Parquet write.")
        return

    data = metrics_dict.get("data", [])
    if not data:
        print("No metrics to write to metrics Parquet dataset.")
        return

    rows = []
    network_depth = len(cfg.training.dense_features)
    network_width = cfg.training.dense_features[0] if cfg.training.dense_features else 0
    optimal_path_available = bool(cfg.path_mode != PathMode.NONE)
    for item in data:
        rows.append(
            {
                # minimal run-level metadata for grouping
                "learning_rate": float(cfg.training.learning_rate),
                "network_depth": int(network_depth),
                "network_width": int(network_width),
                "seed": int(cfg.seed),
                "optimal_path_available": optimal_path_available,
                # metric payload
                "metric": str(item.get("metric")),
                "step": int(item.get("step", 0)),
                "value": float(item.get("value", 0.0)),
                # keep numeric values in `value` and reserve strings for `json_value`
                "json_value": None,
                "step_type": str(item.get("step_type", "global_step")),
            }
        )
    rows.append(
        {
            # include run-level metadata for consistent filtering
            "learning_rate": float(cfg.training.learning_rate),
            "network_depth": int(network_depth),
            "network_width": int(network_width),
            "seed": int(cfg.seed),
            "optimal_path_available": optimal_path_available,
            "metric": "model_params_json",
            "step": int(global_step),
            # keep numeric column null; store JSON string separately
            "value": None,
            "json_value": model_params_json,
            "step_type": "global_step",
        }
    )
    for step_idx, path in enumerate(path_list):
        rows.append(
            {
                "learning_rate": float(cfg.training.learning_rate),
                "network_depth": int(network_depth),
                "network_width": int(network_width),
                "seed": int(cfg.seed),
                "optimal_path_available": optimal_path_available,
                "metric": "path",
                "step": int(step_idx + 1),
                "value": None,
                "json_value": json.dumps(path),
                "step_type": "global_step",
            }
        )

    table = pa.Table.from_pylist(rows)

    # Write to metrics subdirectory under the hyperparameter-partitioned base dir
    metrics_dir = os.path.join(metrics_parquet_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    part_path = os.path.join(metrics_dir, f"part-{uuid.uuid4().hex}.parquet")
    pq.write_table(table, part_path, compression="zstd", compression_level=3)
    print(f"Metrics Parquet saved to {part_path}")


def eval_env(cfg, envs, path_list, model_params, action_mode, actions_list, seed):
    dict_path_list = [json.loads(p) if isinstance(p, str) else p for p in path_list]
    print("****Evaluating****")
    obs, _ = envs.reset(seed=cfg.seed)
    terminations = np.array([False])
    truncations = np.array([False])

    hardcoded_actions = cfg.eval_params.hardcoded_actions
    action_index = 0
    dense_features_count = len(model_params["params"]) - 1
    dense_features_size = model_params["params"]["Dense_1"]["kernel"].shape[0]
    dense_features = [dense_features_size for _ in range(dense_features_count)]

    q_network = Network(
        feature_dims=dense_features,
        action_dim=envs.action_space.n,
    )

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=model_params,
        target_params=model_params,
        tx=optax.adam(learning_rate=cfg.training.learning_rate),
    )
    q_network.apply = jax.jit(q_network.apply)

    if action_mode == "RECORDED_ACTIONS":
        timesteps = len(actions_list)
    elif path_list:
        timesteps = len(dict_path_list)
    else:
        timesteps = cfg.training.total_timesteps
    for global_step in tqdm(range(timesteps)):
        if action_mode == "RECORDED_ACTIONS":
            envs.unwrapped.path = set(tuple(x) for x in dict_path_list[global_step])
        # ALGO LOGIC: put action logic here

        if action_mode == "HARDCODED_ACTIONS":
            actions = jnp.array([hardcoded_actions[action_index]])
            action_index += 1
            action_index %= len(hardcoded_actions)
        elif action_mode == "FINAL_POLICY":
            q_values = q_network.apply(
                q_state.params,
                jnp.expand_dims(jnp.array(obs["image"]), 0),
            )
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        elif action_mode == "RECORDED_ACTIONS":
            actions = jnp.array(actions_list[global_step])

        # TRY NOT TO MODIFY: execute the game and log data.
        if terminations or truncations:
            # key, new_key = jax.random.split(key)
            # seed = int(jax.random.randint(new_key, (), 0, 2**30))
            # rng = np.random.default_rng(np.asarray(new_key))
            obs, _ = envs.reset()
            terminations = np.array([False])
            truncations = np.array([False])
        else:
            actions = jnp.array(actions_list[global_step])
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        terminations = np.expand_dims(terminations, axis=0)
        truncations = np.expand_dims(truncations, axis=0)


def train_env(
    cfg,
    envs,
    q_key,
    writer,
    run_name,
    runs_dir,
    parquet_dir,
):
    metrics_dict = {}
    print("Default JAX device:", jax.devices()[0])
    print("All available devices:", jax.devices())

    # Print configuration for optimal path visibility
    if cfg.path_mode != PathMode.NONE:
        print(f"Path generation enabled: {cfg.path_mode}")

    obs, _ = envs.reset(seed=cfg.seed)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)

    terminations = np.array([False])
    truncations = np.array([False])
    q_network = Network(
        feature_dims=cfg.training.dense_features,
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
        tx=optax.adam(learning_rate=cfg.training.learning_rate),
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
        cfg.training.buffer_size,
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
        next_q_value = rewards + (1 - dones) * cfg.training.gamma * q_next_target

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

    path_list = []
    for global_step in tqdm(range(cfg.training.total_timesteps)):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            cfg.training.start_e,
            cfg.training.end_e,
            cfg.training.exploration_fraction * cfg.training.total_timesteps,
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
        if terminations or truncations:
            # key, new_key = jax.random.split(key)
            # seed = int(jax.random.randint(new_key, (), 0, 2**30))
            # rng = np.random.default_rng(np.asarray(new_key))
            next_obs, _ = envs.reset()
            terminations = np.array([False])
            truncations = np.array([False])
            rewards = np.array([0])
            infos = {"episode": {"r": 0, "l": 0}}
        else:

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        path_list.append(list(envs.unwrapped.path))
        log_metric(writer, metrics_dict, "action", actions[0].item(), global_step)

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
                envs.unwrapped.num_episodes,
                "episode",
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
        if global_step > cfg.training.learning_starts:
            if global_step % cfg.training.train_frequency == 0:
                data = rb.sample(cfg.training.batch_size)
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
            if global_step % cfg.training.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(
                        q_state.params, q_state.target_params, cfg.training.tau
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
            cfg.training.env_id,
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

            repo_name = f"{cfg.training.env_id}-{cfg.exp_name}-seed{cfg.seed}"
            repo_id = f"{cfg.hf_entity}/{repo_name}" if cfg.hf_entity else repo_name
            push_to_hub(
                cfg,
                episodic_returns,
                repo_id,
                "DQN",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )
    print(f"Summary Parquet saving to {parquet_dir}")

    def jax_tree_to_py(tree):
        # Recursively convert JAX arrays to lists for JSON serialization
        if isinstance(tree, dict):
            return {k: jax_tree_to_py(v) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            return [jax_tree_to_py(v) for v in tree]
        elif hasattr(tree, "tolist"):
            return tree.tolist()
        else:
            return tree

    params_py = jax_tree_to_py(q_state.params)
    params_json = json.dumps(params_py)
    if cfg.path_mode == PathMode.SHORTEST_PATH:
        metrics_path = f"{runs_dir}/metrics_optimal_path.pkl"
    else:
        metrics_path = f"{runs_dir}/metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics_dict, f)
    print(f"Metrics saved to {metrics_path}")

    write_parquet_metrics(
        cfg, run_name, metrics_dict, parquet_dir, params_json, global_step, path_list
    )


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg):
    start_time = time.time()
    print("Hydra loaded config:")
    print(cfg)

    if cfg.dry_run:
        return

    # check that specified path mode is valid
    assert cfg.path_mode in [mode.value for mode in PathMode], "Invalid path mode"
    assert cfg.training.num_envs == 1, "vectorized envs are not supported at the moment"
    dense_features_str = "_".join(str(f) for f in cfg.training.dense_features)
    run_name = (
        f"{cfg.training.env_id}__{cfg.exp_name}__seed_{cfg.seed}__{int(time.time())}"
        f"__{cfg.experiment_description}__learning_rate_{cfg.training.learning_rate}"
        f"__dense_features_{dense_features_str}__agent_view_size_{cfg.agent_view_size}"
    )

    # Extract learning rate, network depth, and network width from cfg
    learning_rate_str = str(cfg.training.learning_rate)
    network_depth = len(cfg.training.dense_features)
    network_width = cfg.training.dense_features[0] if cfg.training.dense_features else 0
    if cfg.show_landmarks:
        path_mode_str = "LANDMARKS"
    else:
        path_mode_str = cfg.path_mode

    runs_dir = os.path.join(
        cfg.run_folder,
        f"path_mode_{path_mode_str}",
        f"learning_rate_{learning_rate_str}",
        f"network_depth_{network_depth}",
        f"network_width_{network_width}",
        f"seed_{cfg.seed}",
    )
    parquet_dir = os.path.join(
        cfg.parquet_folder,
        f"path_mode_{path_mode_str}",
        f"learning_rate_{learning_rate_str}",
        f"network_depth_{network_depth}",
        f"network_width_{network_width}",
        f"seed_{cfg.seed}",
    )
    print(f"Runs directory: {runs_dir}")
    print(f"Parquet directory: {parquet_dir}")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(parquet_dir, exist_ok=True)
    if cfg.track:
        import wandb

        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
        if cfg.track_offline:
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

    writer = SummaryWriter(f"{runs_dir}")
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
        cfg.training.env_id,
        cfg.seed,
        0,
        cfg.capture_video,
        run_name,
        cfg.render_options.show_grid_lines,
        cfg.agent_view_size,
        cfg.render_options.show_walls_pov,
        cfg.render_options.show_optimal_path,
        cfg.path_mode,
        cfg.show_landmarks,
        cfg.nonstationary_path_decay_pixels,
        cfg.nonstationary_path_inclusion_pixels,
        cfg.nonstationary_path_decay_chance,
        cfg.nonstationary_visitations_before_path_appearance,
        cfg.nonstationary_steps_before_path_visible,
        cfg.nonstationary_only_optimal,
        cfg.tile_size,
    )()
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    print(f"train: {cfg.train}, eval: {cfg.eval}")
    path_list = []
    model_params = []

    def py_tree_to_jax(tree):
        if isinstance(tree, dict):
            return {k: py_tree_to_jax(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            return jnp.array(tree)
        else:
            return tree

    if cfg.eval:
        with open(cfg.eval_params.parquet_path, "rb") as f:

            metrics_df = pq.read_table(f).to_pandas()
            path_list = metrics_df[metrics_df["metric"] == "path"][
                "json_value"
            ].tolist()
            model_params = json.loads(
                metrics_df[metrics_df["metric"] == "model_params_json"][
                    "json_value"
                ].values[0]
            )
            model_params = py_tree_to_jax(model_params)
            actions_list = metrics_df[metrics_df["metric"] == "action"][
                "value"
            ].tolist()
            seed = int(metrics_df["seed"].iloc[0])
        eval_env(
            cfg,
            envs,
            path_list,
            model_params,
            cfg.eval_params.action_mode,
            actions_list,
            seed,
        )
    else:
        train_env(
            cfg,
            envs,
            q_key,
            writer,
            run_name,
            runs_dir,
            parquet_dir,
        )

    envs.close()
    writer.close()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*50}")
    print(
        f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"{'='*50}")

    return envs


if __name__ == "__main__":
    main()

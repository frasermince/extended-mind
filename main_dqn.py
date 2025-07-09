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
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from networks_jax import Network

from env import TILE_PIXELS, PartialAndTotalRecordVideo, GrayscaleObservation
from gymnasium.envs.registration import register

register(
    id="MiniGrid-SaltAndPepper-v0-custom",
    entry_point="env:SaltAndPepper",
    kwargs={"size": 11},
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "SaltAndPepper"
    """the wandb's project name"""
    wandb_entity: str = "frasermince"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    experiment_description: str = "seen-goal"
    show_grid_lines: bool = False
    agent_view_size: int = 5
    # Algorithm specific arguments
    env_id: str = "MiniGrid-SaltAndPepper-v0-custom"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 16
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    eval: bool = False
    train: bool = True


def make_env(
    env_id, seed, idx, capture_video, run_name, show_grid_lines, agent_view_size
):
    def thunk():
        if capture_video and idx == 1:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                show_grid_lines=show_grid_lines,
                agent_view_size=agent_view_size,
            )
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=TILE_PIXELS)
            env = PartialAndTotalRecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % 50 == 0 or x == 1,
            )
            env = GrayscaleObservation(env)
        else:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                show_grid_lines=show_grid_lines,
                agent_view_size=agent_view_size,
            )
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
            env = GrayscaleObservation(env)
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


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__seed_{args.seed}__{int(time.time())}__{args.experiment_description}__learning_rate_{args.learning_rate}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = make_env(
        args.env_id,
        args.seed + 1,
        1,
        args.capture_video,
        run_name,
        args.show_grid_lines,
        args.agent_view_size,
    )()
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)

    def eval_env():
        print("****Evaluating****")
        obs, _ = envs.reset(seed=args.seed)
        terminations = np.array([False])
        truncations = np.array([False])
        seed = args.seed

        hardcoded_actions = [
            envs.unwrapped.actions.left,
            envs.unwrapped.actions.forward,
        ]
        action_index = 0
        for global_step in tqdm(range(args.total_timesteps)):
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

            import matplotlib.pyplot as plt

            plt.imshow(obs["image"], cmap="gray", vmin=0, vmax=255)
            plt.savefig("previous_obs_image.png")
            plt.close()

            plt.imshow(next_obs["image"], cmap="gray", vmin=0, vmax=255)
            plt.savefig("next_obs_image.png")
            plt.close()

    def train_env():

        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=args.seed)

        terminations = np.array([False])
        truncations = np.array([False])
        seed = args.seed
        q_network = Network(
            # obs_shape=envs.observation_space.shape,
            action_dim=envs.action_space.n,
        )
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
            tx=optax.adam(learning_rate=args.learning_rate),
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
            target_params=optax.incremental_update(
                q_state.params, q_state.target_params, 1
            )
        )

        rb = DictReplayBuffer(
            args.buffer_size,
            gym.spaces.Dict(
                {
                    "image": envs.observation_space["image"],
                }
            ),
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
            print("device", next_observations.device)
            q_next_target = q_network.apply(
                q_state.target_params, next_observations
            )  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

            def mse_loss(params):
                q_pred = q_network.apply(
                    params, observations
                )  # (batch_size, num_actions)
                q_pred = q_pred[
                    jnp.arange(q_pred.shape[0]), actions.squeeze()
                ]  # (batch_size,)
                return ((q_pred - next_q_value) ** 2).mean(), q_pred

            (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
                q_state.params
            )
            q_state = q_state.apply_gradients(grads=grads)
            return loss_value, q_pred, q_state

        for global_step in tqdm(range(args.total_timesteps)):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
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
            # next_obs = np.expand_dims(obs["image"], axis=0)
            terminations = np.expand_dims(terminations, axis=0)
            truncations = np.expand_dims(truncations, axis=0)

            # import matplotlib.pyplot as plt

            # plt.imshow(obs["image"], cmap="gray", vmin=0, vmax=255)
            # plt.savefig("previous_obs_image.png")
            # plt.close()

            # plt.imshow(next_obs["image"], cmap="gray", vmin=0, vmax=255)
            # plt.savefig("next_obs_image.png")
            # plt.close()

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if np.any(truncations) or np.any(terminations):
                writer.add_scalar("epsilon", epsilon, global_step)
                writer.add_scalar(
                    "charts/episodic_return", infos["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", infos["episode"]["l"], global_step
                )
                writer.add_scalar(
                    "charts/episode_count", envs.unwrapped.num_episodes, global_step
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

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()

            rb.add(
                {"image": obs["image"]},
                {"image": real_next_obs["image"]},
                actions,
                rewards,
                terminations,
                infos,
            )
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = rb.sample(args.batch_size)
                    # perform a gradient-descent step
                    loss, old_val, q_state = update(
                        q_state,
                        data.observations["image"].numpy(),
                        data.actions.numpy(),
                        data.next_observations["image"].numpy(),
                        data.rewards.flatten().numpy(),
                        data.dones.flatten().numpy(),
                    )

                    if global_step % 100 == 0:
                        writer.add_scalar(
                            "losses/td_loss", jax.device_get(loss), global_step
                        )
                        writer.add_scalar(
                            "losses/average_q_values",
                            jax.device_get(old_val).mean(),
                            global_step,
                        )
                        writer.add_scalar(
                            "losses/max_q_value",
                            jax.device_get(old_val).max(),
                            global_step,
                        )
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar(
                            "charts/SPS",
                            int(global_step / (time.time() - start_time)),
                            global_step,
                        )

                # update target network
                if global_step % args.target_network_frequency == 0:
                    q_state = q_state.replace(
                        target_params=optax.incremental_update(
                            q_state.params, q_state.target_params, args.tau
                        )
                    )

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.dqn_jax_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                epsilon=0.05,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = (
                    f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                )
                push_to_hub(
                    args,
                    episodic_returns,
                    repo_id,
                    "DQN",
                    f"runs/{run_name}",
                    f"videos/{run_name}-eval",
                )

    if args.eval:
        eval_env()
    else:
        train_env()

    envs.close()
    writer.close()

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import gymnasium as gym
from gymnasium.envs.registration import register
import hydra
import os
import time
import uuid
import json
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import pyarrow as pa  
import pyarrow.parquet as pq 

from main_dqn import _get_metric_series, _get_last_metric_value
from env import PathMode
from wrappers import PartialAndTotalRecordVideo

register(
    id="MiniGrid-SaltAndPepper-v0-custom",
    entry_point="env:SaltAndPepper",
    kwargs={"size": 15},
)

def agent_observation_transform(img_obs: jnp.ndarray, agent_pixel_view_edge_dim: int, obs_dims: tuple):
    start_idx = (obs_dims[0] - agent_pixel_view_edge_dim) // 2
    end_idx = start_idx + agent_pixel_view_edge_dim
    img_obs_cropped = img_obs[start_idx:end_idx, start_idx:end_idx, :]
    flattened_img_obs_cropped = img_obs_cropped.flatten().astype(jnp.float32) / 255.0
    return flattened_img_obs_cropped


class LinearQNetworkNumpy:
    def __init__(self, obs_dim: int, num_actions: int, seed: int = 42):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        key = jax.random.PRNGKey(seed)
        weight = nn.initializers.lecun_normal()(key, (obs_dim, num_actions))
        bias = jnp.zeros(num_actions)
        
        self.params = {
            'weight': weight,
            'bias': bias
        }
    
    def apply(self, x: jnp.ndarray):
        return jnp.dot(x, self.params['weight']) + self.params['bias']
    


@jax.jit
def closed_form_q_update(weight: jnp.ndarray, bias: jnp.ndarray, x: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, x_next: jnp.ndarray, done: jnp.bool_, discount: float, step_size: float):

    q_values = jnp.dot(x, weight) + bias  # get q values for the state
    q_curr = q_values[a]  # select the q value for the action taken
    q_next_values = jnp.dot(x_next, weight) + bias
    q_next_max = jnp.where(done, 0.0, jnp.max(q_next_values))  # if done, q_next_target is 0, otherwise it is the max q value of the next state
    td_error = (r + discount * q_next_max) - q_curr  # the classic td error
    
    # Compute gradients for weight and bias
    weight_gradient = jnp.outer(x, td_error)  # outer product gives us the gradient matrix
    bias_gradient = td_error  # bias gradient is just the td_error
    
    # Update parameters
    new_weight = weight + step_size * weight_gradient
    new_bias = bias + step_size * bias_gradient
    
    loss = 0.5 * jnp.square(td_error)  # return the squared td error, for record keeping
    
    return new_weight, new_bias, loss


class NumpyQLearningAgent:

    def __init__(self, env_obs_dims : tuple, num_actions: int, discount: float, step_size: float, seed: int, start_epsilon: float, end_epsilon: float,
     exploration_fraction: float, total_timesteps: int, agent_pixel_view_edge_dim: int):
        
        self.obs_dims = env_obs_dims
        if(env_obs_dims[0] != env_obs_dims[1]):
            raise ValueError(f"Image observation must be a square")
    
        if(agent_pixel_view_edge_dim > env_obs_dims[0]):
            raise ValueError(f"Target edge length ({agent_pixel_view_edge_dim}) cannot be larger than original size ({env_obs_dims[0]})")

        if((env_obs_dims[0] - agent_pixel_view_edge_dim) % 2 != 0):
            raise ValueError(f"Target edge length ({agent_pixel_view_edge_dim}) must be even")
            
        self.num_states = agent_pixel_view_edge_dim * agent_pixel_view_edge_dim * env_obs_dims[2]
        self.num_actions = num_actions
        self.discount = discount
        self.step_size = step_size
        self.seed = seed
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.agent_pixel_view_edge_dim = agent_pixel_view_edge_dim
        self.linear_q_network = LinearQNetworkNumpy(self.num_states, num_actions, seed)
        

    # epsilon greedy action selection
    def select_action(self, x: jnp.ndarray, key: jnp.ndarray, global_step: int): 
        x = self._agent_observation_transform(x)
        q_values = self.linear_q_network.apply(x)  # get the q values for the state / observation
        greedy_action = jnp.argmax(q_values)  # greedy action is one with the max q value
        random_action = jax.random.randint(key, (), 0, self.num_actions)  # random action
        
        # Calculate current epsilon using linear decay
        epsilon = linear_schedule(
            self.start_epsilon,
            self.end_epsilon,
            int(self.exploration_fraction * self.total_timesteps),
            global_step
        )
        
        p = jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)  # probability of choosing a random action
        action = jnp.where(p < epsilon, random_action, greedy_action)  # choose a random action with probability epsilon, otherwise choose the greedy action

        return int(action)

    def update_q_values(self, x: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, x_next: jnp.ndarray, done: jnp.bool_, discount: float):
        x = self._agent_observation_transform(x)
        x_next = self._agent_observation_transform(x_next)
        
        weight = self.linear_q_network.params['weight']
        bias = self.linear_q_network.params['bias']
        
        new_weight, new_bias, loss = closed_form_q_update(
            weight, bias, x, a, r, x_next, done, self.discount, self.step_size
        )
        
        self.linear_q_network.params = {
            'weight': new_weight,
            'bias': new_bias
        }
        
        return loss

    def _agent_observation_transform(self, img_obs: jnp.ndarray):
        return agent_observation_transform(img_obs, self.agent_pixel_view_edge_dim, self.obs_dims)


class LinearQNetworkFlax(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        q_values = nn.Dense(
            self.num_actions, 
            use_bias=True,
            kernel_init=nn.initializers.lecun_normal() # this is flax's default init, I am just making it explicit
        )(x)
        return q_values

@jax.jit
def q_update(agent_train_state: TrainState, x: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, x_next: jnp.ndarray, done: jnp.bool_, discount: float):
    
    def loss_fn(params: jnp.ndarray):
        
        q_values = agent_train_state.apply_fn(params, x) # get q values for the state
        q_curr = q_values[a] # select the q value for the action taken
        q_next_max = jnp.where(done, 0.0, jnp.max(agent_train_state.apply_fn(params, x_next))) # if done, q_next_target is 0, otherwise it is the max q value of the next state
        td_error = jax.lax.stop_gradient(r + discount * q_next_max) - q_curr # the classic td error
        return 0.5 * jnp.square(td_error) # return the squared td error

    grad_fn = jax.value_and_grad(loss_fn) # create the gradient update function
    loss, grads = grad_fn(agent_train_state.params) # compute loss and gradients
    new_agent_train_state = agent_train_state.apply_gradients(grads=grads) # update the agent q values based on gradients
    return new_agent_train_state, loss


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


class FlaxQLearningAgent:
    def __init__(self, env_obs_dims : tuple, num_actions: int, discount: float, step_size: float, seed: int, start_epsilon: float, end_epsilon: float,
     exploration_fraction: float, total_timesteps: int, agent_pixel_view_edge_dim: int):
        self.obs_dims = env_obs_dims
        if(env_obs_dims[0] != env_obs_dims[1]):
            raise ValueError(f"Image observation must be a square")
    
        if(agent_pixel_view_edge_dim > env_obs_dims[0]):
            raise ValueError(f"Target edge length ({agent_pixel_view_edge_dim}) cannot be larger than original size ({env_obs_dims[0]})")

        if((env_obs_dims[0] - agent_pixel_view_edge_dim) % 2 != 0):
            raise ValueError(f"Target edge length ({agent_pixel_view_edge_dim}) must be even")
            
        self.num_states = agent_pixel_view_edge_dim * agent_pixel_view_edge_dim * env_obs_dims[2]
        self.num_actions = num_actions
        self.discount = discount
        self.step_size = step_size
        self.seed = seed
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.agent_pixel_view_edge_dim = agent_pixel_view_edge_dim

        linear_q_network = LinearQNetworkFlax(num_actions)
        linear_q_network_params = linear_q_network.init(jax.random.PRNGKey(seed), jnp.zeros(( self.num_states,)))

        optimizer = optax.sgd(step_size)
        self.train_state = TrainState.create(
            apply_fn=linear_q_network.apply,
            params=linear_q_network_params,
            tx=optimizer
        )

    # epsilon greedy action selection
    def select_action(self, x: jnp.ndarray, key: jnp.ndarray, global_step: int): 
        x = self._agent_observation_transform(x)
        q_values = self.train_state.apply_fn(self.train_state.params, x) # get the q values for the state / observation
        greedy_action = jnp.argmax(q_values) # greedy action is one with the max q value
        random_action = jax.random.randint(key, (), 0, self.num_actions) # random action
        
        # Calculate current epsilon using linear decay
        epsilon = linear_schedule(
            self.start_epsilon,
            self.end_epsilon,
            int(self.exploration_fraction * self.total_timesteps),
            global_step
        )
        
        p = jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0) # probability of choosing a random action
        action = jnp.where(p < epsilon, random_action, greedy_action) # choose a random action with probability epsilon, otherwise choose the greedy action

        return int(action)

    def update_q_values(self, x: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, x_next: jnp.ndarray, done: jnp.bool_, discount: float):
        x = self._agent_observation_transform(x)
        x_next = self._agent_observation_transform(x_next)
        result = q_update(self.train_state, x, a, r, x_next, done, self.discount)
        self.train_state, loss = result
        return loss

    def _agent_observation_transform(self, img_obs: jnp.ndarray):
        return agent_observation_transform(img_obs, self.agent_pixel_view_edge_dim, self.obs_dims)


def write_parquet_metrics(
    cfg,
    run_name,
    metrics_dict,
    metrics_parquet_dir,
    model_params_json,
    global_step,
    path_list,
    part_number=0,
):
    """Write all log_metric entries as a time-series Parquet part file.

    Each element in metrics_dict["data"] becomes one row with run metadata
    attached for efficient filtering across runs.
    """
    data = metrics_dict.get("data", [])
    if not data:
        print("No metrics to write to metrics Parquet dataset.")
        return

    rows = []
    step_size = float(cfg.training.step_size)
    agent_pixel_view_edge_dim = int(cfg.training.agent_pixel_view_edge_dim)
    optimal_path_available = bool(cfg.path_mode != "NONE")
    
    for item in data:
        rows.append(
            {
                # minimal run-level metadata for grouping
                "step_size": step_size,
                "agent_pixel_view_edge_dim": agent_pixel_view_edge_dim,
                "seed": int(cfg.seed),
                "optimal_path_available": optimal_path_available,
                "path_mode": str(cfg.path_mode),
                # metric payload
                "metric": str(item.get("metric")),
                "step": int(item.get("step", 0)),
                "value": float(item.get("value", 0.0)),
                # keep numeric values in `value` and reserve strings for `json_value`
                "json_value": None,
                "step_type": str(item.get("step_type", "global_step")),
            }
        )
    
    # Add model params if provided
    if model_params_json:
        rows.append(
            {
                # include run-level metadata for consistent filtering
                "step_size": step_size,
                "agent_pixel_view_edge_dim": agent_pixel_view_edge_dim,
                "seed": int(cfg.seed),
                "optimal_path_available": optimal_path_available,
                "path_mode": str(cfg.path_mode),
                "metric": "model_params_json",
                "step": int(global_step),
                # keep numeric column null; store JSON string separately
                "value": None,
                "json_value": model_params_json,
                "step_type": "global_step",
            }
        )
    
    # Add path_list entries if provided
    if path_list:
        for step_idx, path in enumerate(path_list):
            rows.append(
                {
                    "step_size": step_size,
                    "agent_pixel_view_edge_dim": agent_pixel_view_edge_dim,
                    "seed": int(cfg.seed),
                    "optimal_path_available": optimal_path_available,
                    "path_mode": str(cfg.path_mode),
                    "metric": "path",
                    "step": int(step_idx + 1),  # step_idx is 0-indexed, so add 1
                    "value": None,
                    "json_value": json.dumps(path),
                    "step_type": "global_step",
                }
            )

    metrics_dir = os.path.join(metrics_parquet_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    part_path = os.path.join(metrics_dir, f"part-{part_number}.parquet")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, part_path, compression="zstd", compression_level=3)
    print(f"Metrics Parquet saved to {part_path} ({len(rows)} rows)")
        
def compare_numpy_vs_flax_losses(cfg, num_comparisons: int = 5):
    '''
    This should show output like this to ptove that teh Flax and numpy implementations are consistent:

    Numpy loss: 2.612682819366455, Flax loss: 2.612682819366455
    Numpy loss: 1.1294746398925781, Flax loss: 1.1294746398925781
    Numpy loss: 0.444693922996521, Flax loss: 0.444693922996521
    Numpy loss: 0.9867634177207947, Flax loss: 0.9867634177207947
    Numpy loss: 0.8875203132629395, Flax loss: 0.8875203132629395
    Numpy mean loss: 1.212227
    Flax mean loss: 1.212227
    Mean difference: 0.000000
    Implementations are consistent (difference < 1e-05)
    '''


    print(cfg)

    env = gym.make(
        "MiniGrid-SaltAndPepper-v0-custom",
        render_mode="rgb_array",
        show_grid_lines=cfg.render_options.show_grid_lines,
        agent_view_size=cfg.agent_view_size,
        show_walls_pov=cfg.render_options.show_walls_pov,
        seed=cfg.seed,
        show_optimal_path=cfg.render_options.show_optimal_path,
        path_mode=cfg.path_mode,
        show_landmarks=cfg.show_landmarks,
        nonstationary_path_decay_pixels=cfg.nonstationary_path_decay_pixels,
        nonstationary_path_inclusion_pixels=cfg.nonstationary_path_inclusion_pixels,
        nonstationary_path_decay_chance=cfg.nonstationary_path_decay_chance,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.Autoreset(env)
    env.action_space.seed(cfg.seed)
    image_shape = env.observation_space["image"].shape

    print("Comparing Numpy vs Flax implementations...")
    
    image_shape = env.observation_space["image"].shape
    
    
    numpy_agent = NumpyQLearningAgent(
        env_obs_dims=image_shape, 
        num_actions=env.action_space.n, 
        discount=cfg.training.discount, 
        step_size=cfg.training.step_size, 
        seed=cfg.seed, 
        start_epsilon=cfg.training.start_epsilon,
        end_epsilon=cfg.training.end_epsilon,
        exploration_fraction=cfg.training.exploration_fraction,
        total_timesteps=cfg.training.total_timesteps,
        agent_pixel_view_edge_dim=cfg.training.agent_pixel_view_edge_dim
    )
    
    flax_agent = FlaxQLearningAgent(
        env_obs_dims=image_shape, 
        num_actions=env.action_space.n, 
        discount=cfg.training.discount, 
        step_size=cfg.training.step_size, 
        seed=cfg.seed, 
        start_epsilon=cfg.training.start_epsilon,
        end_epsilon=cfg.training.end_epsilon,
        exploration_fraction=cfg.training.exploration_fraction,
        total_timesteps=cfg.training.total_timesteps,
        agent_pixel_view_edge_dim=cfg.training.agent_pixel_view_edge_dim
    )
    shared_key = jax.random.PRNGKey(cfg.seed)
    num_states = cfg.training.agent_pixel_view_edge_dim * cfg.training.agent_pixel_view_edge_dim * image_shape[2]
    shared_weight = nn.initializers.lecun_normal()(shared_key, (num_states, env.action_space.n))
    shared_bias = jnp.zeros(env.action_space.n)
    
    numpy_agent.linear_q_network.params = {
        'weight': shared_weight,
        'bias': shared_bias
    }
    
    flax_agent.train_state.params['params']['Dense_0']['kernel'] = shared_weight
    flax_agent.train_state.params['params']['Dense_0']['bias'] = shared_bias
    
    original_numpy_params = {
        'weight': numpy_agent.linear_q_network.params['weight'].copy(),
        'bias': numpy_agent.linear_q_network.params['bias'].copy()
    }
    original_flax_params = flax_agent.train_state.params.copy()
    
    rng = jax.random.PRNGKey(cfg.seed)
    numpy_losses = []
    flax_losses = []
    differences = []
    
    
    for _ in range(num_comparisons):
        # Reset both agents to original parameters for fair comparison
        numpy_agent.linear_q_network.params = {
            'weight': original_numpy_params['weight'].copy(),
            'bias': original_numpy_params['bias'].copy()
        }
        flax_agent.train_state = flax_agent.train_state.replace(params=original_flax_params.copy())
        
        rng, key = jax.random.split(rng)
        obs, _ = env.reset()
        
        rng, key = jax.random.split(rng)
        a = jax.random.randint(key, (), 0, env.action_space.n) # random action
        
        obs_next, r, terminated, truncated, _ = env.step(int(a))
        # fake random reward (for variety, otherwise, most of the time will be zero in this env)
        r = jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0)
        done = terminated or truncated
        
        numpy_loss = numpy_agent.update_q_values(obs["image"], int(a), r, obs_next["image"], done, cfg.training.discount)
        flax_loss = flax_agent.update_q_values(obs["image"], int(a), r, obs_next["image"], done, cfg.training.discount)
        print(f"Numpy loss: {numpy_loss}, Flax loss: {flax_loss}")
        numpy_losses.append(float(numpy_loss))
        flax_losses.append(float(flax_loss))
        differences.append(abs(float(numpy_loss) - float(flax_loss)))
        

    mean_difference = jnp.mean(jnp.array(differences))
    print(f"Numpy mean loss: {jnp.mean(jnp.array(numpy_losses)):.6f}")
    print(f"Flax mean loss: {jnp.mean(jnp.array(flax_losses)):.6f}")
    print(f"Mean difference: {mean_difference:.6f}")
 
    
    tolerance = 1e-5
    if mean_difference < tolerance:
        print(f"Implementations are consistent (difference < {tolerance})")
    else:
        print(f"Implementations differ significantly (difference >= {tolerance})")
    
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

def py_tree_to_jax(tree):
    
    if isinstance(tree, dict):
        return {k: py_tree_to_jax(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return jnp.array(tree)
    else:
        return tree

def get_model_params_json(agent: NumpyQLearningAgent | FlaxQLearningAgent):
    if isinstance(agent, NumpyQLearningAgent):
        params_py = jax_tree_to_py(agent.linear_q_network.params)
    elif isinstance(agent, FlaxQLearningAgent):
        params_py = jax_tree_to_py(agent.train_state.params)
    else:
        raise ValueError(f"Unknown agent type")
    return json.dumps(params_py)

def train(agent: NumpyQLearningAgent | FlaxQLearningAgent, env: gym.Env, total_timesteps: int, writer, parquet_dir, cfg, result_writing_interval: int = 250000):

    metrics_dict = {}
    rng = jax.random.PRNGKey(agent.seed)
    episode_rewards = []
    episode_lengths = []
    step_rewards = []
    global_step = 0
    episode_count = 0
    start_time = time.time()
    path_list = []
    part_counter = 0  # Track part file number for sequential naming
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    avg_reward_per_global_step = 0
    run_name = os.path.basename(parquet_dir)

    unwrapped_env = env.unwrapped

    print(f"Grid width: {unwrapped_env.width}")
    print(f"Grid height: {unwrapped_env.height}")
    print(f"Grid size: {unwrapped_env.size}")
    print(f"Agent start position: {unwrapped_env.agent_pos}")
    print(f"Goal position: {unwrapped_env.goal_position}")
    print("Optimal policy expected steps: ", abs(unwrapped_env.agent_pos[0] - unwrapped_env.goal_position[0]) + abs(unwrapped_env.agent_pos[1] - unwrapped_env.goal_position[1]))
    print(f"Starting training for {total_timesteps} timesteps...")
    
    for global_step in range(1, total_timesteps + 1):
        rng, key = jax.random.split(rng)
        a = agent.select_action(obs["image"], key, global_step)
        obs_next, r, terminated, truncated, _ = env.step(jnp.array(a))
        done = terminated or truncated
        
        if cfg.path_mode == "VISITED_CELLS":
            path_list.append(list(unwrapped_env.path))
        
        log_metric(writer, metrics_dict, "action", a, global_step)
        log_metric(writer, metrics_dict, "reward_per_timestep", r, global_step)
        log_metric(writer, metrics_dict, "is_done", done, global_step)
        
        agent.update_q_values(obs["image"], a, r, obs_next["image"], done, agent.discount)

        obs = obs_next
        total_reward += r
        episode_steps += 1
        step_rewards.append(r)
        
        if done:
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_steps)
            episode_count += 1
            
            avg_reward_per_global_step = sum(step_rewards) / global_step
            
            # Log episode metrics
            log_metric(writer, metrics_dict, "charts/success_rate", total_reward, episode_count, "episode")
            log_metric(writer, metrics_dict, "charts/average_episodic_reward", total_reward / episode_steps, episode_count, "episode")
            log_metric(writer, metrics_dict, "charts/episodic_length", episode_steps, global_step)
            log_metric(writer, metrics_dict, "charts/episode_count", episode_count, global_step)
            
            print(f"Episode {episode_count} - Episode Length: {episode_steps} - Total Reward: {total_reward}, Episode Steps: {episode_steps}, Global Step: {global_step}, Avg Reward/Global Step: {avg_reward_per_global_step:.4f}")
            
            total_reward = 0
            episode_steps = 0
        
        # Log SPS (Steps Per Second) every 100 steps
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            log_metric(writer, metrics_dict, "charts/SPS", sps, global_step)
        
        # Progress indicator every 10000 steps
        if global_step % 10000 == 0:
            print(f"Training progress: {global_step}/{total_timesteps} steps completed")
        
        if global_step % result_writing_interval == 0:

            params_json = get_model_params_json(agent)
            write_parquet_metrics(
                cfg, run_name, metrics_dict, parquet_dir, params_json, global_step, path_list, part_counter
            )
            # Clear memory after writing
            metrics_dict = {"data": []}
            path_list = []
            part_counter += 1  # Increment for next part file


    min_episode_length = min(episode_lengths) if episode_lengths else 0
    print(f"Min episode length: {min_episode_length}")
    
    params_json = get_model_params_json(agent)
    write_parquet_metrics(
        cfg, run_name, metrics_dict, parquet_dir, params_json, global_step, path_list, part_counter
    )

def eval_env(cfg, envs, path_list, model_params, action_mode, actions_list, seed, video_dir=None):
    
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        envs = PartialAndTotalRecordVideo(envs, video_dir)
        print(f"Video recording enabled: {video_dir}")
    
    dict_path_list = [json.loads(p) if isinstance(p, str) else p for p in path_list]
    print("****Evaluating****")
    obs, _ = envs.reset(seed=seed)
    terminations = np.array([False])
    truncations = np.array([False])

    hardcoded_actions = cfg.eval_params.hardcoded_actions
    action_index = 0
    
    model_params = py_tree_to_jax(model_params)

    if 'weight' in model_params and 'bias' in model_params:
        # Numpy agent format
        weight = model_params['weight']
        bias = model_params['bias']
        
        image_shape = envs.observation_space["image"].shape
        agent_pixel_view_edge_dim = cfg.training.agent_pixel_view_edge_dim
        
        def apply_q_network(x):
            x_transformed = agent_observation_transform(x, agent_pixel_view_edge_dim, image_shape)
            return jnp.dot(x_transformed, weight) + bias
        
        apply_q_network = jax.jit(apply_q_network)
        
    elif 'params' in model_params:
        # Flax agent format
        linear_q_network = LinearQNetworkFlax(num_actions=envs.action_space.n)
        
        image_shape = envs.observation_space["image"].shape
        agent_pixel_view_edge_dim = cfg.training.agent_pixel_view_edge_dim
        
        q_state = TrainState.create(
            apply_fn=linear_q_network.apply,
            params=model_params,
            tx=optax.sgd(learning_rate=cfg.training.step_size),
        )
        linear_q_network.apply = jax.jit(linear_q_network.apply)
        
        def apply_q_network(x):
            x_transformed = agent_observation_transform(x, agent_pixel_view_edge_dim, image_shape)
            return q_state.apply_fn(q_state.params, x_transformed)
        
    else:
        raise ValueError(f"Unknown model_params format. Expected 'weight'/'bias' or 'params' keys, got: {list(model_params.keys())}")

    if action_mode == "RECORDED_ACTIONS":
        timesteps = len(actions_list)
    elif path_list:
        timesteps = len(dict_path_list)
    else:
        timesteps = cfg.training.total_timesteps
    
    for global_step in tqdm(range(timesteps)):
        if action_mode == "RECORDED_ACTIONS":
            envs.unwrapped.path = set(tuple(x) for x in dict_path_list[global_step])
        
        # todo : this needs to be tested
        if action_mode == "HARDCODED_ACTIONS":
            actions = jnp.array([hardcoded_actions[action_index]])
            action_index += 1
            if hardcoded_actions:
                action_index %= len(hardcoded_actions)

        
        elif action_mode == "FINAL_POLICY":
            q_values = apply_q_network(obs["image"])
            actions = jnp.array([jnp.argmax(q_values)])
            actions = jax.device_get(actions)
            
        elif action_mode == "RECORDED_ACTIONS":
            actions = jnp.array([actions_list[global_step]])

        if terminations or truncations:
            next_obs, _ = envs.reset()
            terminations = np.array([False])
            truncations = np.array([False])
        else:
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        terminations = np.expand_dims(terminations, axis=0)
        truncations = np.expand_dims(truncations, axis=0)
        
        obs = next_obs
    
    # Close the environment to ensure video is saved properly
    envs.close()

        
@hydra.main(config_path="../", config_name="linear_qlearning_config.yaml", version_base=None)
def main(cfg):
    start_time = time.time()
    print("Hydra loaded config:")
    print(cfg)


    step_size_str = str(cfg.training.step_size)
    
    # Build base path components
    path_components = [
        cfg.parquet_folder,
        f"agent_name_{cfg.agent_name}",
        f"path_mode_{cfg.path_mode}",
        f"step_size_{step_size_str}",
        f"agent_pixel_view_edge_dim_{cfg.training.agent_pixel_view_edge_dim}",
    ]
    
    # Only include nonstationary_path fields when path_mode is VISITED_CELLS
    if cfg.path_mode == "VISITED_CELLS":
        nonstationary_path_decay_pixels = cfg.nonstationary_path_decay_pixels
        nonstationary_path_decay_chance = cfg.nonstationary_path_decay_chance
        nonstationary_path_inclusion_pixels = cfg.nonstationary_path_inclusion_pixels
        decay_chance_str = f"{float(nonstationary_path_decay_chance):.2f}".rstrip('0').rstrip('.')
        path_components.extend([
            f"nonstationary_path_decay_pixels_{nonstationary_path_decay_pixels}",
            f"nonstationary_path_decay_chance_{decay_chance_str}",
            f"nonstationary_path_inclusion_pixels_{nonstationary_path_inclusion_pixels}",
        ])
    
    path_components.append(f"seed_{cfg.seed}")
    parquet_dir = os.path.join(*path_components)
    print(f"Output directory: {parquet_dir}")
    os.makedirs(parquet_dir, exist_ok=True)


    writer = SummaryWriter(f"{parquet_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    env = gym.make(
        "MiniGrid-SaltAndPepper-v0-custom",
        render_mode="rgb_array",
        show_grid_lines=cfg.render_options.show_grid_lines,
        agent_view_size=cfg.agent_view_size,
        show_walls_pov=cfg.render_options.show_walls_pov,
        seed=cfg.seed,
        show_optimal_path=cfg.render_options.show_optimal_path,
        path_mode=cfg.path_mode,
        show_landmarks=cfg.show_landmarks,
        nonstationary_path_decay_pixels=cfg.nonstationary_path_decay_pixels,
        nonstationary_path_inclusion_pixels=cfg.nonstationary_path_inclusion_pixels,
        nonstationary_path_decay_chance=cfg.nonstationary_path_decay_chance,
    )
    
    if cfg.capture_video:
        video_dir = os.path.join(parquet_dir, "videos", "train")
        os.makedirs(video_dir, exist_ok=True)
        env = PartialAndTotalRecordVideo(env, video_dir)
        print(f"Video recording enabled: {video_dir}")
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.Autoreset(env)
    env.action_space.seed(cfg.seed)
    image_shape = env.observation_space["image"].shape

    if cfg.training.q_network_type == "flax":
        agent = FlaxQLearningAgent(
            env_obs_dims=image_shape, 
            num_actions=env.action_space.n, 
            discount=cfg.training.discount, 
            step_size=cfg.training.step_size, 
            seed=cfg.seed, 
            start_epsilon=cfg.training.start_epsilon,
            end_epsilon=cfg.training.end_epsilon,
            exploration_fraction=cfg.training.exploration_fraction,
            total_timesteps=cfg.training.total_timesteps,
            agent_pixel_view_edge_dim= cfg.training.agent_pixel_view_edge_dim
        )
    elif cfg.training.q_network_type == "numpy":
        agent = NumpyQLearningAgent(
            env_obs_dims=image_shape, 
            num_actions=env.action_space.n, 
            discount=cfg.training.discount, 
            step_size=cfg.training.step_size, 
            seed=cfg.seed, 
            start_epsilon=cfg.training.start_epsilon,
            end_epsilon=cfg.training.end_epsilon,
            exploration_fraction=cfg.training.exploration_fraction,
            total_timesteps=cfg.training.total_timesteps,
            agent_pixel_view_edge_dim= cfg.training.agent_pixel_view_edge_dim
        )
    else:
        raise ValueError(f"Invalid q network type: {cfg.training.q_network_type}")

    if cfg.eval:
        if not cfg.eval_params.parquet_path:
            raise ValueError("eval_params.parquet_path must be set when eval=True")
        
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
            actions_list = metrics_df[metrics_df["metric"] == "action"][
                "value"
            ].tolist()
            seed = int(metrics_df["seed"].iloc[0])
        
        action_mode = cfg.eval_params.action_mode
        
        video_dir = None
        if cfg.capture_video:
            video_dir = os.path.join(parquet_dir, "videos", "eval", action_mode.lower())
        
        eval_env(
            cfg,
            env,
            path_list,
            model_params,
            action_mode,
            actions_list,
            seed,
            video_dir=video_dir,
        )
    else:
        train(agent, env, total_timesteps=cfg.training.total_timesteps, writer=writer, parquet_dir=parquet_dir, cfg=cfg)
    
    env.close()
    writer.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*50}")
    


if __name__ == "__main__":
    main()

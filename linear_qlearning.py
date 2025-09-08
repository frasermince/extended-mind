import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import gymnasium as gym
from gymnasium.envs.registration import register
import pickle
import hydra


register(
    id="MiniGrid-SaltAndPepper-v0-custom",
    entry_point="env:SaltAndPepper",
    kwargs={"size": 15},
)

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
        start_idx = (img_obs.shape[0] - self.agent_pixel_view_edge_dim) // 2
        end_idx = start_idx + self.agent_pixel_view_edge_dim
        img_obs_cropped = img_obs[start_idx:end_idx, start_idx:end_idx, :]
        flattened_img_obs_cropped = img_obs_cropped.flatten().astype(jnp.float32) / 255.0 
        return flattened_img_obs_cropped


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
        start_idx = (img_obs.shape[0] - self.agent_pixel_view_edge_dim) // 2
        end_idx = start_idx + self.agent_pixel_view_edge_dim
        img_obs_cropped = img_obs[start_idx:end_idx, start_idx:end_idx, :]
        flattened_img_obs_cropped = img_obs_cropped.flatten().astype(jnp.float32) / 255.0 
        return flattened_img_obs_cropped


        
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
        generate_optimal_path=cfg.training.generate_optimal_path,
        show_optimal_path=cfg.render_options.show_optimal_path,
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
    


def train(agent: NumpyQLearningAgent | FlaxQLearningAgent, env: gym.Env, total_timesteps: int, save_path="rewards.pkl"):

    rng = jax.random.PRNGKey(agent.seed)
    episode_rewards = []
    episode_lengths = []
    step_rewards = []
    global_step = 0
    episode_count = 0
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    avg_reward_per_global_step = 0

    unwrapped_env = env.unwrapped

    print(f"Grid width: {unwrapped_env.width}")
    print(f"Grid height: {unwrapped_env.height}")
    print(f"Grid size: {unwrapped_env.size}")
    print(f"Agent start position: {unwrapped_env.agent_pos}")
    print(f"Goal position: {unwrapped_env.goal_position}")
    print("Optimal policy expected steps: ", abs(unwrapped_env.agent_pos[0] - unwrapped_env.goal_position[0]) + abs(unwrapped_env.agent_pos[1] - unwrapped_env.goal_position[1]))
    
    for global_step in range(1, total_timesteps + 1):
        rng, key = jax.random.split(rng)
        a = agent.select_action(obs["image"], key, global_step)
        obs_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        
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
            
            print(f"Episode {episode_count} - Episode Length: {episode_steps} - Total Reward: {total_reward}, Episode Steps: {episode_steps}, Global Step: {global_step}, Avg Reward/Global Step: {avg_reward_per_global_step:.4f}")
            
            total_reward = 0
            episode_steps = 0
        
        if (global_step + 1) % 1000 == 0:
            training_data = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'step_rewards': step_rewards,
                'global_step': global_step + 1,
                'episode_count': episode_count,
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(training_data, f)

    min_episode_length = min(episode_lengths)
    print(f"Min episode length: {min_episode_length}")
    
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'step_rewards': step_rewards,
        'global_step': global_step + 1,
        'episode_count': episode_count,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(training_data, f)
        
@hydra.main(config_path=".", config_name="linear_qlearning_config.yaml", version_base=None)
def main(cfg):
    # compare_numpy_vs_flax_losses(cfg)
    print(cfg)

    env = gym.make(
        "MiniGrid-SaltAndPepper-v0-custom",
        render_mode="rgb_array",
        show_grid_lines=cfg.render_options.show_grid_lines,
        agent_view_size=cfg.agent_view_size,
        show_walls_pov=cfg.render_options.show_walls_pov,
        seed=cfg.seed,
        generate_optimal_path=cfg.training.generate_optimal_path,
        show_optimal_path=cfg.render_options.show_optimal_path,
    )
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

    train(agent, env, total_timesteps=cfg.training.total_timesteps)
    


if __name__ == "__main__":
    main()

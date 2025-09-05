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



class LinearQNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        q_values = nn.Dense(self.num_actions, use_bias=False)(x)
        return q_values

@jax.jit
def q_update(agent_train_state: TrainState, x: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, x_next: jnp.ndarray, done: jnp.bool_, discount: float):
    
    def loss_fn(params: jnp.ndarray):
        
        q_values = agent_train_state.apply_fn(params, x) # get q values for the state
        q_curr = q_values[a] # select the q value for the action taken
        q_next_max = jnp.where(done, 0.0, jnp.max(agent_train_state.apply_fn(params, x_next))) # if done, q_next_target is 0, otherwise it is the max q value of the next state
        td_error = (r + discount * q_next_max) - q_curr # the classic td error
        return jnp.square(td_error) # return the squared td error

    grad_fn = jax.value_and_grad(loss_fn) # create the gradient update function
    loss, grads = grad_fn(agent_train_state.params) # compute loss and gradients
    new_agent_train_state = agent_train_state.apply_gradients(grads=grads) # update the agent q values based on gradients
    return new_agent_train_state, loss


class QLearningAgent:
    def __init__(self, env_obs_dims : tuple, num_actions: int, discount: float, step_size: float, seed: int, epsilon: float, agent_pixel_view_edge_dim: int):
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
        self.epsilon = epsilon
        self.agent_pixel_view_edge_dim = agent_pixel_view_edge_dim

        self.linear_q_network = LinearQNetwork(num_actions)
        self.linear_q_network_params = self.linear_q_network.init(jax.random.PRNGKey(seed), jnp.zeros(( self.num_states,)))

        self.optimizer = optax.sgd(step_size)
        self.train_state = TrainState.create(
            apply_fn=self.linear_q_network.apply,
            params=self.linear_q_network_params,
            tx=self.optimizer
        )

    # epsilon greedy action selection
    def select_action(self, x: jnp.ndarray, key: jnp.ndarray): 
        x = self._agent_observation_transform(x)
        q_values = self.linear_q_network.apply(self.train_state.params, x) # get the q values for the state / observation
        greedy_action = jnp.argmax(q_values) # greedy action is one with the max q value
        random_action = jax.random.randint(key, (), 0, self.num_actions) # random action
        p = jax.random.uniform(key, shape=(), minval=0.0, maxval=1.0) # probability of choosing a random action
        action = jnp.where(p < self.epsilon, random_action, greedy_action) # choose a random action with probability epsilon, otherwise choose the greedy action

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


        
def train(agent: QLearningAgent, env: gym.Env, total_timesteps: int, save_path="rewards.pkl"):

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
        a = agent.select_action(obs["image"], key)
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

    agent = QLearningAgent(
        env_obs_dims=image_shape, 
        num_actions=env.action_space.n, 
        discount=cfg.training.discount, 
        step_size=cfg.training.step_size, 
        seed=cfg.seed, 
        epsilon=cfg.training.epsilon,
        agent_pixel_view_edge_dim= cfg.training.agent_pixel_view_edge_dim
    )

    train(agent, env, total_timesteps=cfg.training.total_timesteps)
    


if __name__ == "__main__":
    main()

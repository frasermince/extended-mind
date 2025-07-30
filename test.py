import pytest
from main_dqn import main, make_env, eval_env
from env import Actions
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# ============================
# Test Configuration Fixture
# ============================

@pytest.fixture
def test_config():
    cfg = OmegaConf.create()
    cfg.env_id = "MiniGrid-SaltAndPepper-v0-custom"
    cfg.seed = 42
    cfg.num_envs = 1
    cfg.eval = True
    cfg.eval_episodes = 1
    cfg.eval_freq = 1000
    cfg.exp_name = "test"
    cfg.experiment_description = "test"
    cfg.learning_rate = 0.0003
    cfg.feature_dim = 128
    cfg.agent_view_size = 3
    cfg.track = False
    cfg.capture_video = False
    cfg.show_grid_lines = False
    cfg.show_walls_pov = False
    cfg.train = False
    cfg.total_timesteps = 1000
    cfg.dense_features = [16, 16]


# ============================
# Test 1: Seeding Consistency
# ============================

def test_seeding(test_config):
    """
    Verifies that the environment is deterministic when using the same seed,
    and produces different observations with a different seed.
    """
    test_config.hardcoded_actions = [Actions.forward]
    
    # Clone config with a different seed
    different_seed_test_config = test_config.copy()
    different_seed_test_config.seed = test_config.seed + 2
    different_seed_test_config.hardcoded_actions = [Actions.forward]

    # Initialize environments
    env_1 = main(test_config)
    env_2 = main(test_config)
    env_different_seed = main(different_seed_test_config)

    # Reset all envs with appropriate seeds
    obs_1, _ = env_1.reset(seed=test_config.seed)
    obs_2, _ = env_2.reset(seed=test_config.seed)
    obs_different_seed, _ = env_different_seed.reset(seed=different_seed_test_config.seed)

    # Same seed -> same observations
    assert (obs_1["image"] == obs_2["image"]).all()

    # Re-reset with same seed -> still deterministic
    obs_2_reset, _ = env_2.reset(seed=test_config.seed)
    assert (obs_2["image"] == obs_2_reset["image"]).all()

    # Different seed -> different observations
    assert (obs_1["image"] != obs_different_seed["image"]).any()

    # Step forward and validate continued determinism
    env_1.step(Actions.forward)
    env_2.step(Actions.forward)
    env_different_seed.step(Actions.forward)
    assert (obs_1["image"] == obs_2["image"]).all()
    assert (obs_1["image"] != obs_different_seed["image"]).any()


# ================================================
# Test 2: Agent View Matches Ground Truth Tiles
# ================================================

def test_unique_tiles(test_config):
    """
    Verifies that the agent's observed local image matches the corresponding
    unique tile patches from the full-grid representation.
    """
    test_config.hardcoded_actions = [Actions.backward]
    
    # Create and reset environment
    env = make_env(
        test_config.training.env_id,
        test_config.seed,
        1,
        test_config.capture_video,
        test_config.exp_name,
        test_config.render_options.show_grid_lines,
        test_config.agent_view_size,
        test_config.render_options.show_walls_pov,
        test_config.generate_optimal_path,
        test_config.render_options.show_optimal_path,
    )()
    obs, _ = env.reset(seed=test_config.seed)

    # Extract 3x3 padded unique tile block around the agent
    top_x = env.unwrapped.pad_width + env.unwrapped.agent_pos[0] - 1
    bottom_x = top_x + 3
    top_y = env.unwrapped.pad_width + env.unwrapped.agent_pos[1] - 1
    bottom_y = top_y + 3
    local_unique = env.unwrapped.padded_unique_tiles[
        top_x:bottom_x, top_y:bottom_y, :, :, 0
    ]

    # Match each observed tile in agent image with corresponding unique tile
    for i in range(obs["image"].shape[0] // 8):
        for j in range(obs["image"].shape[1] // 8):
            i_start = i * 8
            i_end = i_start + 8
            j_start = j * 8
            j_end = j_start + 8
            image = obs["image"][i_start:i_end, j_start:j_end, 0]

            # Note: numpy array coordinates are [y, x], hence [j, i]
            assert (image == local_unique[j, i, :, :]).all()


# ====================================
# Test 3: Agent Hits Expected Walls
# ====================================

def test_wall_collisions(test_config):
    """
    Confirms that the agent's hardcoded movement patterns cause it to hit
    the expected corners (walls) of the grid.
    
    This validates that movement, wall collisions, and environment boundaries
    are functioning correctly.
    """
    action_sets = [
        ([Actions.left, Actions.forward] * 50, (1, 1)),       # Top-left corner
        ([Actions.right, Actions.forward] * 50, (13, 1)),     # Top-right corner
        ([Actions.left, Actions.backward] * 50, (1, 13)),     # Bottom-left corner
        ([Actions.right, Actions.backward] * 50, (13, 13)),   # Bottom-right corner
    ]

    for action_set, expected_pos in action_sets:
        test_config.hardcoded_actions = action_set
        envs = main(test_config)

        # Agent should end up in expected corner due to hitting wall
        assert envs.unwrapped.agent_pos == expected_pos

import pytest
from main_dqn import main, make_env, eval_env
from env import Actions
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


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

    return cfg


def test_seeding(test_config):
    test_config.hardcoded_actions = [Actions.forward]
    different_seed_test_config = test_config.copy()
    different_seed_test_config.seed = test_config.seed + 2
    different_seed_test_config.hardcoded_actions = [Actions.forward]

    env_1 = main(test_config)
    env_2 = main(test_config)
    env_different_seed = main(different_seed_test_config)

    obs_1, _ = env_1.reset(seed=test_config.seed)
    obs_2, _ = env_2.reset(seed=test_config.seed)
    obs_different_seed, _ = env_different_seed.reset(
        seed=different_seed_test_config.seed
    )
    assert (obs_1["image"] == obs_2["image"]).all()
    obs_2_reset, _ = env_2.reset(seed=test_config.seed)
    assert (obs_2["image"] == obs_2_reset["image"]).all()
    assert (obs_1["image"] != obs_different_seed["image"]).any()
    env_1.step(Actions.forward)
    env_2.step(Actions.forward)
    env_different_seed.step(Actions.forward)
    assert (obs_1["image"] == obs_2["image"]).all()
    assert (obs_1["image"] != obs_different_seed["image"]).any()


def test_unique_tiles(test_config):
    # self.unique_tiles
    test_config.hardcoded_actions = [Actions.backward]
    env = make_env(
        test_config.env_id,
        test_config.seed,
        1,
        test_config.capture_video,
        test_config.exp_name,
        test_config.show_grid_lines,
        test_config.agent_view_size,
        test_config.show_walls_pov,
    )()
    obs, _ = env.reset(seed=test_config.seed)
    top_x = env.unwrapped.pad_width + env.unwrapped.agent_pos[0] - 1
    bottom_x = top_x + 3
    top_y = env.unwrapped.pad_width + env.unwrapped.agent_pos[1] - 1
    bottom_y = top_y + 3
    local_unique = env.unwrapped.padded_unique_tiles[
        top_x:bottom_x, top_y:bottom_y, :, :, 0
    ]
    for i in range(obs["image"].shape[0] // 8):
        for j in range(obs["image"].shape[1] // 8):
            i_start = i * 8
            i_end = i_start + 8
            j_start = j * 8
            j_end = j_start + 8
            image = obs["image"][i_start:i_end, j_start:j_end, 0]

            # Numpy images go y, x so the coordinates are flipped
            assert (image == local_unique[j, i, :, :]).all()


def test_wall_collisions(test_config):
    action_sets = [
        ([Actions.left, Actions.forward] * 50, (1, 1)),
        ([Actions.right, Actions.forward] * 50, (13, 1)),
        ([Actions.left, Actions.backward] * 50, (1, 13)),
        ([Actions.right, Actions.backward] * 50, (13, 13)),
    ]

    for action_set, expected_pos in action_sets:
        test_config.hardcoded_actions = action_set
        envs = main(test_config)
        assert envs.unwrapped.agent_pos == expected_pos

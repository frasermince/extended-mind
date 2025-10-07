import pytest
from main_dqn import main, make_env, eval_env
from env import Actions
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# ============================
# Test Configuration Fixture
# ============================


@pytest.fixture
def test_config():
    cfg = OmegaConf.create()

    cfg.eval = True
    cfg.train = False
    cfg.seed = 42
    cfg.exp_name = "test"
    cfg.agent_view_size = 3
    cfg.track = False
    cfg.capture_video = False
    cfg.generate_optimal_path = False
    cfg.dry_run = False
    cfg.experiment_description = "test"

    cfg.training = OmegaConf.create()
    cfg.training.env_id = "MiniGrid-SaltAndPepper-v0-custom"
    cfg.training.num_envs = 1
    cfg.training.eval_episodes = 1
    cfg.training.eval_freq = 1000
    cfg.training.learning_rate = 0.0003
    cfg.training.feature_dim = 128
    cfg.training.train = False
    cfg.training.total_timesteps = 1000
    cfg.training.dense_features = [16, 16]

    cfg.render_options = OmegaConf.create()
    cfg.render_options.show_grid_lines = False
    cfg.render_options.show_walls_pov = False
    cfg.render_options.show_optimal_path = True

    cfg.show_landmarks = False
    cfg.path_mode = "NONE"
    cfg.run_folder = ""
    return cfg


# ============================
# Test 1: Seeding Consistency
# ============================


def test_seeding(test_config):
    """
    Verifies that the environment is deterministic when using the same seed,
    and produces different observations with a different seed.
    """
    for path in ["NONE", "SHORTEST_PATH"]:
        test_config.path_mode = path
        test_config.run_folder = ""
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
        obs_different_seed, _ = env_different_seed.reset(
            seed=different_seed_test_config.seed
        )

        # Same seed -> same observations
        assert (obs_1["image"] == obs_2["image"]).all()

        # Re-reset with same seed -> still deterministic
        obs_2_reset, _ = env_2.reset(seed=test_config.seed)
        assert (obs_2["image"] == obs_2_reset["image"]).all()

        # Different seed -> different observations
        assert (obs_1["image"] != obs_different_seed["image"]).any()

        # Step forward and validate continued determinism
        env_1.step(np.array([Actions.forward.value]))
        env_2.step(np.array([Actions.forward.value]))
        env_different_seed.step(np.array([Actions.forward.value]))
        assert (obs_1["image"] == obs_2["image"]).all()
        assert (obs_1["image"] != obs_different_seed["image"]).any()


# ================================================
# Test 2: Agent View Matches Ground Truth Tiles
# ================================================


@pytest.mark.skip(reason="Run this individually. Takes too long.")
def test_visitation_path_matches_unique_tiles(test_config):
    """
    Verifies that the agent's observed local image matches the corresponding
    unique tile patches from the full-grid representation.
    """
    test_config.hardcoded_actions = [Actions.backward]
    test_config.capture_video = True
    test_config.path_mode = "VISITED_CELLS"

    # Create and reset environment
    env = make_env(
        test_config.training.env_id,
        test_config.seed,
        0,
        test_config.capture_video,
        test_config.exp_name,
        test_config.render_options.show_grid_lines,
        test_config.agent_view_size,
        test_config.render_options.show_walls_pov,
        test_config.render_options.show_optimal_path,
        test_config.path_mode,
        test_config.show_landmarks,
    )()
    obs, _ = env.reset(seed=test_config.seed)
    env.unwrapped.grid.test_mode = True

    terminations = np.array([False])
    truncations = np.array([False])
    for i in trange(10000):
        if terminations or truncations:
            obs, _ = env.reset(seed=test_config.seed)
            env.unwrapped.grid.test_mode = True
            terminations = np.array([False])
            truncations = np.array([False])
            continue
        obs, _, terminations, truncations, _ = env.step(
            np.array([np.random.choice([a.value for a in Actions])])
        )
        full_obs = env.unwrapped.get_full_render(
            highlight=False, tile_size=8, reveal_all=True
        )

        # Match each observed tile in agent image with corresponding unique tile
        top_x_cell = env.unwrapped.agent_pos[0] - 1
        top_y_cell = env.unwrapped.agent_pos[1] - 1
        for i in range(obs["image"].shape[0] // 8):
            for j in range(obs["image"].shape[1] // 8):
                i_start = i * 8
                i_end = i_start + 8
                j_start = j * 8
                j_end = j_start + 8
                cell_image_partial = obs["image"][i_start:i_end, j_start:j_end, 0]
                cell_image_full = full_obs[
                    (top_y_cell + i) * 8 : (top_y_cell + i + 1) * 8,
                    (top_x_cell + j) * 8 : (top_x_cell + j + 1) * 8,
                ]
                plt.imsave(
                    f"cell_image_full_grid.png",
                    full_obs,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                plt.close()
                plt.imsave(f"cell_image_full.png", cell_image_full)
                plt.close()
                plt.imsave(
                    f"cell_image_partial.png",
                    cell_image_partial,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                plt.close()
                # Reduce cell_image_full from (8,8,3) to (8,8,1) using value mapping
                # We'll use numpy's vectorized operations to "switch" certain RGB values to grayscale codes
                # Example: map [255,255,255] -> 255, [0,0,0] -> 0, [0,0,255] (blue path) -> 128, else -> 64
                cell_image_full_gray = np.zeros(
                    cell_image_full.shape[:2], dtype=np.uint8
                )
                # Map white
                mask_white = np.all(cell_image_full == [255, 255, 255], axis=-1)
                cell_image_full_gray[mask_white] = 255
                # Map black
                mask_black = np.all(cell_image_full == [0, 0, 0], axis=-1)
                cell_image_full_gray[mask_black] = 0

                mask_lines = np.all(cell_image_full == [100, 100, 100], axis=-1)
                cell_image_full_gray[mask_lines] = 255

                # Map blue (path)
                mask_blue = np.all(cell_image_full == [0, 0, 255], axis=-1)
                cell_image_full_gray[mask_blue] = 0
                # Map any other color to 64
                mask_other = ~(mask_white | mask_black | mask_blue)
                cell_image_full_gray[mask_other] = 255
                # Expand dims to (8,8,1)
                plt.imsave(
                    f"cell_image_full_gray.png",
                    cell_image_full_gray,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                plt.close()
                if not (cell_image_partial == cell_image_full_gray).all():
                    import pdb

                    pdb.set_trace()

                # Note: numpy array coordinates are [y, x], hence [j, i]
                assert (cell_image_partial == cell_image_full_gray).all()


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
        test_config.path_mode,
        test_config.show_landmarks,
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
        ([Actions.left, Actions.forward] * 50, (1, 1)),  # Top-left corner
        ([Actions.right, Actions.forward] * 50, (13, 1)),  # Top-right corner
        ([Actions.left, Actions.backward] * 50, (1, 13)),  # Bottom-left corner
        ([Actions.right, Actions.backward] * 50, (13, 13)),  # Bottom-right corner
    ]

    for action_set, expected_pos in action_sets:
        test_config.hardcoded_actions = action_set
        envs = main(test_config)

        # Agent should end up in expected corner due to hitting wall
        assert envs.unwrapped.agent_pos == expected_pos

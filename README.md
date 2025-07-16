# Extended Mind

## Installation
uv venv
source .venv/bin/activate
uv sync

## Run
python main_dqn.py --experiment-description="big-grid" --learning-rate="1e-4" --agent-view-size=3 --no-capture_video --feature_dim=256
git config --global user.name "laurie-hxf"
git config --global user.email "1847238294@qq.com"
apt update && apt install procps -y
# export UV_CACHE_DIR=/workspace/.cache/uv
# # 假设 PVC 挂载在 /workspace
# export UV_INSTALL_DIR=/workspace/.local/bin
# export VENV_DIR="${VENV_DIR:-/.venv}"
# curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$UV_INSTALL_DIR sh

# # 然后加到 PATH
# export PATH="/workspace/.local/bin:$PATH"

# uv venv --python 3.11 "$VENV_DIR"

# source "$VENV_DIR/bin/activate"

# uv pip install -r requirements.txt
set -a
source /workspace/.secrets/wandb.env
set +a

wandb login --relogin "$WANDB_API_KEY"
if [ -z "$1" ]; then
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 24

# Verify the Node.js version:
node -v # Should print "v24.15.0".

# Verify npm version:
npm -v # Should print "11.12.1".
npm i -g ccman
npm i -g @openai/codex
ccman gmn
fi
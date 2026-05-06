pip install --cache-dir /workspace/.pip-cache -r requirements.txt
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

git config --global user.name "laurie-hxf"
git config --global user.email "1847238294@qq.com"

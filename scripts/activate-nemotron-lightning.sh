# Source this file inside your compute allocation: source scripts/activate-nemotron-lightning.sh
export AGENT_ROOT="/data/${USER:?}/pinnacles-agents"
export PATH="$AGENT_ROOT/envs/nemotron-vllm/bin:$AGENT_ROOT/tools/opencode-v1.18.30:$PATH"
export HF_HOME="$AGENT_ROOT/huggingface"
export XDG_CONFIG_HOME="$AGENT_ROOT/opencode-nemotron-lightning/config"
export XDG_DATA_HOME="$AGENT_ROOT/opencode-nemotron-lightning/data"
export XDG_STATE_HOME="$AGENT_ROOT/opencode-nemotron-lightning/state"
export XDG_CACHE_HOME="/scratch/$USER/pinnacles-agents/opencode-nemotron-lightning-cache"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

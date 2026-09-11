# Source this file inside your compute allocation: source scripts/activate-gemma.sh
export AGENT_ROOT="/data/${USER:?}/pinnacles-agents"
export PATH="$AGENT_ROOT/envs/gemma-vllm/bin:$AGENT_ROOT/tools/opencode-v1.18.30:$PATH"
export HF_HOME="$AGENT_ROOT/huggingface"
export XDG_CONFIG_HOME="$AGENT_ROOT/opencode/config"
export XDG_DATA_HOME="$AGENT_ROOT/opencode/data"
export XDG_STATE_HOME="$AGENT_ROOT/opencode/state"
export XDG_CACHE_HOME="/scratch/$USER/pinnacles-agents/opencode-cache"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

# Source inside the compute allocation. This does not change any H200 environment.
export AGENT_ROOT="/data/${USER:?}/pinnacles-agents"
export PATH="$AGENT_ROOT/tools/llama.cpp-f8def7fe168bab245fbf15d3f18b26dbb1ef73c8/bin:$AGENT_ROOT/tools/opencode-v1.18.30:$PATH"
export XDG_CONFIG_HOME="$AGENT_ROOT/opencode-gguf/config"
export XDG_DATA_HOME="$AGENT_ROOT/opencode-gguf/data"
export XDG_STATE_HOME="$AGENT_ROOT/opencode-gguf/state"
export XDG_CACHE_HOME="/scratch/$USER/pinnacles-agents/opencode-gguf-cache"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

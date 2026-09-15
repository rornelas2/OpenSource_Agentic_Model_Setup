# Open-weight coding agents on UC Merced Pinnacles

A tutorial for students participating in Lawrence Livermore National Laboratory's Data Science
Challenge. Learn how to run an open-weight model on Pinnacles GPUs, serve it
through an API, and use **OpenCode or Codex CLI** to read code, edit files, and run tests.

You need a regular sponsored Pinnacles account and basic familiarity with a
terminal. This tutorial uses the public `gpu` and `cenvalarc.gpu` partitions.

## What you will build

```text
You → OpenCode or Codex CLI → model server (vLLM or llama.cpp) → GPU model
              │                     │
              └── tools and results ┘
              │
              └── read files, edit code, run tests
```

The **model** proposes actions. The **server** loads its weights onto GPUs and
provides an HTTP endpoint. **OpenCode or Codex CLI** is the coding harness: it manages the
conversation, supplies tools, executes approved actions, and returns their
results to the model. **Slurm** allocates the compute node where the server runs;
the endpoint lasts only as long as that job.

OpenCode supports local servers through an OpenAI-compatible API. Here,
“OpenAI-compatible” describes the request format; inference runs on Pinnacles.
Files and shell commands are handled on the machine running the harness. A laptop
client therefore edits laptop files unless its tools are explicitly configured
otherwise. See the [OpenCode provider documentation](https://opencode.ai/docs/providers/).
Codex uses the server’s Responses API; it still runs your selected open-weight
model on Pinnacles. See [choosing Codex instead of OpenCode](#16-choose-codex-instead-of-opencode)
for installation, launch commands, and the validation status of each profile.

**Start here:** follow the Gemma steps below, then the
[Meta lesson](#9-run-meta-muse-glimmer-through-opencode), the
[NVIDIA lesson](#10-run-nvidia-nemotron-3-super-through-opencode), the
[quantized Muse on A100 lesson](#11-run-quantized-meta-muse-glimmer-on-an-a100-gpu), the
[Gemma QAT on A100/L40S lesson](#12-run-google-gemma-4-31b-it-qat-on-an-a100-or-l40s-gpu), the
[Nemotron 3.5 Lightning on A100/L40S lesson](#13-run-nvidia-nemotron-35-lightning-30b-nvfp4-on-an-a100-or-l40s-gpu), and the
[multi-GPU 128K context scaling lesson](#14-scale-context-to-128k-on-multiple-gpus-dual-a100-or-l40s). For another checkpoint, use the
[unlisted-model guide](#15-connect-a-model-that-is-not-listed). The
[reference section](#reference-gpus-memory-and-model-selection) explains GPU
memory, storage, quantization, and benchmark results.

**My recommendation:** use quantized Muse Glimmer on **two A100 GPUs** for
coding on Pinnacles. In my experience, this allocation is easier to obtain
than an H200 and offers a good balance of coding capability and GPU
accessibility. Follow the
[two-GPU setup in Section 14](#14-scale-context-to-128k-on-multiple-gpus-dual-a100-or-l40s).

## 1. Log in and open the tutorial

On your **laptop**, connect to the campus VPN when off campus, then replace
`UCM_USERNAME` with your UC Merced username:

```bash
ssh UCM_USERNAME@login.rc.ucmerced.edu
```

On the **Pinnacles login node**:

```bash
git clone https://github.com/rornelas2/OpenSource_Agentic_Model_Setup.git
cd OpenSource_Agentic_Model_Setup
```

If you already have a clone, enter that directory instead. Use the login node
for editing files and submitting jobs. Run inference, builds requiring
substantial compute, and agent-triggered tests in a compute allocation.
[Campus login instructions](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/campus-clusters/#centralized-login)
and [cluster policies](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/policies/)
explain access and appropriate use.

## 2. Install the Gemma and OpenCode tooling

This exercise uses **Gemma 4 31B IT in BF16 on one H200**, with 8 CPU cores,
128 GB system RAM, and a 98,304-token context limit. You do not need a paid
model API account. Budget **90 GB free in data** and **30 GB free in scratch**
for the checkpoint, environment, and installation/build caches.

Validated on Pinnacles: model loading, chat, streaming,
a tool-call round trip, and an OpenCode file repair with all four exercise tests
passing. The current configured ceiling is 98,304 tokens (96K); the exercise
uses a shorter conversation and is not a full-context stress test.

On the **login node**, from the tutorial directory:

```bash
module load anaconda3/2023.09-0
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 bash scripts/setup-gemma.sh
```

This requests a CPU allocation for installation; it does not reserve a GPU.
Slurm may queue the request. The script creates a Python environment under
`/data/$USER/pinnacles-agents`, installs the pinned dependencies, and downloads
OpenCode and the Gemma tool-call template with checksum verification. Wait for
`Setup complete`. Run it once; you can rerun it to restore the pinned setup.

The supplied versions are Python 3.11, vLLM 0.29.0, and OpenCode 1.18.30.
Students choosing Codex additionally install the pinned Codex CLI 0.154.0
using [Section 16](#16-choose-codex-instead-of-opencode); the model setup stays the same.
The complete Python dependency list is in
[`env/gemma-requirements.lock`](env/gemma-requirements.lock).
The installation uses prebuilt packages and requires no `sudo` or Docker.

### Environment setup and future project requirements

The setup above installs the **model-serving environment**. Either harness also needs
an environment in which to run the code it edits. The introductory repair
exercise uses Python's standard-library `unittest`, so it needs no additional
project packages. A later data-science project may need its own dependencies. Keep that project's
environment separate from the serving runtime so its package changes do not
alter the tested model setup. The Muse GGUF lesson uses a compiled llama.cpp
server instead of the Python/vLLM serving environment.

Your project can live outside this tutorial directory and use its own venv,
Conda environment, or another suitable environment manager. See
[working in your own project](#working-in-your-own-project) for how the running
model server, your chosen harness, and the project's environment fit together.

**PyTorch is already included:** all three serving lock files pin
`torch==2.13.0`, alongside `vllm==0.29.0` and `numpy==2.3.5`.


Read the [detailed environment guide and questions](env/README.md) for:

- Prerequisites, installation locations, and what each setup script installs.
- Why Python 3.11, virtual environments, `uv`, and pinned dependencies are used.
- How activation and OpenCode determine which Python executes project code.
- Where future PyTorch and test dependencies belong, and how to
  select, record, and validate them.
- CUDA versus the GPU driver, storage, reproducibility, and troubleshooting.

The guide distinguishes the existing tutorial setup from future project
extensions that still need Pinnacles validation.

## 3. Download Gemma 4 31B

On the **login node**, still in the tutorial directory:

```bash
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=01:00:00 bash scripts/download-gemma.sh
```

The script downloads about **62.5 GB** of official model weights plus tokenizer
and configuration files to `/data/$USER/pinnacles-agents/models/gemma-4-31B-it`.
The checkpoint is pinned to revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Wait for the destination path and a
successful command exit before requesting a GPU. Download time depends on
network and storage load; subsequent runs reuse completed files.

This revision was accessible without authentication during validation. If
Hugging Face requires authentication for your account, run this on the **login
node** after setup, then rerun the download:

```bash
HF_HOME="/data/$USER/pinnacles-agents/huggingface" \
  "/data/$USER/pinnacles-agents/envs/gemma-vllm/bin/hf" auth login
```

Read the [model card](https://huggingface.co/google/gemma-4-31B-it)
for the checkpoint's license and usage information.

Use `Ctrl-C` to stop the foreground installation or download job. Rerun the same
step to continue; completed model files stay in your data directory.

## 4. Request an H200 compute shell

On the **login node**:

```bash
srun --partition=cenvalarc.gpu --nodes=1 --ntasks=1 \
  --gres=gpu:nvidia_h200_nvl:1 --cpus-per-task=8 --mem=128G \
  --time=01:00:00 --pty bash
```

Wait for the compute shell. It stays open until you exit or the hour expires.
All following commands, including OpenCode and its tests, run **inside this
compute shell**. Keep this terminal connected while working.

Confirm that you are on the compute node before starting the server:

```bash
hostname
echo "$SLURM_JOB_ID"
```

The hostname should look like `gnode###.cluster`, and `SLURM_JOB_ID` should
contain a job number. A hostname such as `rclogin02` means you are on the login
node. Do not start the server or run its `curl` checks there.

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-gemma.sh
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
```

The GPU should be an H200 NVL. The activation script adds the installed tools
to this shell's PATH and keeps OpenCode's configuration and session data in the
tutorial's data directory. This does not modify your shell startup files.

## 5. Start the model endpoint

Inside the **same compute shell**, from the tutorial directory:

```bash
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/gemma-${SLURM_JOB_ID}.log"
bash scripts/serve-gemma.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

The first startup loads the checkpoint and compiles GPU kernels; allow several
minutes (about five minutes in the initial validation). Wait for
`Application startup complete`, then press **Ctrl-C to leave `tail`**. The
background model server continues running. If startup reports an error, inspect
the last part of this log before proceeding.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

The health check succeeds with an empty response body. The models response
should include **`gemma4-31b`**, the name that OpenCode will send in API requests.
The server listens on the compute node's loopback interface. The address is
reachable by OpenCode in this allocation, not by a browser on your laptop.
Loopback is local to the node, not to your Slurm job or Unix account: other
processes on that node can reach this unauthenticated tutorial endpoint.
These launch scripts are for the individual exercise; a shared service needs
authentication and tested concurrency limits.


## 6. Send a request to your model

Inside the **compute shell**:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-31b",
    "messages": [{"role": "user", "content": "Explain a Python list comprehension in two sentences."}],
    "max_tokens": 128
  }'
```

Look for the generated text in `choices[0].message.content`. This verifies the
endpoint before introducing the coding harness. No external model service is
involved in this request.

## 7. Use Gemma as a coding agent through OpenCode

**Harness choice:** the steps below use OpenCode. For Codex, see
[Section 16](#16-choose-codex-instead-of-opencode), including the validation
status of the `gemma` profile, before substituting the agent launch step.

Start with the supplied exercise: a small function with a bug and four tests.
Copy it into your own work directory so the original exercise stays reusable.
Use a fresh directory name if `first-agent` already exists.

Inside the **compute shell**:

```bash
mkdir -p "$AGENT_ROOT/work"
cp -R "$TUTORIAL_DIR/examples/repair-mean" "$AGENT_ROOT/work/first-agent"
cd "$AGENT_ROOT/work/first-agent"
cp "$TUTORIAL_DIR/config/opencode-gemma.json" opencode.json
python -m unittest -v
```

The tests **should fail initially**: the function does not handle missing
readings correctly. Now start OpenCode in this exercise directory:

```bash
opencode
```

The selected model should be `pinnacles/gemma4-31b`. Enter this prompt:

> Read summary.py and test_summary.py. Fix mean_readings in summary.py so that
> it ignores None, includes zero, and raises ValueError when no readings remain.
> Do not edit test_summary.py. Use your file-editing tool to make the change,
> then run python -m unittest -v. Report the test result.

Review and approve the proposed file edit and test command. You should see
OpenCode read the files, modify `summary.py`, and run the tests. The model
proposes the tool calls; OpenCode performs the actions and sends the results
back to Gemma. File reads are allowed by the supplied configuration; edits and
shell commands prompt for approval.

Exit OpenCode with `/exit`, then verify the result yourself:

```bash
python -m unittest -v
diff -u "$TUTORIAL_DIR/examples/repair-mean/summary.py" summary.py
cmp "$TUTORIAL_DIR/examples/repair-mean/test_summary.py" test_summary.py
```

Success means all four tests pass, the function changed, and the test file is
unchanged. `diff` returns exit status 1 when it finds the expected changes;
`cmp` prints nothing when the test files match.

### Working in your own project

**Your project does not need to be inside `OpenSource_Agentic_Model_Setup`.**
The tutorial directory supplies the setup scripts and model configurations.
Both harnesses work in the project directory from which you launch them. The exercise
above already uses a separate work directory.

After starting the model server using your chosen lesson, leave it running and
change into your own project directory in the activated compute shell. Copy the
matching configuration from the tutorial's `config/` directory into your project
as `opencode.json`, merging it with any existing configuration instead of
overwriting it. The model ID, endpoint, and context limits must match the server.
Then run `opencode` from your project directory.

On later visits, once that configuration exists, launching OpenCode means
changing into the project and typing `opencode`, provided the tools are on
`PATH` and the matching server is running. The activation scripts set up the
current shell; a new shell needs its own activation. The supplied endpoint is
`127.0.0.1:8000`, so this workflow runs OpenCode on the same compute node as the
server, inside the appropriate allocation.

For Codex, run the [shared launcher](#16-choose-codex-instead-of-opencode) from
your project directory with the matching profile; no `opencode.json` copy is
needed. Its session state is separate from your normal Codex configuration.

The [OpenCode documentation](https://opencode.ai/docs/providers/) explains custom
providers; [permissions](https://opencode.ai/docs/permissions/) control tool use.

### Changing thinking levels in either harness

All ten supplied profiles expose their model's native controls:

| Model family (all hardware/context profiles) | Selectable levels | Initial default |
|---|---|---|
| Muse Glimmer BF16 and GGUF | `low`, `medium`, `high`, `xhigh` | `high` |
| Gemma 4 BF16 and QAT | `none` (off), `high` (on) | `none` |
| Nemotron 3 Super | `none` (off), `low`, `high` (full thinking) | `high` |
| Nemotron 3.5 Lightning | `none` (off), `high` (on) | `high` |

For the binary switches, `high` means **thinking enabled**, not a separately
trained high-effort tier. Muse has four strengths and no advertised off mode.
Thinking tokens and the final answer share the output allowance; selecting a
higher effort does not increase the configured output or context limit.

**OpenCode:** merge the updated matching `config/opencode-*.json` into your
project's `opencode.json`, including the model's `options`, `reasoning`, and
`variants` fields. Restart OpenCode, select the local model, and press **Ctrl+T**
to cycle variants. The displayed variant applies to subsequent requests.
Unsupported automatically generated variants are explicitly disabled in the
supplied configuration. Keep those `disabled` entries when merging.
An unselected/default variant uses the model's `options` above; it does not
mean thinking is disabled. For a noninteractive run, select it explicitly:

```bash
opencode run --variant low "Explain this project's entry point."
```

That example is for Muse; use `--variant none` or `--variant high` for Gemma
and Nemotron (Super also accepts `--variant low`). To change the initial default
in `opencode.json`, edit
`provider.pinnacles.models.MODEL_ID.options.chat_template_kwargs`:
Muse uses `{"reasoning_strength":"medium"}`, for example; Gemma/Nemotron
use `{"enable_thinking":true}` or `{"enable_thinking":false}`. Super additionally
uses `"low_effort":true` for low and `false` for full thinking/off. Variants override
this default. Existing project copies are not updated by `git pull`.
See [OpenCode variants](https://opencode.ai/docs/models/#variants).

**Codex:** start through the tutorial launcher, run **`/model`**, choose the
listed local model, and select its reasoning effort. The descriptions identify
on/off models. To set the initial effort from the command line, put
`--thinking` **before the profile**:

```bash
python3 "$TUTORIAL_DIR/scripts/launch-codex.py" --thinking low muse-gguf-128k
python3 "$TUTORIAL_DIR/scripts/launch-codex.py" --thinking high gemma
```

Choose the profile matching your running server. `--list` shows each profile's
supported levels. Unsupported `--thinking` values fail before connecting.
Restart an older launcher session once to load the updated catalog and adapter;
subsequent `/model` effort changes need no server restart. Launches start with
the default in the table unless `--thinking` is supplied. Codex's usual
`model_reasoning_effort` setting is described in the
[official configuration reference](https://developers.openai.com/codex/config-reference/).

The Codex launcher runs a small loopback adapter for its own session. It converts
`reasoning.effort` into the model's `chat_template_kwargs`, then streams the
Responses API reply unchanged. This is necessary because the pinned servers do
not consistently map standard effort fields to model-native settings. It uses
an automatically selected local port, closes when Codex exits, and keeps the
original GPU endpoint available to OpenCode. No extra package is needed.

Model controls come from the [Muse model card](https://huggingface.co/meta-models/Muse-Glimmer-30B#best-practices),
the supplied Gemma templates, and the pinned NVIDIA
[Super template](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/blob/ff433f5493e25d631c9f12b5d55c674229923d02/chat_template.jinja) and
[Lightning template](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/blob/bee7596271d1495f6992ae224aefde4410e816b8/chat_template.jinja). These settings control
inference behavior, not whether the harness displays reasoning text.

**Validation (2026-09-14):** on Pinnacles, the pinned Codex 0.154.0 and
OpenCode 1.18.30 clients sent the expected native settings for every configured
level in request-capture checks. On one A100 40GB, Muse GGUF at 32K completed
real Responses requests and OpenCode runs at all four strengths; the public
Codex launcher also completed a streamed file-read tool round trip at `low`.
Gemma and NVIDIA thinking changes have configuration/request validation;
their GPU inference and Codex tool round trips were not rerun for this feature.
These short checks do not establish speed or quality improvements.

### Asking for fewer permissions in OpenCode

The supplied profiles ask for edits and shell commands. For my recommended
workflow—allow routine actions and ask for `sudo`, `rm -r`, and `rm -rf`—replace
only the top-level `permission` object in your project's **`opencode.json`**:

```json
"permission": {
  "*": "allow",
  "bash": {
    "*": "allow",
    "*sudo*": "ask",
    "*rm *-r*": "ask",
    "*rm *-R*": "ask",
    "*rm *-fr*": "ask",
    "*rm *--recursive*": "ask"
  }
}
```

This is a JSON member to merge into the existing file, keeping the provider
and model configuration. Restart OpenCode after editing. `allow` runs without
a prompt, `ask` requests approval, and `deny` blocks the matching action. Change
the listed `ask` values to `deny` if those commands should never be approved.
To ask before every shell command, set `"bash": "ask"`; to ask before file
changes, add `"edit": "ask"` next to `bash`. To return to the tutorial defaults,
restore its original `permission` object.

OpenCode uses wildcard matches with the **last matching rule winning**, so put
`"*"` first. The broad patterns above also cover common absolute paths, `-rf`,
`-fr`, `-R`, and `--recursive`, and can ask for harmless commands containing
similar text. They are command approval rules, not a security sandbox: aliases,
scripts, Python, or other allowed tools can perform equivalent operations.
Global, project, and agent-specific configuration can affect the effective
policy; check those overrides if prompts differ from what you expect. Consult
[OpenCode permissions](https://opencode.ai/docs/permissions/) for the rule syntax.

### Using your project's own environment

**Yes, the agent can use a project's own environment and install project
dependencies there while the model server stays running.** That environment
can be a Python venv, a Conda environment, or another suitable choice for the
project. It does not have to be stored inside the project directory: a venv
often lives in `.venv`, while a Conda environment may be stored elsewhere.

The relationship is:

| Component | What its environment is for |
|---|---|
| Running model server | Serving the model with the selected runtime and pinned dependencies |
| OpenCode or Codex | Connecting to the server and executing file and shell tools in your project |
| Project code and tests | Using the project's own interpreter and packages, such as NumPy and Matplotlib |

The intended sequence is to activate the serving tools, launch the model server
using the tutorial, change into the project, select its environment, and launch
your chosen harness. The project environment can be created beforehand or set up by the
agent through its shell tools, subject to your configured permissions. Tell the
agent which environment to use and where dependencies should be installed.

These are independent environments, not nested layers of Python packages.
Selecting a project environment changes which interpreter subsequent project
commands use; it does not change the already-running server's interpreter or
unload its model. Install project dependencies into the project environment,
and record them in the project's dependency files. Packages in the serving
environment are not automatically available in the project environment.

For reliable agent execution, record the environment and run/test commands in
your project's `AGENTS.md`. For example, specify `.venv/bin/python` for a venv,
or identify the intended Conda environment and how commands should select it.
Activation in one temporary agent shell command may not persist into the next,
so each command must use the intended environment. When activating before
launching either harness, select the project environment **after** sourcing the model
activation script, which can put the serving Python first on `PATH`.

This explains how to extend the setup; a particular project's venv or Conda
installation still needs validation on Pinnacles. Separate environments isolate
packages, but still share the allocation's CPU, RAM, and GPU resources. See the
[environment guide](env/README.md#12-which-python-will-opencode-use-for-my-project)
for interpreter checks and project dependency management.

## 8. Stop the server and release your GPU

After exiting your chosen harness, inside the **compute shell**:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

The server stops and `exit` ends the allocation, returning you to the login
node. A nonzero `wait` status can occur after terminating a process. Downloaded
weights and environments remain available for your next session: repeat steps
4–7 instead of reinstalling everything.

If your terminal disconnects, log in again and run `squeue --me`. Cancel the
specific remaining job with `scancel JOB_ID`, replacing `JOB_ID` with its
numeric ID. Wall-time expiration also stops the server and makes the endpoint
unavailable.

### Troubleshooting the Gemma exercise

| Symptom | What to check |
|---|---|
| Job stays pending | The H200 queue may be busy. Use `squeue --me`; do not start the model on the login node. |
| `Invalid account` or partition access denied | Confirm that your sponsored account can use `cenvalarc.gpu`. |
| Setup cannot find Python 3.11 | Run `module load anaconda3/2023.09-0` before submitting setup. |
| Download or install runs out of space | Check your personal data/scratch usage; the free space of the entire filesystem is not your quota. |
| `Connection refused` | Wait for startup and inspect the server log; run the client on the same compute node. |
| Port 8000 already in use | Check your server log. If the listener is yours, stop that earlier server; otherwise choose another port in the serving script, OpenCode's `baseURL` and curl checks. A successful health check alone might belong to another user's server on the same node. |
| Out of GPU memory | Confirm an H200 was allocated and that another server is not already using your allocated GPU. |
| OpenCode returns text but makes no edits | Check approval prompts and that both Gemma parsers and the supplied chat template are enabled. |
| Request exceeds context limit | Begin a new session or reduce attached files/tool output; input and output share the context budget. |

## 9. Run Meta Muse Glimmer through OpenCode

**Harness choice:** the steps below use OpenCode. For Codex, see
[Section 16](#16-choose-codex-instead-of-opencode), including the validation
status of the `muse` profile, before substituting the agent launch step.

After completing Gemma, repeat the workflow with **Muse Glimmer 30B in BF16 on
one H200**. Use 8 CPU cores, 128 GB system RAM, and a 131,072-token context limit.
This lesson uses a separate Python environment and model directory. Stop your
Gemma server before starting Muse on the same node: both use port 8000.
If following the lessons in order, finish step 8 first so that you are back on
the login node and have released the Gemma allocation.

Budget an additional **90 GB free in data** and **30 GB free in scratch**.
Keeping both checkpoints and environments uses storage for both; check your
remaining quota before downloading. Muse uses the same pinned Python 3.11,
vLLM 0.29.0, and OpenCode 1.18.30 versions as Gemma. Its dependency list is in
[`env/muse-requirements.lock`](env/muse-requirements.lock).

Validated on Pinnacles: chat, streaming, an automatic
tool-call round trip, and an OpenCode repair with all four tests passing and the
test file unchanged. The first startup took about **5½ minutes**, including
checkpoint loading and kernel compilation. Model loading reported **52.07 GiB**;
device memory after the agent run was **120,661 MiB** (about 117.8 GiB), including
reserved cache and runtime state. This validates a short coding session at a
configured 32K ceiling, not a full-context stress test.

### Install and download on a CPU allocation

On the **login node**, from the tutorial directory:

```bash
module load anaconda3/2023.09-0
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 bash scripts/setup-muse.sh
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=64G --time=01:00:00 bash scripts/download-muse.sh
```

The download requests 64 GB of system RAM; a 16 GB download allocation ran out
of memory during validation. Wait for each command to finish successfully.
The scripts install the runtime at `/data/$USER/pinnacles-agents/envs/muse-vllm` and download the official
checkpoint (about **59.6 GB** of weights) to
`/data/$USER/pinnacles-agents/models/Muse-Glimmer-30B`.
The model revision is pinned to
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. Rerunning the download reuses completed
files. Consult [Meta's model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
for its license and usage information.

### Start Muse on an H200

On the **login node**, request a fresh compute shell:

```bash
srun --partition=cenvalarc.gpu --nodes=1 --ntasks=1 \
  --gres=gpu:nvidia_h200_nvl:1 --cpus-per-task=8 --mem=128G \
  --time=01:00:00 --pty bash
```

Inside the **compute shell**:

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-muse.sh
hostname
echo "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/muse-${SLURM_JOB_ID}.log"
bash scripts/serve-muse.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

Wait for `Application startup complete`, then press **Ctrl-C to leave `tail`**.
The background server continues running. Check it from this **compute shell**:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "muse-glimmer-30b",
    "messages": [{"role": "user", "content": "Explain a Python list comprehension in two sentences."}],
    "max_tokens": 2048
  }'
```

The model list should contain **`muse-glimmer-30b`**. Muse uses reasoning before
answering; a very small output budget can expire before an answer appears.
The server uses the checkpoint's chat template, **high reasoning strength**,
and Muse-specific tool and reasoning parsers. It preserves the checkpoint's
sampling defaults: temperature 1.0, top-p 0.95, and top-k 64. This exercise is
text-only, with one active request and an 88% GPU-memory budget.
See [vLLM's model support](https://docs.vllm.ai/en/stable/models/supported_models/)
and [tool-calling documentation](https://docs.vllm.ai/en/stable/features/tool_calling/).

### Repair a fresh copy of the exercise

Inside the **same compute shell**:

```bash
mkdir -p "$AGENT_ROOT/work"
cp -R "$TUTORIAL_DIR/examples/repair-mean" "$AGENT_ROOT/work/muse-agent"
cd "$AGENT_ROOT/work/muse-agent"
cp "$TUTORIAL_DIR/config/opencode-muse.json" opencode.json
python -m unittest -v
opencode
```

Use a new directory name if `muse-agent` already exists. The tests should fail
before the repair. In OpenCode, confirm the model is
**`pinnacles/muse-glimmer-30b`**, then enter:

> Read summary.py and test_summary.py. Fix mean_readings in summary.py so that it
> ignores None, includes zero, and raises ValueError when no readings remain.
> Do not edit test_summary.py. Use your file-editing tool to make the change,
> then run python -m unittest -v. Report the test result.

Review and approve the file edit and test command when prompted. After the
agent finishes, exit OpenCode and verify the result yourself:

```bash
python -m unittest -v
diff -u "$TUTORIAL_DIR/examples/repair-mean/summary.py" summary.py
cmp "$TUTORIAL_DIR/examples/repair-mean/test_summary.py" test_summary.py
```

All four tests should pass, `summary.py` should differ, and `cmp` should print
nothing. The provider configuration reserves an output budget of 8,192 tokens
within the 131,072-token (128K) context window.

When finished, stop the server from the compute shell that launched it and
release the allocation:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

On the login node, use `squeue --me` to confirm the allocation has ended.

## 10. Run NVIDIA Nemotron 3 Super through OpenCode

**Harness choice:** the steps below use OpenCode. For Codex, see
[Section 16](#16-choose-codex-instead-of-opencode), including the validation
status of the `nemotron` profile, before substituting the agent launch step.

After Gemma and Meta, repeat the workflow with the official
**NVIDIA Nemotron 3 Super 120B-A12B NVFP4 checkpoint on one H200**. Use 8 CPU
cores, 192 GB system RAM, and a 32,768-token context limit. Stop any earlier
model server first because each lesson uses port 8000.
Finish the previous lesson's shutdown and return to the login node before
requesting this new allocation.

Budget an additional **100 GB free in data** and **20 GB free in scratch**.
The checkpoint occupies about 75 GiB on disk and its isolated environment
about 7.7 GiB. The scripts use the same pinned Python 3.11, vLLM 0.29.0, and
OpenCode 1.18.30 versions as the other lessons. The dependency list is in
[`env/nemotron-requirements.lock`](env/nemotron-requirements.lock).

Validated on Pinnacles: chat, streaming, an automatic
tool-call round trip, and an OpenCode repair with completed read, edit, and test
tool calls. All four independent tests passed and the test file stayed
unchanged. Initial server startup took about **5½ minutes**. Loading the model
reported **69.46 GiB**; device memory after the agent run was **126,171 MiB**
(about 123.2 GiB), including reserved cache and runtime state. This validates a
short coding session at a configured 32K ceiling, not a full-context stress
test. vLLM also warned that the checkpoint does not provide calibrated FP8
attention q/probability scales. The short repair passed, but longer-session
accuracy with this cache setting has not been measured.

### Install and download on a CPU allocation

On the **login node**, from the tutorial directory:

```bash
module load anaconda3/2023.09-0
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 bash scripts/setup-nemotron.sh
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=96G --time=02:00:00 bash scripts/download-nemotron.sh
```

The download can use substantial system RAM while transferring its 17 weight
shards, so this lesson requests 96 GB. Wait for both commands to finish. The
scripts install the runtime at
`/data/$USER/pinnacles-agents/envs/nemotron-vllm` and download **80.32 GB of
weights** to
`/data/$USER/pinnacles-agents/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.
The checkpoint is pinned to revision
`ff433f5493e25d631c9f12b5d55c674229923d02`. Rerunning the command reuses
completed files. Consult [NVIDIA's model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
for the model license and intended use.

### Start Nemotron on an H200

On the **login node**, request a fresh compute shell:

```bash
srun --partition=cenvalarc.gpu --nodes=1 --ntasks=1 \
  --gres=gpu:nvidia_h200_nvl:1 --cpus-per-task=8 --mem=192G \
  --time=01:00:00 --pty bash
```

Inside the **compute shell**:

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-nemotron.sh
hostname
echo "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/nemotron-${SLURM_JOB_ID}.log"
bash scripts/serve-nemotron.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

Wait for `Application startup complete`, then press **Ctrl-C to leave `tail`**.
The background server continues running. Check it from this **compute shell**:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3-super",
    "messages": [{"role": "user", "content": "Explain a Python list comprehension in two sentences."}],
    "max_tokens": 4096
  }'
```

The model list should contain **`nemotron-3-super`**. The checkpoint mixes FP8
and NVFP4 quantization. H200 does not execute FP4 natively, so the tested vLLM
configuration keeps the compressed weights and uses its Marlin weight-only FP4
kernel for the NVFP4 layers. It also uses an FP8 KV cache, one active request,
NVIDIA's thinking template, the `nemotron_v3` reasoning parser, and the
`qwen3_coder` tool parser. These choices are encoded in
`scripts/serve-nemotron.sh`.

### Repair a fresh copy of the exercise

Inside the **same compute shell**:

```bash
mkdir -p "$AGENT_ROOT/work"
cp -R "$TUTORIAL_DIR/examples/repair-mean" "$AGENT_ROOT/work/nemotron-agent"
cd "$AGENT_ROOT/work/nemotron-agent"
cp "$TUTORIAL_DIR/config/opencode-nemotron.json" opencode.json
python -m unittest -v
opencode
```

Use a new directory name if `nemotron-agent` already exists. The tests should
fail before the repair. In OpenCode, confirm the model is
**`pinnacles/nemotron-3-super`**, then enter:

> Read summary.py and test_summary.py. Fix mean_readings in summary.py so that it
> ignores None, includes zero, and raises ValueError when no readings remain.
> Do not edit test_summary.py. Use your file-editing tool to make the change,
> then run python -m unittest -v. Report the test result.

Review and approve the file edit and test command when prompted. After the
agent finishes, exit OpenCode and verify the result yourself:

```bash
python -m unittest -v
diff -u "$TUTORIAL_DIR/examples/repair-mean/summary.py" summary.py
cmp "$TUTORIAL_DIR/examples/repair-mean/test_summary.py" test_summary.py
```

All four tests should pass, `summary.py` should differ, and `cmp` should print
nothing. The provider configuration reserves 8,192 output tokens within the
32K context window.

When finished, stop the server and release the allocation:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

On the login node, use `squeue --me` to confirm the allocation has ended.

## 11. Run quantized Meta Muse Glimmer on an A100 GPU

**Harness choice:** the steps below use OpenCode. For Codex, see
[Section 16](#16-choose-codex-instead-of-opencode), including the validation
status of the `muse-gguf` profile, before substituting the agent launch step.

When H200s are busy or when working under standard GPU queues, you can run the
official **quantized Meta Muse Glimmer 30B checkpoint on one A100 40 GB GPU**
using the `gpu` partition. This profile serves the official
`Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` checkpoint through a pinned,
standalone build of **llama.cpp with CUDA support**, avoiding large Python
virtual environments while fitting completely within 40 GB of VRAM.

Budget approximately **26 GB free in data** (about 18.3 GiB for model weights and
additional room for the runtime and OpenCode tools) and **1 GB free in scratch**. Pinned versions and
checksums are recorded in [`env/muse-gguf-pins.sh`](env/muse-gguf-pins.sh).

Validated on Pinnacles: chat with reasoning,
streaming, multi-chunk argument tool calling, full 16K and 32K context window
boundaries with over-context refusal, and 12 out of 12 OpenCode coding benchmark
trials passing across four tasks. Warm startup took **5 seconds**; weight loading
offloaded **53/53 layers** into **18,010 MiB** of GPU memory. Peak VRAM during
active 16K coding and 32K generation was **18,981 MiB** and **19,201 MiB**
respectively (under 19 GiB on a 40 GB A100). Near-context decoding achieved
**39.3–39.7 tokens/s**.

Those context and memory measurements used one server slot, which is also the
current default. `MUSE_CONTEXT` specifies context per slot. Increasing
`MUSE_PARALLEL` increases total cache demand and requires new memory measurements.

### Install and download on a CPU allocation

From the tutorial directory on the **login node**:

```bash
module load anaconda3/2023.09-0
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=16G --time=00:30:00 bash scripts/setup-muse-gguf.sh
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=01:00:00 bash scripts/download-muse-gguf.sh
```

The build script compiles llama.cpp `b10353` with CUDA 13, SM80 (A100) and SM89
(L40S) support, and installs it under `/data/$USER/pinnacles-agents/tools/llama.cpp-f8def7fe168bab245fbf15d3f18b26dbb1ef73c8`.
The download script fetches the official text GGUF (19,653,960,832 bytes) to
`/data/$USER/pinnacles-agents/models/Muse-Glimmer-30B-GGUF-70bf1b61ac09f91b24d39038091b41c582bc5d7a`
and verifies its SHA-256 digest (`ac7023d6...`). Download requires a 32 GB RAM
allocation for file hashing. Rerunning either script reuses verified files.
Consult [Meta's model card](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
for its license and usage information.

### Start the quantized Muse server on an A100

On the **login node**, request an interactive compute shell on the `gpu` partition:

```bash
srun --partition=gpu --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G \
  --time=01:00:00 --pty bash
```

Inside the **compute shell**:

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-muse-gguf.sh
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
```

The GPU name should confirm `NVIDIA A100-PCIE-40GB` (40,960 MiB). Now launch the
server in the background:

```bash
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/muse-gguf-${SLURM_JOB_ID}.log"
MUSE_GPU=a100 bash scripts/serve-muse-gguf.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

The server loads weights in approximately 5 seconds. When you see
`all slots are idle`, press **Ctrl-C to leave `tail`**. The background server
remains running. Verify the endpoint:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

You can test basic chat completion and high reasoning from the command line:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "muse-glimmer-30b-dynamic",
    "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the answer."}],
    "temperature": 1.0,
    "max_tokens": 4096
  }'
```

The response includes separate `reasoning_content` and final answer `391`.

### Connect OpenCode and run the coding exercise

In the **same compute shell**, prepare an isolated exercise directory:

```bash
mkdir -p "$AGENT_ROOT/work"
cp -a "$TUTORIAL_DIR/examples/repair-mean" "$AGENT_ROOT/work/repair-mean-muse-gguf"
cd "$AGENT_ROOT/work/repair-mean-muse-gguf"
cp "$TUTORIAL_DIR/config/opencode-muse-gguf.json" opencode.json
```

Before running OpenCode, confirm that the baseline tests fail:

```bash
python -m unittest -v
```

Now start OpenCode:

```bash
opencode
```

In the OpenCode prompt:

```text
Read summary.py and test_summary.py. Fix mean_readings in summary.py so that it ignores None, includes zero, and raises ValueError when no readings remain. Do not edit test_summary.py. Use your file-editing tool to make the change, then run python -m unittest -v. Report the test result.
```

OpenCode will read the files, apply a focused edit to `summary.py`, and run the
unit tests. When the agent finishes, exit OpenCode and independently verify the
results:

```bash
python -m unittest -v
diff -u "$TUTORIAL_DIR/examples/repair-mean/summary.py" summary.py
cmp "$TUTORIAL_DIR/examples/repair-mean/test_summary.py" test_summary.py
```

All four tests should pass, `summary.py` should differ, and `cmp` should print
nothing. The server and provider default to 32,768 context tokens and an
8,192-token output budget. Input and output share that context window.

When finished, stop the server and release the allocation:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

On the login node, use `squeue --me` to confirm the allocation has ended.

### Troubleshooting the quantized Muse exercise

| Symptom | What to check |
|---|---|
| `Cannot verify Slurm job` or cgroup check fails | `serve-muse-gguf.sh` requires running inside a real Slurm compute allocation (`gpu` or `cenvalarc.gpu`), not on a login node. |
| `MUSE_GPU must be a100 or l40s` | Set `MUSE_GPU=a100` before invoking `serve-muse-gguf.sh`. |
| `Run download-muse-gguf.sh in a CPU allocation` | Checkpoint is missing or failed SHA-256 verification. Run `scripts/download-muse-gguf.sh` in a short CPU allocation with at least 32 GB RAM. |
| `Run setup-muse-gguf.sh to install the pinned CUDA runtime` | Pinned llama.cpp runtime is missing or checksum failed. Run `scripts/setup-muse-gguf.sh` in a short CPU allocation. |
| `Port 8000 already in use` | Check your server log. If the listener is yours, stop that earlier server; otherwise set `MUSE_PORT=8001` (or another free port) and update OpenCode's `baseURL`. |
| Out of GPU memory | Confirm you allocated an A100 (40 GB) and no other processes occupy the GPU with `nvidia-smi`. |
| `Connection refused` | Wait ~5–10 seconds for the llama-server HTTP listener to initialize; inspect `$MODEL_SERVER_LOG`. |

## 12. Run Google Gemma 4 31B IT QAT on an A100 or L40S GPU

**Harness choice:** the steps below use OpenCode. For Codex, see
[Section 16](#16-choose-codex-instead-of-opencode), including the validation
status of the `gemma-qat` profile, before substituting the agent launch step.

When H200s are busy or when working under standard GPU queues, you can also run
the official **Google Gemma 4 31B IT QAT (INT4 W4A16) checkpoint on one A100 40 GB
GPU** on the `gpu` partition, or on **one L40S 48 GB GPU** on the `cenvalarc.gpu`
partition. This profile serves the official `google/gemma-4-31B-it-qat-w4a16-ct`
checkpoint through vLLM using compressed-tensors and Marlin INT4 kernels,
using a single active inference slot by default to leave room for conversation
history. Subagent delegation is enabled in OpenCode.

Budget approximately **35 GiB free in data** for a fresh install (about 21.7 GiB
for model weights and 7.7 GiB for the vLLM environment, plus tools)
and **10 GB free in scratch**. Pinned versions and
checksums are recorded in [`env/gemma-qat-pins.sh`](env/gemma-qat-pins.sh).

Validated on Pinnacles: chat with exact answers,
streaming, tool-call round trip, streaming tool calling, over-context HTTP 400
rejection, and OpenCode coding benchmarks passing across four tasks. The server
loaded weights into **18.7 GiB** of VRAM in 22 seconds, allocated **13.71 GiB**
of KV cache (43,471 tokens, 2.65x concurrency at 16K context), and ran with
**32.8 GiB** total memory usage on an A100 40GB. A dedicated subagent spawning
test verified that OpenCode's `task` tool can launch a child agent session and
return its result. Coding checks passed 11/12 trials (recovery: 2/3), meeting
the original acceptance gate. These are small smoke tests, not a comparative
benchmark. L40S support is implemented but has not been validated on Pinnacles.

### Install and download on a CPU allocation

From the tutorial directory on the **login node**:

```bash
module load anaconda3/2023.09-0
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=16G --time=00:30:00 bash scripts/setup-gemma-qat.sh
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --time=01:00:00 bash scripts/download-gemma-qat.sh
```

The setup script verifies the Python 3.11 environment (`gemma-vllm`), installs
OpenCode 1.18.30, and downloads the Gemma 4 tool chat template. The download
script fetches the official quantized safetensors checkpoint (23,265,352,448
bytes) to `/data/$USER/pinnacles-agents/models/gemma-4-31B-it-qat-w4a16-ct-52f3f65bc7a02d555763bc923bd1d9094898219d`
and verifies its SHA-256 digest (`1b9b1d62...`). Download requires a 32 GB RAM
allocation for file hashing. Rerunning either script reuses verified files.
Consult [Google's model card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)
for its license and usage terms.

### Start the Gemma QAT server on an A100

On the **login node**, request an interactive compute shell on the `gpu` partition:

```bash
srun --partition=gpu --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G \
  --time=01:00:00 --pty bash
```

Inside the **compute shell**:

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-gemma-qat.sh
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
```

The GPU name should confirm `NVIDIA A100-PCIE-40GB` (40,960 MiB) or `NVIDIA L40S`
(46,068 MiB). Now launch the server in the background:

```bash
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/gemma-qat-${SLURM_JOB_ID}.log"
GEMMA_GPU=a100 bash scripts/serve-gemma-qat.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

Wait for `Application startup complete`, then press **Ctrl-C to leave `tail`**.
The background server continues running. Check it from this **compute shell**:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-31b-qat",
    "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the answer."}],
    "max_tokens": 2048
  }'
```

The model list should contain **`gemma-4-31b-qat`** and the chat output should
return **`391`**. The server uses vLLM's `compressed-tensors` Marlin INT4 kernel,
Triton attention, FlashInfer top-p/top-k sampling, and `--max-num-seqs 1` by
default. Additional requests queue while the active request runs.

### Connect OpenCode and run the coding exercise

In the **same compute shell**, prepare an isolated exercise directory:

```bash
mkdir -p "$AGENT_ROOT/work"
cp -a "$TUTORIAL_DIR/examples/repair-mean" "$AGENT_ROOT/work/repair-mean-gemma-qat"
cd "$AGENT_ROOT/work/repair-mean-gemma-qat"
cp "$TUTORIAL_DIR/config/opencode-gemma-qat.json" opencode.json
```

Before running OpenCode, confirm that the baseline tests fail:

```bash
python -m unittest -v
```

Now start OpenCode:

```bash
opencode
```

In the OpenCode prompt:

```text
Read summary.py and test_summary.py. Fix mean_readings in summary.py so that it ignores None, includes zero, and raises ValueError when no readings remain. Do not edit test_summary.py. Use your file-editing tool to make the change, then run python -m unittest -v. Report the test result.
```

OpenCode will read the files, apply a focused edit to `summary.py`, and run the
unit tests. When the agent finishes, exit OpenCode and independently verify the
results:

```bash
python -m unittest -v
diff -u "$TUTORIAL_DIR/examples/repair-mean/summary.py" summary.py
cmp "$TUTORIAL_DIR/examples/repair-mean/test_summary.py" test_summary.py
```

All four tests should pass, `summary.py` should differ, and `cmp` should print
nothing. The default provider configuration reserves 8,192 output tokens within
a 32,768-token context window.

### Spawning subagents in OpenCode

All OpenCode profiles in this tutorial are configured with `"task": "allow"`
under permissions. When OpenCode encounters a task that benefits from division
of labor (such as researching background files, searching large directories, or
running sub-tasks), it uses its built-in `task` tool to spawn a subagent.

All profiles default to one server sequence slot. Delegation does not by itself prove overlapping
inference: the parent may wait while its child runs, and requests beyond the
available slots may queue. You can ask OpenCode directly to delegate a task:

```text
Use your task tool to spawn a subagent that inspects test_summary.py and reports the expected error cases.
```

When finished, stop the server and release the allocation:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

On the login node, use `squeue --me` to confirm the allocation has ended.

### Troubleshooting the Gemma QAT exercise

| Symptom | What to check |
|---|---|
| `Cannot verify Slurm job` or cgroup check fails | `serve-gemma-qat.sh` requires running inside a real Slurm compute allocation (`gpu` or `cenvalarc.gpu`), not on a login node. |
| `GEMMA_GPU must be a100 or l40s` | Set `GEMMA_GPU=a100` (or `l40s`) before invoking `serve-gemma-qat.sh`. |
| `Run download-gemma-qat.sh in a CPU allocation` | Checkpoint is missing or failed SHA-256 verification. Run `scripts/download-gemma-qat.sh` in a short CPU allocation with at least 32 GB RAM. |
| `compressed-tensors package missing` | Verify `gemma-vllm` environment with `scripts/setup-gemma-qat.sh`. |
| `Port 8000 already in use` | Check your server log. If the listener is yours, stop that earlier server; otherwise set `GEMMA_PORT=8001` (or another free port) and update OpenCode's `baseURL`. |
| Out of GPU memory | Confirm you allocated an A100 (40 GB) or L40S (48 GB) with `nvidia-smi` and that no other processes occupy the GPU. |
| `Connection refused` | Wait ~1–2 minutes for vLLM model loading, torch.compile, and CUDA graph capture to finish; inspect `$MODEL_SERVER_LOG`. |

## 13. Run NVIDIA Nemotron 3.5 Lightning 30B NVFP4 on an A100 or L40S GPU

**Harness choice:** the steps below use OpenCode. For Codex, see
[Section 16](#16-choose-codex-instead-of-opencode), including the validation
status of the `nemotron-lightning` profile, before substituting the agent launch step.

This lesson shows how to run **NVIDIA Nemotron 3.5 Lightning 30B NVFP4** on a
single **NVIDIA A100 (40 GB)** or **L40S (48 GB)** GPU. Nemotron 3.5 Lightning
uses a hybrid Mixture-of-Experts (MoE) + Mamba architecture with 3.5B active
parameters per token and 30B total parameters across 52 layers. It features
official mixed-precision quantization: W4A16 NVFP4 on MoE experts, FP8 on linear
projections. KV-cache precision is a separate runtime setting; the supplied
A100 path uses the runtime default, while the experimental L40S path requests FP8.

The model is served through an isolated vLLM 0.29.0 environment with
`--max-model-len 32768` and `--max-num-seqs 1`, with an 8,192-token
OpenCode output budget. L40S support is implemented but
has not been validated on Pinnacles; the measurements below are A100-only.

The earlier 16K, two-slot smoke test measured:

- **GPU Resource**: 1 A100 PCIe 40GB (`gnode002`), 8 CPUs, 128 GB RAM.
- **Model Loading**: 19.17 GiB VRAM loaded in 26.09 seconds using Humming kernels
  (`HummingNvFp4LinearKernel`, `HummingFP8ScaledMMLinearKernel`, FlashInfer attention).
- **KV Cache**: 14.92 GiB allocated (1,705,301 tokens, 104x concurrency at 16K context).
  These are runtime capacity estimates, not measured request concurrency;
  `--max-num-seqs 2` still limits active sequences. Total VRAM footprint:
  ~35.56 GiB out of 40 GB.
- **API Validation**: Health 200, models list, chat answer 391 in 1.55s (270 reasoning tokens),
  tool-call roundtrip, streaming tool call (18 chunks), streaming content (0.74s to first token),
  over-context HTTP 400 rejection, and cancellation recovery.
- **OpenCode Benchmarks**:
  - `repair-mean`: 3/3 passed (11.5s, 11.0s, 8.1s)
  - `signature`: 3/3 passed (14.4s, 16.2s, 15.2s)
  - `cli-validation`: 3/3 passed (12.7s, 16.6s, 16.0s)
  - `recovery`: 3/3 passed (11.0s, 11.1s, 9.7s)
  - `subagent-spawn`: 1/1 passed (6.88s)! OpenCode invoked the `task` tool, spawned
    a child subagent session, inspected the file, returned the secret token, and the
    parent reported the result. `EVAL_GATE`: True.

### Install and download on a CPU allocation

On the **login node**, from the tutorial directory:

```bash
module load anaconda3/2023.09-0
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 bash scripts/setup-nemotron-lightning.sh
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --time=01:00:00 bash scripts/download-nemotron-lightning.sh
```

The script downloads the 52 safetensors shards totaling **20.08 GiB (21.56 GB)**
to `/data/$USER/pinnacles-agents/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-bee7596271d1495f6992ae224aefde4410e816b8`,
verifies each shard against the pinned SHA-256 manifest, and writes an atomic
`.pinnacles-complete.json` marker. Consult [NVIDIA's model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
for the OpenMDW-1.1 model license.

### Start Nemotron Lightning on an A100 or L40S

On the **login node**, request an interactive GPU allocation:

```bash
# For an A100 40 GB:
srun --partition=gpu --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=128G \
  --time=01:00:00 --pty bash

# Or for an L40S 48 GB:
srun --partition=cenvalarc.gpu --nodes=1 --ntasks=1 \
  --gres=gpu:l40s:1 --cpus-per-task=8 --mem=128G \
  --time=01:00:00 --pty bash
```

Inside the **compute shell**:

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
source scripts/activate-nemotron-lightning.sh
hostname
echo "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Set NEMOTRON_GPU to match your allocation: a100 or l40s
export NEMOTRON_GPU=a100

mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/nemotron-lightning-${SLURM_JOB_ID}.log"
bash scripts/serve-nemotron-lightning.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

Wait for `Application startup complete`, then press **Ctrl-C to leave `tail`**.
The background server continues running. Test the endpoint from this **compute shell**:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3.5-lightning",
    "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the answer number."}],
    "temperature": 0.0,
    "max_tokens": 2048
  }'
```

The model list should contain **`nemotron-3.5-lightning`** and the chat output should
return **`391`**.

### Connect OpenCode and run the coding exercise

In the **same compute shell**, prepare an isolated exercise directory:

```bash
mkdir -p "$AGENT_ROOT/work"
cp -a "$TUTORIAL_DIR/examples/repair-mean" "$AGENT_ROOT/work/repair-mean-nemotron-lightning"
cd "$AGENT_ROOT/work/repair-mean-nemotron-lightning"
cp "$TUTORIAL_DIR/config/opencode-nemotron-lightning.json" opencode.json
```

Confirm that the baseline tests fail:

```bash
python -m unittest -v
```

Now start OpenCode:

```bash
opencode
```

In the OpenCode prompt:

```text
Read summary.py and test_summary.py. Fix mean_readings in summary.py so that it ignores None, includes zero, and raises ValueError when no readings remain. Do not edit test_summary.py. Use your file-editing tool to make the change, then run python -m unittest -v. Report the test result.
```

OpenCode will read the files, apply the edit to `summary.py`, and run the unit tests.
When the agent finishes, exit OpenCode and independently verify the results:

```bash
python -m unittest -v
diff -u "$TUTORIAL_DIR/examples/repair-mean/summary.py" summary.py
cmp "$TUTORIAL_DIR/examples/repair-mean/test_summary.py" test_summary.py
```

### Spawning subagents in OpenCode

In `config/opencode-nemotron-lightning.json`, subagent delegation is enabled:
`"task": "allow"`. The server defaults to one active sequence; delegation works
with queued requests and does not establish simultaneous inference. You can instruct
OpenCode directly:

```text
Spawn 5 parallel subagents, to tell me the top 5 news on Hacker news.
```

When finished, stop the server and release the allocation:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

On the login node, use `squeue --me` to confirm the allocation has ended.

### Troubleshooting the Nemotron Lightning exercise

| Symptom | What to check |
|---|---|
| `Cannot verify Slurm job` or cgroup check fails | `serve-nemotron-lightning.sh` requires running inside a real Slurm compute allocation (`gpu` or `cenvalarc.gpu`), not on a login node. |
| `NEMOTRON_GPU must be a100 or l40s` | Set `NEMOTRON_GPU=a100` (or `l40s`) before invoking `serve-nemotron-lightning.sh`. |
| `Checkpoint size mismatch` or missing shards | Run `scripts/download-nemotron-lightning.sh` in a short CPU allocation with at least 32 GB RAM. |
| `Port 8000 already in use` | Check your server log. Stop any earlier server or set `NEMOTRON_PORT=8001` and update OpenCode's `baseURL`. |
| Out of GPU memory | Confirm you allocated an A100 (40 GB) or L40S (48 GB) with `nvidia-smi` and that no other processes occupy the GPU. |
| Cold startup latency (~4–6 minutes) | Nemotron Lightning warms up Mamba2 SSD Triton kernels and captures CUDA graphs on first start; monitor `$MODEL_SERVER_LOG` until `Application startup complete`. |

## 14. Scale context to 128K on multiple GPUs (dual A100 or L40S)

**Harness choice:** Muse GGUF also has a validated Codex coding route; see
[Section 16](#16-choose-codex-instead-of-opencode) and choose `muse-gguf-128k`.
The `gemma-qat-128k` Codex configuration is available but not yet runtime-validated.

Long coding sessions accumulate source files, tool output, and earlier answers.
Splitting a model across GPUs can leave more memory for this conversation.
The supplied launchers configure **131,072 context tokens and 8,192 output
tokens on two GPUs on one node**. Output is part of the context budget, so
reserve at least 8,192 tokens for the answer, plus room for chat and tool overhead.

Gemma QAT uses vLLM tensor parallelism (`--tensor-parallel-size 2`), which
divides tensor operations across GPUs. Muse GGUF uses llama.cpp layer splitting
(`-sm layer -ts 1,1`), which assigns layers to different GPUs. These approaches
have different communication costs; adding a GPU does not guarantee faster
generation or change the model's trained context limit.
See [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
and the [pinned llama.cpp server reference](https://github.com/ggml-org/llama.cpp/blob/f8def7fe168bab245fbf15d3f18b26dbb1ef73c8/tools/server/README.md).

**Validation scope:** both launchers have recorded successful startup and short
chat on two A100 40 GB GPUs with a configured 128K ceiling. Full-window
generation, long-context tool use, and OpenCode repair at 128K remain
unverified. L40S is supported by the scripts but has not been validated on
Pinnacles. Treat that hardware path as experimental.

### Why memory use differs between models

Both models combine sliding-window attention with full attention. Sliding
layers can retain a bounded recent history, while full-attention layers retain
longer histories. Actual cache allocation depends on the runtime, head counts,
precision, and concurrent requests.

For Muse's 13 full-attention layers, two KV heads, 128-dimensional heads, and
16-bit cache, the full-attention component is approximately
`13 × 2 (K and V) × 2 × 128 × 2 bytes = 13,312 bytes/token`.
At 131,072 tokens that is 1.625 GiB, **before sliding-layer cache, allocation
rounding, temporary tensors, and runtime overhead**. Use the server's measured
allocation rather than treating this estimate as the entire memory footprint.

Both launchers default to one active request. vLLM's startup concurrency
estimate describes cache capacity; it does not override `--max-num-seqs 1`.
llama.cpp's launcher multiplies per-request context by the number of slots.
Increasing slots or context requires another memory and generation check.

### Prepare the model and request two GPUs

First complete the setup and download in Section 12 for Gemma QAT or Section
11 for Muse GGUF. The multi-GPU launchers reuse those pinned installations.
Stop your previous server and release its allocation before requesting a new one.

On the **login node**, request two A100s on one node:

```bash
srun --partition=gpu --nodes=1 --ntasks=1 \
  --gres=gpu:a100:2 --cpus-per-task=16 --mem=128G \
  --time=02:00:00 --pty bash
```

For the experimental L40S path, use `--partition=cenvalarc.gpu` and
`--gres=gpu:l40s:2` instead. Inside the **compute shell**:

```bash
cd ~/OpenSource_Agentic_Model_Setup
export TUTORIAL_DIR="$PWD"
hostname
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.total --format=csv
```

Confirm two GPUs of the requested type. Keep Slurm's `CUDA_VISIBLE_DEVICES`
setting; the launchers check those devices.

### Option A: Gemma QAT with vLLM

Inside the compute shell, from the tutorial directory:

```bash
source scripts/activate-gemma-qat.sh
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/gemma-qat-tp2-$SLURM_JOB_ID.log"
GEMMA_GPU=a100 GEMMA_CONTEXT=131072 GEMMA_OUTPUT=8192 GEMMA_PARALLEL=1 \
  bash scripts/serve-gemma-qat-multigpu.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

Use `GEMMA_GPU=l40s` for an L40S allocation. Wait for
`Application startup complete`, then press Ctrl-C to leave `tail`.
Cached compilation can shorten startup, but its cache-load timing excludes
checkpoint loading, worker startup, and other initialization.

Check the endpoint from the same compute shell:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31b-qat","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":2048}'
```

Look for model alias `gemma-4-31b-qat` and answer `391`. This is a short
chat check, not a test of the full context window.

For an OpenCode session, create a fresh project directory:

```bash
mkdir -p "$AGENT_ROOT/work/gemma-128k"
cd "$AGENT_ROOT/work/gemma-128k"
cp "$TUTORIAL_DIR/config/opencode-gemma-qat-128k.json" opencode.json
opencode
```

The configuration advertises 131,072 context tokens and 8,192 output tokens.
Before relying on it for a large repository, repeat the small edit-and-test
exercise, then test progressively longer prompts while monitoring memory.

### Option B: Muse GGUF with llama.cpp

Choose this option in place of Gemma. If switching within the allocation,
exit OpenCode, stop the server using its saved PID, and wait for it to exit
before starting Muse on the same port. Return to the tutorial directory:

```bash
cd "$TUTORIAL_DIR"
source scripts/activate-muse-gguf.sh
mkdir -p "$AGENT_ROOT/logs"
export MODEL_SERVER_LOG="$AGENT_ROOT/logs/muse-gguf-multigpu-$SLURM_JOB_ID.log"
MUSE_GPU=a100 MUSE_CONTEXT=131072 MUSE_OUTPUT=8192 MUSE_PARALLEL=1 \
  bash scripts/serve-muse-gguf-multigpu.sh > "$MODEL_SERVER_LOG" 2>&1 &
export MODEL_SERVER_PID=$!
tail -f "$MODEL_SERVER_LOG"
```

Use `MUSE_GPU=l40s` for an L40S allocation. Wait for the HTTP listener and
`all slots are idle`, then press Ctrl-C to leave `tail`.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30b-dynamic","messages":[{"role":"user","content":"What is 23 * 29?"}],"max_tokens":4096}'
```

Look for model alias `muse-glimmer-30b-dynamic` and answer `667`; reasoning
may appear in a separate response field. Then configure a fresh OpenCode project:

```bash
mkdir -p "$AGENT_ROOT/work/muse-128k"
cd "$AGENT_ROOT/work/muse-128k"
cp "$TUTORIAL_DIR/config/opencode-muse-gguf-128k.json" opencode.json
opencode
```

### Recorded dual-A100 startup measurements

These short-chat observations do not measure full-window accuracy or sustained
throughput. Device memory includes reserved cache and runtime state.

| Measurement | Gemma QAT, TP=2 | Muse GGUF, layer split |
|---|---|---|
| Configured context / output | 131,072 / 8,192 | 131,072 / 8,192 |
| Model weights | 9.58 GiB per GPU | Approximately 18 GiB across both GPUs |
| Device memory | 36,919 MiB per GPU (about 36.05 GiB each) | 10,751 and 11,435 MiB (about 21.67 GiB combined) |
| Cache capacity | Runtime estimate: 482,421 tokens; one active request configured | Depends on context and slot count |

### Troubleshooting and cleanup

| Symptom | What to check |
|---|---|
| GPU selection rejected | Request exactly two GPUs on one node, preserve Slurm's device list, and set the matching `GEMMA_GPU` or `MUSE_GPU`. |
| Out of memory | Keep one slot; lower server context to 65,536 or 32,768 and lower OpenCode's context limit to match. |
| NCCL or peer-to-peer error | Inspect the server log and `nvidia-smi topo -m`; communication support depends on node topology and runtime. |
| Port unavailable | Stop your earlier server, or set a free `GEMMA_PORT`/`MUSE_PORT` and update OpenCode's `baseURL` and curl checks. |
| Long prompt fails despite successful chat | Verify prompt plus output fits the configured context; short-chat success does not establish full-window capacity. |

When finished, exit OpenCode, stop the saved server PID, and release the allocation:

```bash
kill "$MODEL_SERVER_PID"
wait "$MODEL_SERVER_PID"
exit
```

Back on the login node, use `squeue --me` to check your remaining jobs.

## 15. Connect a model that is not listed

The same workflow applies to other instruction-tuned models:
choose compatible weights and a serving runtime, test the endpoint, then connect
OpenCode. The following is an adaptation guide; each new model needs its own
validation before you treat it as a working Pinnacles recipe.

### Choose and prepare the checkpoint

1. Read the model card for the exact checkpoint. Check its license, access
   requirements, total parameter count, supported context, and tool-calling
   format. Choose an instruction-tuned model with documented tool support.
2. Check the serving runtime's supported architectures and quantization kernels.
   A GGUF checkpoint usually uses llama.cpp; safetensors checkpoints may use
   vLLM if that model and precision are supported on your GPU. File size alone
   does not establish compatibility.
3. Budget weights, KV cache, and runtime overhead separately. Start with one
   request and a modest context. Use the GPU and storage reference below to
   select an allocation your account can access.
4. In a Slurm CPU allocation, install the required runtime in a separate
   environment and download the complete checkpoint: all shards, tokenizer,
   configuration, and chat template. Pin the model revision and runtime version;
   record checksums. Request any required model access before downloading.
   Use the existing setup/download scripts as examples of this structure.

Consult [vLLM model support](https://docs.vllm.ai/en/stable/models/supported_models/),
[vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/), and
[llama.cpp tool calling](https://github.com/ggml-org/llama.cpp/blob/f8def7fe168bab245fbf15d3f18b26dbb1ef73c8/docs/function-calling.md).
Keep model-specific templates, tool parsers, and reasoning parsers together.

### Start and test the server

Request a Slurm GPU shell using the earlier allocation examples. Activate your
new environment there. Follow the model's runtime recipe, bind the endpoint to
`127.0.0.1`, choose an unused port, and give it a short alias such as
`my-local-model`.

For a vLLM-supported model, the launch structure is shown below. Replace every
uppercase placeholder with a verified value from your checkpoint's recipe.
The 8,192-token context is an initial test setting, not a guaranteed fit.

```text
vllm serve /ABSOLUTE/PATH/TO/COMPLETE_CHECKPOINT \
  --served-model-name my-local-model \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --max-num-seqs 1 \
  --enable-auto-tool-choice --tool-call-parser MODEL_TOOL_PARSER
```

Add the required dtype, quantization, chat-template, and reasoning flags for
your model. For llama.cpp, select the complete GGUF, enable its supported
tool-calling template, and set the alias, port, context, and slot count using
the runtime's documented options. Do not copy another model's parser name.

From the same compute shell, check `/health` if supported and `/v1/models`.
Then send a short chat request:

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-local-model","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":2048}'
```

Confirm the alias and a complete response. Next test streaming, a structured
tool call, and a follow-up containing the tool result. If the model prints tool
syntax as ordinary text, check its template and parser before using OpenCode.

### Configure OpenCode and verify an edit

Install the tutorial's pinned OpenCode 1.18.30 if you have not already done so.
In a fresh project directory on the compute node, create `opencode.json`
with this v1 configuration. Merge with an existing file if your project already
has one.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "pinnacles/my-local-model",
  "small_model": "pinnacles/my-local-model",
  "enabled_providers": ["pinnacles"],
  "share": "disabled",
  "autoupdate": false,
  "provider": {
    "pinnacles": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "My Pinnacles model",
      "options": {"baseURL": "http://127.0.0.1:8000/v1"},
      "models": {
        "my-local-model": {
          "name": "My local model",
          "limit": {"context": 8192, "output": 2048}
        }
      }
    }
  },
  "permission": {
    "*": "ask",
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "task": "allow"
  }
}
```

The model key must equal the alias from `/v1/models`. Set OpenCode's context
limit no higher than the tested server limit, and reserve output space within
it. Increase the output allowance if reasoning consumes the initial budget,
then retest the combined input/output limit.

Run `opencode` from that directory. Repeat the Section 7 repair exercise on a
fresh copy: observe read, edit, and test tool calls, then independently rerun
the four tests and confirm the test file stayed unchanged. Only after that
passes should you increase context, enable more simultaneous requests, or
move to a larger project. Record the working model revision, launch command,
GPU memory, and OpenCode configuration so others can reproduce it.

Tools execute where OpenCode runs. When finished, stop your server and release
its Slurm allocation.

## 16. Choose Codex instead of OpenCode

The model server is independent of the coding harness. Keep the selected
lesson's installation, download, allocation, activation, and server startup
steps. Choose OpenCode or Codex when you reach the coding exercise; switching
harnesses does not require restarting the model or downloading its weights again.

### What has been validated

On Pinnacles, **Codex CLI 0.154.0 with Muse GGUF served by llama.cpp at a
configured 128K context limit** completed file reads, a Python implementation
repair, a test run, and a final response through the Responses API. The repository
launcher and pinned package were tested with the supplied repair exercise.
This was a short coding task, not a full-window or multi-agent stress test.
The existing OpenCode validation and benchmark results remain OpenCode results.

The shared launcher recognizes every checked-in model profile. The table
separates available configuration from completed Codex runtime validation:

| Profile | Configured context | Codex validation |
|---|---:|---|
| `muse-gguf-128k` | 131,072 | Short coding repair validated with llama.cpp |
| `muse-gguf` | 32,768 | Streamed file-read tool round trip at low effort; all four adapter strengths completed inference |
| `gemma` | 98,304 | Configuration available; vLLM/Codex tool round trip not yet validated |
| `muse` | 131,072 | Configuration available; vLLM/Codex tool round trip not yet validated |
| `gemma-qat` | 32,768 | Configuration available; vLLM/Codex tool round trip not yet validated |
| `gemma-qat-128k` | 131,072 | Configuration available; vLLM/Codex tool round trip not yet validated |
| `nemotron` | 32,768 | Configuration available; vLLM/Codex tool round trip not yet validated |
| `nemotron-lightning` | 32,768 | Configuration available; vLLM/Codex tool round trip not yet validated |
| `gemma-a100`, `muse-a100` | 32,768 | Historical A100 configurations; Codex not yet validated |

For the unvalidated combinations, continue using the lesson's tested OpenCode
route until the full read/edit/test interaction is checked. An OpenAI-compatible
Chat Completions endpoint alone does not guarantee Codex Responses compatibility.

### Install Codex once

From the tutorial checkout on the **login node**, request a CPU allocation:

```bash
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --mem=4G --time=00:10:00 bash scripts/setup-codex.sh
```

The installer downloads the complete Linux x86-64 Codex 0.154.0 package from
OpenAI's release, verifies its SHA-256, and installs it under
`/data/$USER/pinnacles-agents/tools/codex-v0.154.0`. No npm, sudo, paid OpenAI
account, or OpenAI API key is needed for this local-model route. Rerunning the
installer checks the installed package. It does not change your usual Codex
installation or login.

### Launch either harness with the running Muse model

After starting the **128K Muse GGUF server in Section 14**, stay in that
activated compute shell. Enter your project directory. The examples below
assume `TUTORIAL_DIR` still points to your clone, as established in the lesson.

Choose **OpenCode**. If your project already has `opencode.json`, merge the
provider settings instead of overwriting it with the copy command:

```bash
cp "$TUTORIAL_DIR/config/opencode-muse-gguf-128k.json" opencode.json
opencode
```

Or choose **Codex** in the same project directory:

```bash
python3 "$TUTORIAL_DIR/scripts/launch-codex.py" muse-gguf-128k
```

Accept the workspace trust prompt if shown. Enter the exercise's repair prompt
or your own coding task. Codex starts with workspace-write sandboxing and
on-request approvals; it may edit project files and run permitted commands
without asking for each action. These defaults differ from the OpenCode
profiles, which ask for edits and shell commands. Review changes and run tests
when either agent finishes. Exit Codex with `/quit` or Ctrl-C.

The launcher uses the selected profile's model ID, endpoint, and context limit
from `config/opencode-*.json`, so the two harnesses share those settings. It
checks `/v1/models` before launching and lowers its context limit if the server
advertises a smaller one. It does not start a server or reserve GPUs.
The current directory is the working project; it may be outside the tutorial.

The launcher also supplies a local model catalog: `/model` lists only the
selected profile's model, marked as current. Select it to choose its reasoning
effort; see [thinking levels](#changing-thinking-levels-in-either-harness).
Use `/status` to inspect the active model. A `default` label is not a switch to
an OpenAI model.

To check the endpoint and the catalog loaded by Codex without sending a prompt:

```bash
python3 "$TUTORIAL_DIR/scripts/launch-codex.py" --check muse-gguf-128k
```

The check reports the model ID, local provider, catalog, and isolated state
directory. Asking “who are you?” is not a configuration check: a model can
mistakenly quote `~/.codex/config.toml`, which belongs to your ordinary Codex
setup. The launcher's invocation settings take precedence for this session.
Always start this route through the launcher; typing bare `codex` uses your
ordinary setup. After updating the launcher, exit the old session and relaunch
to load the corrected catalog.

Codex configuration and sessions use
`/data/$USER/pinnacles-agents/codex/PROFILE`, separate from `~/.codex` and
OpenCode. Defaults are passed for this invocation rather than written into
project files. This also means your usual Codex login, plugins, and settings
are not automatically carried into the tutorial sessions.

List available profiles without connecting to a server:

```bash
python3 "$TUTORIAL_DIR/scripts/launch-codex.py" --list
```

For a server you deliberately started on another local port, pass
`--base-url http://127.0.0.1:8001/v1` **before** the profile name. Both clients
run on the same compute node as the server; keep its Slurm allocation alive.

### Project environments and troubleshooting

Both harnesses can use your project's venv or Conda environment. Select it
after model activation and record explicit interpreter/test commands in
`AGENTS.md`, as described in [the project environment section](#using-your-projects-own-environment).
The model's Python environment does not supply packages to Codex tool commands.

| Symptom | What to check |
|---|---|
| `Codex is not installed` | Run the separate CPU installation step above. Existing model setup scripts install OpenCode, not Codex. |
| `Run inside the model's Slurm compute shell` | Return to the active compute allocation and source the lesson's activation script. |
| Connection refused or model not advertised | Check that the server is ready on this node and that the profile and port match. The launcher does not silently switch models. |
| Muse missing from `/model`, or `Model metadata ... not found` | Update the checkout and restart through the launcher. Earlier launcher versions omitted the custom catalog; the current launcher supplies it. Use `--check` to verify. |
| Assistant claims it is GPT or quotes `~/.codex/config.toml` | Check `/status` and the launcher output. Generated identity claims are not runtime evidence. The local route uses its separate state directory and `pinnacles` provider. |
| `/responses` error, unsupported tool, or no edits | The exact server/model combination needs a Codex tool round-trip check. Use its tested OpenCode route while investigating. |
| Wrong Python or missing project package | Use the project's explicit interpreter or Conda command in `AGENTS.md`. |

See the official [Codex CLI documentation](https://learn.chatgpt.com/docs/codex/cli)
and [custom-provider configuration](https://learn.chatgpt.com/docs/config-file/config-advanced).
The automated benchmark runner still uses its pinned OpenCode adapter; this
addition provides a student-facing Codex launch route, not cross-harness benchmark results.

## Reference: GPUs, memory, and model selection

### GPU allocation

A **node** is a physical computer. A **partition** is a Slurm queue containing
nodes. You request a GPU type and count; Slurm assigns available hardware.

Public inventory observed in Slurm (check your allocation for current availability):

| Partition | GPU | Memory per GPU | GPUs per node | Nodes in queue | One-GPU request |
|---|---|---:|---:|---:|---|
| `gpu` | NVIDIA A100 PCIe | 40 GB | 2 | 8 | `--gres=gpu:a100:1` |
| `cenvalarc.gpu` | NVIDIA L40S | 48 GB | 2 | 8 | `--gres=gpu:l40s:1` |
| `cenvalarc.gpu` | NVIDIA H200 NVL | 141 GB | 2 | 4 | `--gres=gpu:nvidia_h200_nvl:1` |

GPU capacities are also listed in the [campus hardware reference](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/campus-clusters/#cluster-hardware-configuration).
An allocation check confirmed A100 at 40,960 MiB and H200 at 143,771 MiB.
The L40S capacity is from campus documentation; its allocation check could not
start while nodes were busy. Product GB labels and tool-reported MiB are
different conventions; use the memory actually reported by your allocation.

#### How many GPUs can you request?

Start with **one GPU on one node**. Change the final `:1` to `:2` to request both
GPUs on that node when the serving runtime can split your model across them.
Two A100s have 80 GB nominal combined memory; two L40S cards have 96 GB; two
H200s have 282 GB. These are separate memory pools. The server must explicitly
distribute the model, and GPU communication adds overhead.

Both public GPU queues currently allow **two nodes per job**, up to **three
days** of wall time, and **four submitted jobs per user under each queue's
QoS**. Submitted jobs include pending and running jobs; this is not a four-GPU
limit. Two nodes with two GPUs each offer a potential four-GPU allocation,
subject to your account limits and resource availability. Multi-node serving
also needs distributed runtime support and working inter-node communication;
it is outside this introductory lesson.

The two-node limit is a scheduler maximum. Use the shortest allocation that
covers your work, and release it when finished. Availability in `sinfo` does
not guarantee immediate scheduling. The optional `test` queue has a one-hour
limit and one submitted job per user.

On the **login node**, inspect the current inventory and account association:

```bash
sinfo -p gpu,cenvalarc.gpu -o '%P|%a|%l|%D|%G'
sacctmgr -n -P show assoc user="$USER" format=User,Account,Partition,QOS
```

The tutorial assumes a regular sponsored account. The live GPU partitions
exclude `project_ucm_guests`; if your account has that association or Slurm
rejects access, ask the Challenge organizers to resolve your account setup.

#### GPU memory, system RAM, and disk are different

**GPU memory (VRAM)** holds model weights and inference state. It is usually the
first constraint on which model you can run. **System RAM** supports loading,
tokenization, and CPU work. Slurm's `--mem=64G` requests system RAM; it does not
increase your GPU's VRAM. In particular, `sinfo`'s `MEMORY` column reports node
RAM, not GPU memory. `cenvalarc.gpu` currently caps a job's system RAM request
at 256,000 MiB per node, even where the hardware reports more installed RAM.

**Disk storage** keeps downloaded weights and environments between jobs. Having
500 GB of disk available does not let a 40 GB GPU hold a 500 GB model. CPU
offloading can move some weights into system RAM, but transfers can make an
interactive coding agent substantially slower.

### Storage

Each standard account has separate **500 GB allocations in `data` and
`scratch`**. Use `/data/$USER` for downloaded models and environments, and
`/scratch/$USER` for disposable build files and temporary output. Keep your
small tutorial checkout and source files in your home directory.

The published quotas are 500G soft / 512G hard for each of `data` and `scratch`.
Budget against 500 GB; the hard limit is not extra working space. These are
separate quotas, so unused scratch capacity does not increase your data quota.
Scratch can be purged; keep important code and results elsewhere.
[Campus storage reference](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/campus-clusters/#file-systems-and-storage)

Downloading BF16 and quantized copies consumes space for both. Leave room for
the serving environment, download staging, and caches. Download only the
checkpoint and format you intend to use. A filesystem-wide free-space reading
from `df` does not report your personal quota allowance.

### Model size, quantization, and KV cache

#### Estimate the space needed for weights

A parameter is a learned number in a model. “31B” means approximately 31 billion
parameters. As a first approximation:

```text
Weight storage ≈ total parameters × bits per weight ÷ 8
Required VRAM ≈ loaded weights + KV cache + temporary tensors + runtime overhead
```

For an illustrative 31-billion-parameter model:

| Weight representation | Approximate weight storage, decimal GB | Main trade-off |
|---|---:|---|
| BF16 / FP16, 16 bits | 62 GB | High precision, largest memory requirement |
| INT8 / FP8, 8 bits | 31 GB | Lower memory use; format and hardware support matter |
| 4-bit quantization | 15.5 GB before metadata | Much smaller weights; quality and speed depend on the quantization |

These are weight estimates, **not total VRAM requirements**. Quantization adds
scales and other metadata, and some tensors may remain at higher precision.
Use the actual checkpoint size when budgeting storage.

Quantization represents weights with fewer bits. It can make a larger model
fit and reduce memory traffic. It may also reduce accuracy, change tool-call
reliability, or require extra conversion work during inference. A smaller file
does not guarantee faster generation. GGUF is a file format used by llama.cpp;
the specific quantization inside it determines the precision. NVFP4 is another
format, with different runtime and hardware requirements. Do not treat all
four-bit checkpoints as interchangeable.

For a **mixture-of-experts (MoE)** model, the active parameter count describes
how much of the model participates in processing a token. Memory planning uses
the **total** parameter count. Nemotron Super's “120B-A12B” therefore does not
mean you only need storage for a 12B model.

#### Leave room for the conversation

The **KV cache** stores attention keys and values for previously processed
tokens, allowing the model to generate the next token without recomputing all
that attention state. A coding session accumulates instructions, source files,
tool outputs, reasoning, and generated text. More cached tokens and more
simultaneous requests generally need more memory.

The **context window** is the token budget for a request, including its input
and generated output. An advertised 256K or 1M window is not a promise that it
will fit your allocation. Long histories also increase prompt processing time.
Reserve output space rather than filling the entire window with source files.

KV-cache quantization is separate from weight quantization. It can free memory
for longer conversations, with its own accuracy and runtime-support trade-offs.
Sliding-window attention and hybrid Mamba/attention models have different cache
behavior, so a single cache-size formula will not accurately describe all
three model families. See [vLLM's KV-cache documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/).

If a server runs out of GPU memory **while loading**, the weights and startup
overhead may be too large. If it runs out **during generation**, investigate
context length, concurrent requests, and temporary tensors as well. A fit must
include successful generation at the intended context, not just loading weights.

### Model families and benchmarks

Use an instruction-tuned or agent-ready checkpoint for OpenCode. A pretrained
base model is not automatically ready to follow instructions and call tools.
Gemma, Meta, and NVIDIA have passed the coding exercise above.

| Model | Role in the tutorial | Capacity consideration |
|---|---|---|
| [Google Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31B-it) | Tested Gemma lesson | BF16 serving and the OpenCode exercise passed on one H200. A single public-queue A100 40 GB or L40S needs smaller weights; an 80 GB A100 baseline was also measured with separate partition access. |
| [Google Gemma 4 31B IT QAT](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) | Tested Gemma A100 lesson; L40S unvalidated | Official compressed-tensors INT4 W4A16 running on one A100 40 GB via vLLM Marlin. Chat, tools, subagent delegation, and 11/12 coding trials passed (recovery: 2/3). |
| [Meta Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | Tested Meta H200 lesson | BF16 serving and the OpenCode exercise passed on one H200 with vLLM. |
| [Meta Muse Glimmer 30B GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) | Tested Meta A100 lesson | Official Dynamic Q4_K_XL GGUF running via CUDA llama.cpp on one A100 40 GB. Chat, tools, streaming, 16K/32K contexts, and 12/12 coding trials passed. |
| [NVIDIA Nemotron 3 Super 120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | Tested NVIDIA lesson | Mixed FP8/NVFP4 serving and the OpenCode exercise passed on one H200 through vLLM's Marlin FP4 fallback. |
| [NVIDIA Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | Tested NVIDIA A100 lesson; L40S unvalidated | Official NVFP4 mixed-precision running on one A100 40 GB via vLLM Humming kernels. Chat, tools, subagent delegation, and 13/13 coding/subagent trials passed. |

Gemma 4 31B has 30.7B parameters. The BF16 H200 run reported **57.91 GiB** for
model loading. Device memory use after the agent exercise was **120,281 MiB**
(about 117.5 GiB), including the server's reserved cache and runtime state.
This allocation is not the minimum memory required to serve the model. vLLM
reserves cache according to its configured memory budget.

If H200s are busy, keep your job queued or cancel your own pending job and
return later. Changing only the GPU request to an A100 or L40S will not make
these H200 recipes fit. Quantized versions preserve larger models on
smaller GPUs: Section 11 validates the official Dynamic Q4 GGUF on a single A100,
Section 12 validates Gemma 4 31B QAT, and Section 13 validates Nemotron 3.5 Lightning.
Gemma 26B-A4B is another candidate; its active 4B count does not describe its
total weight memory.
[Google model card](https://ai.google.dev/gemma/docs/core/model_card_4)

Nemotron 3.5 Lightning 30B-A3B is validated on A100 in Section 13. Its
[official card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
describes a hybrid MoE model and deployments on Ampere, Ada, and Hopper GPUs. The
larger Super quantized card documents Blackwell examples; this tutorial uses a
separately validated H200 Marlin path for Super and an A100 Humming path for Lightning.

Nemotron 3 Ultra 550B-A55B is a further capacity candidate for a future
multi-node exercise. Four-bit weight arithmetic alone is about 275 GB before
metadata and higher-precision tensors. Its [official deployment guidance](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/raw/main/README.md)
uses larger configurations, including four Blackwell GPUs or eight H100s.
Neither that recipe nor aggregate memory arithmetic establishes a working
four-H200 Pinnacles deployment. **Ultra remains unvalidated on Pinnacles.**

#### What do the benchmark scores mean?

Benchmarks measure different abilities. For a coding agent, repository repair
and terminal tasks are more directly relevant than answering isolated questions.

| Benchmark | What is tested | What the score does not establish |
|---|---|---|
| [SWE-bench Verified](https://www.swebench.com/) | Resolving real repository issues, evaluated through tests | That every generated patch is maintainable or that your project will work |
| [Terminal-Bench](https://www.tbench.ai/benchmarks) | Completing multi-step tasks through a terminal | Equivalent results across different benchmark versions or harnesses |
| [LiveCodeBench](https://livecodebench.github.io/) | Coding problems evaluated against tests | Reliable repository navigation or tool use over a long session |
| [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Knowledge and reasoning across academic subjects | Practical software engineering ability |

Selected **publisher-reported** results. Scores are
percentages; higher is better within the same evaluation. These are not local
measurements and are not a controlled ranking across publishers.

| Model evaluated | Published result | Evaluation context and source |
|---|---|---|
| Gemma 4 31B | LiveCodeBench v6: **80.0%**; MMLU-Pro: **85.2%** | Google's instruction-tuned evaluation: [Google card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| Gemma 4 31B | SWE-bench Verified: **66.6%**; Terminal-Bench 2.1: **43.4%** | Meta's comparison, Gemma thinking mode; Terminal-Bench uses Terminus2: [Meta card](https://huggingface.co/meta-models/Muse-Glimmer-30B#benchmarks) |
| Muse Glimmer 30B | SWE-bench Verified: **76.0%**; Terminal-Bench 2.1: **51.7%** | Meta's high-reasoning evaluation; Terminal-Bench uses Terminus2: [Meta card](https://huggingface.co/meta-models/Muse-Glimmer-30B#benchmarks) |
| Nemotron 3 Super | SWE-bench Verified: **59.20% with OpenCode**, **60.47% with OpenHands** | NVIDIA's BF16 card, different harnesses: [NVIDIA card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16#benchmarks) |
| Nemotron 3.5 Lightning | SWE-bench Verified: **51.56%** | NVIDIA's BF16 evaluation: [NVIDIA card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16#evaluation-results) |

A SWE-bench score of 76% means the evaluated system resolved that fraction of
benchmark tasks under its evaluation conditions. It is not a 76% probability
that an arbitrary answer is correct. Super's different results with OpenCode
and OpenHands illustrate why the harness matters alongside the weights.

The published results suggest Gemma is a strong general coding candidate and
Muse Glimmer is worth testing on agent tasks. Super offers a larger MoE option
at higher memory and deployment cost; Lightning offers a smaller alternative.
These are selection hypotheses, not measured Pinnacles advantages. Compare the
actual quantized checkpoints on the same edit-and-test task before choosing.

Also measure **time to first token** (how long you wait for a response to begin),
**generation tokens per second** (how fast it continues), **peak GPU memory**,
and **time to a correct, tested patch**. A high benchmark score with very slow
tool turns may be a poor fit for an interactive workshop.

#### Locally measured pilot results

These are 15 pilot runs across five serving profiles, with one attempt per task:
10 LiveCodeBench problems, 10 Aider refactoring tasks, and 28 MMLU-Pro questions
(two per subject). They use direct API generation. Aider's local grader checks
Python AST structure and approximate node counts; it does not prove behavioral
equivalence or run the projects' full tests. These scores therefore do not
measure OpenCode's repository-repair workflow.

The saved manifests specify **32,768 context / 8,192 output** for LiveCodeBench
and Aider, and **16,384 context / 4,096 output** for MMLU-Pro. BF16 runs used
80 GB A100s in `dept.appliedmath`, which requires separate account access.
Quantized runs used public-queue A100 40 GB GPUs. The table's weight sizes are
checkpoint bytes in decimal GB, not loaded VRAM.

| Profile | Weight format and size | LiveCodeBench pass@1 | Aider AST pass rate | MMLU-Pro accuracy | Recorded median generation latency, LCB |
|---|---|---|---|---|---|
| `gemma-bf16-a100` | BF16, 62.5 GB | 90.0% (9/10) | 90.0% (9/10) | 82.1% (23/28) | 19.747 s |
| `gemma-qat-a100` | INT4 W4A16 QAT, 23.3 GB | 80.0% (8/10) | 90.0% (9/10) | 50.0% (14/28) | 8.613 s |
| `muse-bf16-a100` | BF16, 59.6 GB | 90.0% (9/10) | 80.0% (8/10) | 71.4% (20/28) | 51.599 s |
| `muse-dynamic-a100` | GGUF Dynamic Q4_K_XL, 19.7 GB | 80.0% (8/10) | 100.0% (10/10) | 67.9% (19/28) | 33.561 s |
| `lightning-nvfp4-a100` | NVFP4 mixed, 21.6 GB | 60.0% (6/10) | 60.0% (6/10) | 60.7% (17/28) | 4.160 s |

See the [aggregate report](benchmarks/results/aggregate/report.md) for failure
counts and confidence intervals, and [summary metadata](benchmarks/results/aggregate/summary.json)
for hardware, budgets, model/runtime revisions, and provenance hashes.
The [benchmark guide](benchmarks/README.md) explains how to run and interpret
the harness. Raw journals and generated code stay outside Git.

Read the results with these limits in mind:

- **Small samples:** one additional success changes a 10-task score by 10
  percentage points. A 10/10 result has a Wilson 95% interval of approximately
  72.2–100%; it does not establish perfect accuracy.
- **Quantization comparisons:** Gemma's Aider scores tie on this subset; its
  MMLU-Pro gap is 32.1 percentage points. Muse's MMLU-Pro gap is about 3.6
  percentage points (one question out of 28). GPU capacity, kernels, runtime,
  templates, and reasoning behavior differ. These observations do not isolate
  quantization as the cause or prove that a capability was preserved completely.
- **Output budgets:** reasoning and final code share the generation allowance.
  A response ending with `finish_reason=length` may be truncated. The harness
  groups output-limit stops under `timeout`, so inspect the finish reason to
  distinguish a token limit from a wall-clock timeout. More output headroom can
  help, but 8K does not guarantee every task finishes.
- **Latency:** these are generation timings for records that contain that
  measurement, excluding server startup and offline grading. Earlier
  generation-timeout records omitted timing: Lightning's LCB median covers
  six of ten trials, for example. Compare success rates, response lengths,
  and timing coverage together; this table is not a throughput ranking.
- **Evaluation scope:** SWE-bench Verified and Terminal-Bench have no measured
  results here. Their container/evaluator prerequisites remain unresolved.

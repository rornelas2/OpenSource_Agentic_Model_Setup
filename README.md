# Open-weight coding agents on UC Merced Pinnacles

A tutorial for students in Lawrence Livermore National Laboratory's Data Science
Challenge. Learn how to run an open-weight model on Pinnacles GPUs, serve it
through an API, and use **OpenCode** to read code, edit files, and run tests.

You need a regular sponsored Pinnacles account and basic familiarity with a
terminal. This tutorial uses the public `gpu` and `cenvalarc.gpu` partitions.

## What you will build

```text
You → OpenCode → model server (vLLM or llama.cpp) → GPU model
         ↑                 │
         └── tool request ─┘
         │
         └── read files, edit code, run tests → send results back to model
```

The **model** proposes actions. The **server** loads its weights onto GPUs and
provides an HTTP endpoint. **OpenCode** is the coding harness: it manages the
conversation, supplies tools, executes approved actions, and returns their
results to the model. **Slurm** allocates the compute node where the server runs;
the endpoint lasts only as long as that job.

OpenCode supports local servers through an OpenAI-compatible API. Here,
“OpenAI-compatible” describes the request format; inference runs on Pinnacles.
Files and shell commands are handled on the machine running OpenCode. A laptop
client therefore edits laptop files unless its tools are explicitly configured
otherwise. See the [OpenCode provider documentation](https://opencode.ai/docs/providers/).

**Start here:** follow the practical lessons below. The [reference section](#reference-gpus-memory-and-model-selection) explains GPU memory, storage, quantization, and benchmark results.

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
128 GB system RAM, and a 32,768-token context limit. You do not need a paid
model API account. Budget **90 GB free in data** and **30 GB free in scratch**
for the checkpoint, environment, and installation/build caches.

Validated on Pinnacles on **September 10, 2026**: model loading, chat, streaming,
a tool-call round trip, and an OpenCode file repair with all four exercise tests
passing. The 32,768-token limit is the configured ceiling; the exercise uses a
shorter conversation and is not a full-context stress test.

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
The complete Python dependency list is in
[`env/gemma-requirements.lock`](env/gemma-requirements.lock).
The installation uses prebuilt packages and requires no `sudo` or Docker.

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
Hugging Face requires authentication for your account, use its interactive
`hf auth login` command with `HF_HOME=/data/$USER/pinnacles-agents/huggingface`,
then rerun the download. Read the [model card](https://huggingface.co/google/gemma-4-31B-it)
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

The script sets BF16 precision, a 32,768-token context limit, one active request,
and an 85% GPU-memory budget. It enables Gemma's tool and reasoning parsers and
uses a matching chat template. This first exercise is text-only with thinking
disabled. The launch options follow the
[vLLM Gemma recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html).

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

For your own project, start OpenCode from that project's directory and copy the
same provider configuration there, merging it with any existing `opencode.json`.
The model ID and context limits in the configuration must match your server.
The [OpenCode documentation](https://opencode.ai/docs/providers/) explains custom
providers; [permissions](https://opencode.ai/docs/permissions/) control tool use.

## 8. Stop the server and release your GPU

After exiting OpenCode, inside the **compute shell**:

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
| Port 8000 already in use | Stop your earlier server, or choose another port in both the serving script and OpenCode's `baseURL`. |
| Out of GPU memory | Confirm an H200 was allocated and that another server is not already using your allocated GPU. |
| OpenCode returns text but makes no edits | Check approval prompts and that both Gemma parsers and the supplied chat template are enabled. |
| Request exceeds context limit | Begin a new session or reduce attached files/tool output; input and output share the context budget. |

The next model lessons will cover **Meta Muse Glimmer**, followed by **NVIDIA
Nemotron**. Their serving configurations remain untested here.


## Reference: GPUs, memory, and model selection

### GPU allocation

A **node** is a physical computer. A **partition** is a Slurm queue containing
nodes. You request a GPU type and count; Slurm assigns available hardware.

Public inventory checked against Slurm on **September 10, 2026**:

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
Gemma has passed the coding exercise above. Meta and NVIDIA remain candidates
awaiting their own Pinnacles serving lessons.

| Model | Role in the tutorial | Capacity consideration |
|---|---|---|
| [Google Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31B-it) | Tested Gemma lesson | BF16 serving and the OpenCode exercise passed on one H200. A single A100 or L40S needs quantization for GPU-resident weights. |
| [Meta Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | Agent-focused Meta target | BF16 is a candidate for H200; Meta also provides quantized GGUF releases for smaller GPUs. |
| [NVIDIA Nemotron 3 Super 120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | Larger Nemotron candidate | Quantization reduces storage substantially; the exact quantized runtime must support Pinnacles hardware. |

Gemma 4 31B has 30.7B parameters. The BF16 H200 run reported **57.91 GiB** for
model loading. Device memory use after the agent exercise was **120,281 MiB**
(about 117.5 GiB), including the server's reserved cache and runtime state.
This allocation is not the minimum memory required to serve the model. vLLM
reserves cache according to its configured memory budget.

If this configuration proves impractical for your allocation,
the 26B-A4B model is the next smaller family member to consider. Reducing
precision or using an H200 may preserve the 31B target without downsizing.
[Google model card](https://ai.google.dev/gemma/docs/core/model_card_4)

Nemotron 3.5 Lightning 30B-A3B is a smaller alternative if Super cannot be served
well. Its [official card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
describes a hybrid MoE model and deployments on Ampere and Hopper GPUs. The
larger Super quantized card documents Blackwell examples; copying a B200 recipe
does not establish H200 compatibility.

Nemotron 3 Ultra 550B-A55B is a further capacity candidate for a future
multi-node exercise. Four-bit weight arithmetic alone is about 275 GB before
metadata and higher-precision tensors. Its [official deployment guidance](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/raw/main/README.md)
uses larger configurations, including four Blackwell GPUs or eight H100s.
Neither that recipe nor aggregate memory arithmetic establishes a working
four-H200 Pinnacles deployment. **The largest usable Nemotron is still to be
determined.**

#### What do the benchmark scores mean?

Benchmarks measure different abilities. For a coding agent, repository repair
and terminal tasks are more directly relevant than answering isolated questions.

| Benchmark | What is tested | What the score does not establish |
|---|---|---|
| [SWE-bench Verified](https://www.swebench.com/) | Resolving real repository issues, evaluated through tests | That every generated patch is maintainable or that your project will work |
| [Terminal-Bench](https://www.tbench.ai/benchmarks) | Completing multi-step tasks through a terminal | Equivalent results across different benchmark versions or harnesses |
| [LiveCodeBench](https://livecodebench.github.io/) | Coding problems evaluated against tests | Reliable repository navigation or tool use over a long session |
| [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Knowledge and reasoning across academic subjects | Practical software engineering ability |

Selected **publisher-reported** results, checked September 10, 2026. Scores are
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

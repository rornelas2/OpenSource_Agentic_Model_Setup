# Open-Weight Agentic Models on UC Merced Pinnacles

## Implementation plan

- **Status:** Draft for review
- **Prepared:** 2026-09-10
- **Target audience:** Lawrence Livermore National Laboratory Data Science Challenge students using UC Merced's Pinnacles cluster
- **Primary outcome:** A beginner-safe, reproducible tutorial and tested scripts for launching open-weight instruction/agentic models through Slurm without relying on private partitions such as `dept.appliedmath`

---

## 1. Purpose

This repository will teach a student how to go from a new Pinnacles account to a working, locally hosted model endpoint. The finished tutorial must explain each command, use only resources available to the intended student accounts, and distinguish configurations that were actually tested from configurations that merely appear to fit on paper.

The first supported model families will be:

1. Google Gemma 4, beginning with a small onboarding model and scaling through the 31B dense model.
2. NVIDIA Nemotron, beginning with Nemotron 3.5 Lightning and testing larger Nemotron 3 variants where the runtime and memory footprint permit.
3. Meta Llama, using the current downloadable Llama 4 Scout model as the practical target and treating Llama 4 Maverick as an experimental multi-node candidate.
4. A small, curated expansion set of other popular open-weight models, initially Mistral Small 4. Larger Mistral and Qwen candidates will be considered only after the core paths are reliable.

“Host” means run an inference server inside a Slurm allocation and connect to it from an authorized client. It does **not** mean a permanent public internet service. Pinnacles jobs have wall-time limits, shared queues, and cluster networking rules, so the tutorial will teach an ephemeral, authenticated workflow.

---

## 2. Success criteria

The project is complete for its first release when a student can:

1. Verify which Pinnacles account, queues, GPU types, and limits they actually have.
2. Create a reproducible user-space software environment without `sudo` or Docker.
3. Store model data outside the small home-directory quota and inspect available space before downloading.
4. Obtain any required Hugging Face access without placing a token in Git, a Slurm script, shell history, or a job log.
5. Launch at least one small model on one public GPU using a batch script.
6. Call an OpenAI-compatible `/v1/chat/completions` endpoint from inside the allocation and, if cluster policy permits it, through a documented SSH tunnel.
7. Launch tested Gemma, Nemotron, and Llama configurations by changing a model profile rather than rewriting the Slurm script.
8. Understand why a model fits or does not fit, including weights, quantization, KV cache, runtime overhead, and requested context length.
9. Reproduce recorded smoke-test and benchmark results on A100, L40S, and H200 hardware that their account can access.
10. Diagnose the most common failures from Slurm state, logs, CUDA/runtime errors, out-of-memory errors, access-gated model errors, and disk quota errors.

Release acceptance additionally requires:

- No core example references `dept.appliedmath`, a PI partition, the MERCED recharge cluster, or a developer-specific account/path.
- Every core command has been run by a clean test account or a Challenge student account, not only by a maintainer's established environment.
- Every supported model row carries a status of `verified`, `partially verified`, `experimental`, or `not supported` and records the date, checkpoint revision, GPU, GPU count, context length, runtime version, and result.
- The documented default requests the smallest resource known to work.
- No example binds a model server to a public interface by default.
- No secret, model weight, cache, generated log, or environment directory is tracked by Git.

---

## 3. Non-goals for the first release

The first release will not:

- train a foundation model from scratch;
- promise uninterrupted or production-SLA hosting on a batch-scheduled research cluster;
- expose an unauthenticated endpoint to the public internet;
- bypass Slurm, account associations, queue limits, or CIRT policy;
- depend on private partitions, purchased condo nodes, or MERCED billing;
- download every theoretically compatible checkpoint;
- claim the advertised maximum context window is usable at useful concurrency;
- treat a successful model load as proof of useful inference speed or output correctness;
- redistribute gated model weights or accept a model license on a student's behalf; or
- execute model-generated tool calls without a separate, explicit sandbox and approval design.

Fine-tuning, RAG, multi-agent orchestration, and web UIs are follow-on modules after base serving is stable.

---

## 4. Evidence collected so far

### 4.1 Published UC Merced documentation

The current campus documentation describes Pinnacles as Rocky Linux 9.8 with Slurm. It lists the following GPU resources:

| Public partition | Nodes | GPUs per node | VRAM per GPU | Aggregate VRAM per node | Published maximum |
|---|---:|---:|---:|---:|---|
| `gpu` | 8 | 2 x NVIDIA A100 PCIe | 40 GB | 80 GB | 2 nodes, 3 days |
| `cenvalarc.gpu` | 8 | 2 x NVIDIA L40S | 48 GB | 96 GB | 2 nodes, 3 days |
| `cenvalarc.gpu` | 4 | 2 x NVIDIA H200 NVL | 141 GB | 282 GB | 2 nodes, 3 days |
| `test` | shared access across node types | varies | varies | varies | 2 nodes, 1 hour, one submitted job |

Published default storage allocations are 70 GB soft/75 GB hard for `HOME`, 500/512 GB for `data`, and 500/512 GB for `scratch`. Scratch is purgeable and is not a backup. These limits matter because a single frontier-scale quantized checkpoint can consume most of one storage allocation.

The official documentation also says:

- compute-intensive work must run as a Slurm job, never on a login node;
- GPU jobs must explicitly request a GRES;
- Docker is not supported; SingularityCE is the supported container mechanism;
- Conda environments are supported in user space; and
- GPU resources are heavily used, so jobs may wait and underutilized jobs may be terminated.

Sources:

- [UC Merced HPC campus clusters](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/campus-clusters/)
- [UC Merced cluster policies](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/policies/)
- [UC Merced GPU best practices](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/hpc-tutorials/gpu_best_practices/)
- [UC Merced Slurm guide](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/hpc-tutorials/slurm/)
- [UC Merced Conda guide](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/running-jobs/conda/)
- [UC Merced container guide](https://ucm-it.github.io/hpc_docs/docs/hpcdocs/HPC-clusters/running-jobs/singularity/)

### 4.2 Live, read-only Slurm probe

On 2026-09-10, a read-only probe from `rclogin02` confirmed:

- `gpu`: 8 two-GPU A100 nodes, 3-day wall time, maximum 2 nodes;
- `cenvalarc.gpu`: 8 two-GPU L40S nodes and 4 two-GPU H200 NVL nodes, 3-day wall time, maximum 2 nodes;
- `test`: visibility of A100, L40S, H200 NVL and other condo GPUs, with a 1-hour limit;
- current modules include Anaconda, CUDA-related software, and `singularityce/4.1.0`; and
- the login nodes do not contain a GPU, as expected.

The live partition configuration also revealed a critical qualification not made prominent in the public table: `gpu` and `cenvalarc.gpu` currently contain `DenyAccounts=project_ucm_guests`, while `test` has `AllowAccounts=ALL`. The maintainer account used for this probe is PI-associated; it is **not** evidence that a Challenge student or guest account can submit to the same queues.

Therefore the final tutorial must not call A100, L40S, or H200 access “available to all students” until Phase 0 is completed with the actual Challenge account type. If Challenge students use `project_ucm_guests`, the one-hour `test` path may be their only default path unless CIRT creates or authorizes another project association.

### 4.3 Known documentation drift

The UC Merced web documentation and live Slurm state do not perfectly match in every detail. For example, the page says “6 public queues” while its table lists additional CENVAL-ARC queues, and older support text still refers to only eight public GPU nodes. The repository will treat live scheduler output as the operational source of truth for a run and the official web documentation as the policy/reference source. Both must be date-stamped.

---

## 5. Access profiles to support

The tutorial will choose one of two profiles after running the access probe.

### Profile A: PI-associated student with public GPU access

Expected queues:

- `gpu` for A100;
- `cenvalarc.gpu` for L40S or H200; and
- `test` for short hardware and compatibility checks.

This profile receives the full tutorial, including longer batch serving and the large-model experiments.

### Profile B: guest/test-only student

Expected queue:

- `test`, subject to one-hour wall time and the student's QoS.

This profile receives a bounded workshop path:

- use a pre-staged, small model;
- request one GPU for 15–45 minutes;
- run a local smoke test rather than an unattended service;
- keep startup time low by avoiding a model download during the allocation; and
- do not advertise large-model or multi-node examples as runnable.

If the Challenge requires longer hosting for Profile B, the resolution is administrative: the Challenge organizers and CIRT must agree on the project association, reservation, or access policy. The code must not attempt a technical workaround.

---

## 6. Candidate model matrix

Checkpoint size is only a lower bound on required VRAM. A server also needs memory for CUDA kernels, activations, quantization metadata, the KV cache, multimodal encoders, and runtime workspaces. MoE models retain all experts' weights in memory even though only a subset is active for each token.

The initial planning matrix is:

| Priority | Model/checkpoint | Published checkpoint facts | Likely Pinnacles target | Initial status |
|---:|---|---|---|---|
| 0 | Gemma 4 E2B IT | Google's starter example downloads about 10.2 GB; 128K advertised context | 1 x A100, L40S, or H200; use a deliberately small context | Core onboarding candidate |
| 1 | Gemma 4 26B A4B IT | 26B total MoE; roughly 50.5 GB BF16 or 14.6 GB Q4 GGUF; 256K advertised context | quantized on 1 x A100/L40S; BF16 on 1 x H200 or 2 smaller GPUs | Core candidate |
| 1 | Gemma 4 31B IT | 31B dense; official BF16 repository about 62.6 GB; 256K advertised context | 1 x H200 or 2 x A100/L40S; quantized single-GPU path if quality is acceptable | Core large Gemma candidate |
| 1 | NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 | 30B total/3B active; repository about 21.6 GB; official model card lists Ampere W4A16 and Hopper support | 1 x A100, L40S, or H200 | Core Nemotron candidate |
| 1 | Mistral Small 4 119B NVFP4 | 119B total/6.5B active; repository about 70.8 GB | 1 x H200 or 2 x A100/L40S, subject to runtime validation | Expansion candidate |
| 2 | NVIDIA Nemotron 3 Super 120B-A12B NVFP4 | 120B total/12B active; repository about 80.4 GB; vendor minimum names B200 or DGX Spark | 1 x H200 has capacity on paper; 2-GPU fallback may be needed | Experimental until Hopper runtime is proven |
| 2 | Llama 4 Scout 17B-16E Instruct | 109B total/17B active; BF16; Meta says on-the-fly INT4 can fit one H100 | 1 x H200 INT4 or 2 x H200 BF16; possibly 2 x A100/L40S INT4 | Core Llama target, gated and unverified here |
| 3 | Llama 4 Maverick 17B-128E Instruct FP8 | about 400B total/17B active; FP8 weight shards are roughly 430+ GB | 4 x H200 across 2 nodes has aggregate capacity, but topology and runtime are unverified | Stretch experiment, not a promised tutorial path |
| 3 | NVIDIA Nemotron 3 Ultra 550B-A55B NVFP4 | repository about 352 GB; official minimum is 4 x B200, and model card examples target newer interconnects | aggregate capacity exists on 4 x H200 across 2 nodes, but this is below vendor recommendation | Not supported unless an explicit experiment succeeds |
| 3 | Mistral Large 3 675B NVFP4 | 675B total/41B active; repository about 403 GB; official launch recipe uses tensor parallel size 8 | aggregate capacity exists on 4 x H200, but the recommended GPU count and single-node topology are unavailable | Not supported unless a 4-GPU recipe is validated |
| 3 | Qwen 3.5 397B-A17B official weights | official repository about 807 GB | does not fit in 4 x H200 as published; third-party quantizations require a separate provenance review | Backlog only |

Relevant primary model sources:

- [Gemma 4 overview](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 Hugging Face inference guide](https://ai.google.dev/gemma/docs/core/huggingface_inference)
- [Meta Llama 4 announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Llama 4 Scout official model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [NVIDIA Nemotron model family](https://developer.nvidia.com/topics/ai/nemotron)
- [Nemotron 3.5 Lightning official model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- [Nemotron 3 Super official model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
- [Nemotron 3 Ultra official model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
- [Mistral Small 4 official model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
- [Mistral Large 3 NVFP4 official model card](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4)
- [Qwen 3.5 397B official model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

### Model selection rule

A model becomes a core tutorial model only if all of the following are true:

1. The checkpoint is published by the model developer or an explicitly approved quantizer.
2. Its license and access workflow can be explained to students.
3. The checkpoint plus one environment fits within normal student storage quotas with safe headroom.
4. It loads without CPU offload on an allowed allocation.
5. It retains at least 10% free VRAM after a representative request at the documented context setting.
6. It returns valid chat output and survives repeated and concurrent requests.
7. Startup, throughput, queue cost, and failure modes are measured.
8. The exact setup is reproducible from a clean account.

The largest model that merely satisfies aggregate weight arithmetic will be called “theoretical.” The largest model that passes the above gate will be called “largest verified.”

---

## 7. Repository design

The intended repository layout is:

```text
.
├── README.md                         # Student tutorial and supported-model table
├── implementation.md                 # This project plan and decision record
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── .gitignore
├── env/
│   ├── environment.yml              # Pinned common client/runtime dependencies
│   └── requirements-lock.txt         # Exact resolved versions used in validation
├── config/
│   └── models.yaml                  # Data-only model profiles
├── scripts/
│   ├── probe/
│   │   ├── account.sh               # Association/partition/QoS report
│   │   ├── gpu.sbatch               # Short nvidia-smi/CUDA/network probe
│   │   └── storage.sh               # Quota and cache-path checks
│   ├── setup/
│   │   ├── create_env.sh            # Idempotent environment creation
│   │   └── download_model.py         # Revision-pinned, resumable download
│   ├── serve/
│   │   ├── serve.sbatch             # Generic one-node server
│   │   ├── serve_multinode.sbatch   # Experimental, isolated from core path
│   │   └── launch.py                # Validates a profile and builds argv safely
│   ├── client/
│   │   ├── smoke_test.py
│   │   └── benchmark.py
│   └── lib/
│       └── common.sh
├── tests/
│   ├── test_model_config.py
│   ├── test_launch_contract.py
│   ├── test_no_private_partitions.py
│   ├── test_no_secrets.py
│   └── fixtures/
├── results/
│   ├── README.md                     # Result schema and reproduction rules
│   └── YYYY-MM-DD/<model>/<gpu>.json # Small, sanitized validation records
└── docs/
    ├── troubleshooting.md
    ├── model-access.md
    ├── networking.md
    ├── sizing.md
    └── maintainers.md
```

Generated environments, downloaded weights, raw logs, tokens, and Slurm output must be ignored. Only sanitized, small JSON result summaries belong under `results/`.

---

## 8. Configuration and command contracts

### 8.1 Model profile contract

`config/models.yaml` will be the source of truth for supported launch profiles. Each profile must contain:

```yaml
id: gemma-4-e2b-it
model_id: google/gemma-4-E2B-it
revision: <immutable commit SHA>
runtime: transformers
task: any-to-any
gated: true
license_url: https://ai.google.dev/gemma/terms
partition: cenvalarc.gpu
gres: gpu:l40s:1
nodes: 1
gpus_per_node: 1
cpus_per_task: 8
memory_gb: 64
wall_time: "00:30:00"
max_model_len: 8192
max_concurrency: 1
gpu_memory_utilization: 0.85
extra_args: []
verification_status: unverified
```

Validation rules:

- `revision` must be an immutable model repository commit, not `main`.
- `partition` must be in an allowlist: `test`, `gpu`, or `cenvalarc.gpu`.
- `nodes` must be 1 for core profiles and at most 2 for experiments.
- `gpus_per_node` must match the live GRES shape and be 1 or 2 for public nodes.
- wall time must not exceed the selected partition's live maximum.
- `gpu_memory_utilization` must be in `(0, 0.90]` for examples.
- `max_model_len` must be explicit; no profile inherits a model's advertised maximum.
- `extra_args` must be an argv list. Shell fragments and command substitution are rejected.
- a gated model must link to its terms and document the manual access step.
- unknown keys and unknown model IDs fail closed with an actionable error.

### 8.2 Launcher contract

Input: one validated model profile plus optional safe overrides for wall time, port, and output directory.

Output: a deterministic argument vector and a Slurm submission command. The launcher must print the selected partition, GRES, checkpoint revision, storage path, context limit, and log path before submission.

Failures:

- invalid configuration: exit 2, no job submitted;
- inaccessible model or absent token: exit 3, no retry loop;
- insufficient storage: exit 4 before download;
- unsupported GPU/runtime combination: exit 5 before server launch;
- server readiness timeout: exit 6 after cleaning up child processes;
- smoke-test failure: exit 7 and preserve sanitized diagnostics.

The launcher must not evaluate model-config text as shell code. It must propagate termination to the inference server and wait for it to exit so a canceled job does not leave child processes behind.

### 8.3 Result record contract

Each validation record will include:

- date and maintainer;
- Git commit and model checkpoint revision;
- account profile (`student-public`, `guest-test`, or `maintainer`; never the account name);
- partition, node count, GPU model/count, driver and CUDA versions;
- runtime and dependency versions;
- precision/quantization, context length, concurrency, prompt/input modality;
- checkpoint bytes on disk;
- startup time, peak GPU memory, prompt tokens/second, output tokens/second;
- smoke-test result and benchmark result;
- Slurm terminal state and elapsed time; and
- sanitized error category when unsuccessful.

Raw prompts containing student or Challenge data will not be committed.

---

## 9. Implementation phases

### Phase 0 — Confirm student access and policy

**Goal:** Establish the exact resource boundary before writing runnable claims.

Tasks:

1. Run the read-only probe from one actual Challenge student account:

   ```bash
   hostname
   id
   sacctmgr -n -P show assoc user="$USER" \
     format=Cluster,Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit
   sinfo -a -o '%P|%a|%l|%D|%G|%f'
   scontrol show partition test
   scontrol show partition gpu
   scontrol show partition cenvalarc.gpu
   quota -vs
   ```

2. Submit a two-minute, non-intensive probe to each apparently permitted GPU type. The job will run `hostname`, `nvidia-smi --query-gpu=...`, inspect `CUDA_VISIBLE_DEVICES`, report filesystem mounts, and test only the network endpoints needed for package/model downloads.
3. Record whether Challenge students belong to `project_ucm_guests` or a PI/project association.
4. Ask CIRT to confirm, in writing if possible:
   - intended Challenge account/project association;
   - access to `gpu` and `cenvalarc.gpu`;
   - whether model downloads are permitted from login and/or compute nodes;
   - whether loopback model serving and SSH port forwarding are permitted;
   - whether a shared read-only model cache or reservation will be provided; and
   - whether Challenge data has handling constraints beyond the published P3/P4 restriction.
5. Save only sanitized capability results. Do not publish account names, node-specific sensitive configuration, or tokens.

**Exit criteria:** The access profile, permitted queues, GRES names, wall time, GPU inventory, download route, storage plan, and networking method are confirmed for the students.

**Blocking rule:** If actual student access cannot be tested, the README must label all student-access statements unverified and provide only the `test`-partition minimum path.

### Phase 1 — Repository safety and documentation skeleton

**Goal:** Establish a safe base before downloading or executing models.

Tasks:

1. Add `.gitignore` entries for:
   - `.env`, tokens, credential files, and Hugging Face cache metadata;
   - `slurm-*.out`, raw logs, PIDs, sockets, and benchmark scratch files;
   - `.conda`, virtual environments, Python caches, and build products;
   - `*.safetensors`, `*.gguf`, `*.bin`, `*.pt`, `*.pth`, and `*.sif`; and
   - downloaded datasets and generated outputs.
2. Add a secret scan and a test rejecting private partition names from runnable scripts.
3. Create the README headings in the order a first-time student will use them:
   - prerequisites and account access;
   - five-minute conceptual overview;
   - login/VPN;
   - inspect access and storage;
   - install the environment;
   - obtain model access;
   - pre-download a model;
   - submit a small server;
   - call and stop the server;
   - select another model/GPU;
   - benchmark and troubleshoot;
   - cleanup and data handling.
4. Add contribution and security guidance.
5. Establish Markdown linting and link checking in CI. External links may be retried, but broken primary references fail a scheduled documentation check.

**Exit criteria:** CI can validate the empty/skeleton project, large weights cannot be accidentally staged, and the README contains no unverified resource claim stated as fact.

### Phase 2 — Capability probe and reproducible environment

**Goal:** Make setup repeatable from a clean account without administrator access.

Tasks:

1. Implement `scripts/probe/account.sh` using read-only Slurm commands. Normalize output into a small JSON report without usernames.
2. Implement `scripts/probe/storage.sh` to resolve symlinks, print quota/free space, verify that cache paths are under `data` or `scratch`, and fail before a large download if headroom is insufficient.
3. Choose a Python version supported by the tested PyTorch/runtime stack. Pin exact package versions after installation succeeds on all target GPU architectures.
4. Compare two environment paths:
   - **Core:** Conda environment stored under `/data/$USER/...`, following the cluster guide.
   - **Maintainer/advanced:** a remotely built SingularityCE image executed with NVIDIA GPU passthrough, only if CIRT policy and the installed Singularity version support the required runtime.
5. Do not make Docker commands runnable in the Pinnacles tutorial. Vendor Docker examples must be translated and tested or clearly marked “not for Pinnacles.”
6. Make setup idempotent: a second run verifies or updates the pinned environment instead of corrupting it.
7. Produce `python -m pip freeze`, PyTorch/CUDA visibility, and runtime version diagnostics from a GPU allocation.

**Exit criteria:** A clean account can create the environment twice, import the runtime on each allowed GPU type, and run a tiny CUDA operation inside Slurm.

### Phase 3 — Secure, revision-pinned model acquisition

**Goal:** Download models once, safely and reproducibly, before consuming scarce GPU time.

Tasks:

1. Document Hugging Face account creation and the manual license/access workflow for gated Gemma and Llama repositories.
2. Use `huggingface_hub`/`hf download` with:
   - an immutable revision;
   - a local directory under the selected data/cache path;
   - resume support;
   - no symlink assumptions across filesystems; and
   - a restricted include list when the repository contains multiple formats.
3. Read the token from an interactive login or a permission-restricted credential store. Never pass it on the Slurm command line or export it in a committed file.
4. Query repository metadata and calculate expected bytes before download. Require projected post-download free space of at least `checkpoint size + environment reserve + 10% filesystem safety margin`.
5. Verify the resolved checkpoint revision and file count after download. Record checksums supplied by the host; do not invent a second manifest that silently tracks mutable `main`.
6. Design for a Challenge-wide shared cache if CIRT approves it. It should be staged once by an organizer, read-only to students, and keyed by model plus immutable revision. Otherwise each student follows the personal quota path.
7. Never download large weights during a short GPU allocation unless the exercise specifically measures cold start.

**Exit criteria:** One small public/gated model can be downloaded, resumed, revision-verified, and loaded by a Slurm job without leaking credentials.

### Phase 4 — Small-model vertical slice

**Goal:** Prove the complete workflow with Gemma 4 E2B before adding complexity.

Tasks:

1. Start with direct Transformers inference if Gemma 4 support in the selected serving runtime is not yet stable.
2. Add an OpenAI-compatible server using a tested vLLM release or another maintained backend only after its Gemma 4 recipe passes.
3. Use one GPU, an 8K or smaller initial context, one concurrent request, and a 30-minute allocation.
4. Bind to `127.0.0.1` by default and write a readiness file containing node, port, model ID, revision, and job ID but no token.
5. Poll a health endpoint with a bounded timeout; do not use a fixed long sleep.
6. Send deterministic smoke prompts covering:
   - plain text chat;
   - system instruction adherence;
   - a bounded generation length;
   - malformed request rejection; and
   - one image input if the selected backend supports the modality.
7. Capture startup time, peak memory, throughput, server exit, and Slurm state.
8. Test `scancel` and wall-time termination to verify child-process cleanup.

**Exit criteria:** A clean-account student follows the draft README verbatim and receives a valid response with no login-node computation.

### Phase 5 — Generic server and model profiles

**Goal:** Separate Slurm/resource policy from model-specific runtime flags.

Tasks:

1. Implement the validated YAML profile schema.
2. Implement a launcher that constructs an argument vector without `eval`.
3. Add one-node batch templates for:
   - A100 (`gpu`, `gpu:a100:N`);
   - L40S (`cenvalarc.gpu`, `gpu:l40s:N`); and
   - H200 (`cenvalarc.gpu`, `gpu:nvidia_h200_nvl:N`).
4. Confirm live GRES spellings during each run; fail with a useful message if the cluster changes them.
5. Set cache, temporary, compilation, and model output directories to explicit per-user locations. Never rely on login-node `/tmp`.
6. Add traps for `TERM`, `INT`, and normal exit. Stop the server, wait for children, and preserve the final sanitized metrics.
7. Add a readiness timeout and bounded retry behavior only for idempotent health checks.
8. Keep the default API loopback-only. If SSH forwarding is approved, document a tunnel through `login.rc.ucmerced.edu`; otherwise keep the client in the same allocation or use approved Open OnDemand access.
9. Require an API token if a server is bound beyond loopback, even on the cluster network.

**Exit criteria:** Adding a tested model requires a data profile and result record, not a copied Slurm script.

### Phase 6 — Core family validation

**Goal:** Establish tested, escalating examples across Google, NVIDIA, and Meta.

Validation order:

1. Gemma 4 E2B on one L40S or A100.
2. Gemma 4 26B A4B in a vendor or trusted quantized format on one GPU.
3. Gemma 4 31B BF16 on one H200, then two A100/L40S if tensor parallelism is supported.
4. Nemotron 3.5 Lightning NVFP4 on A100, L40S, and H200, using the vendor's documented W4A16 fallback where needed.
5. Nemotron 3 Super NVFP4 on H200 only after the runtime supports its hybrid Mamba/MoE architecture on Hopper.
6. Llama 4 Scout after the maintainer and student separately accept the Llama license. Test a memory-safe INT4 path first; test BF16 on two H200s only if the checkpoint, host RAM, and runtime leave safe headroom.
7. Mistral Small 4 NVFP4 as the first additional family.

For every configuration:

- begin on `test` with a 10–30 minute request;
- reduce context/concurrency before adding GPUs;
- test one GPU before two when memory permits;
- run at least three warm requests and one repeated/concurrent set;
- record OOM and unsupported-kernel failures as results rather than hiding them;
- compare output format against the model's official chat template; and
- promote to a 3-day public GPU queue only after startup and utilization are understood.

**Exit criteria:** At least one model in each core family is verified, and every other listed configuration is accurately labeled.

### Phase 7 — Largest-model experiments

**Goal:** Identify the largest **verified** model without turning theoretical aggregate VRAM into a promise.

Tasks:

1. Create a separate experimental multi-node launcher using `torchrun`, Ray, or the serving runtime's documented Slurm method. Select only one mechanism after a minimal two-node collective test succeeds.
2. Verify interface selection and NCCL across HDR-100 InfiniBand without hard-coding a private node name.
3. Confirm whether tensor parallel sizes required by each model are compatible with four GPUs and two GPUs per node.
4. Stage model weights before allocation and verify that 256 GB host RAM per node is enough for load/conversion behavior.
5. Test in this order:
   - Llama 4 Scout BF16 on two H200s, if not already verified;
   - Llama 4 Maverick FP8 on four H200s;
   - Nemotron 3 Ultra NVFP4 on four H200s; and
   - Mistral Large 3 NVFP4 on four H200s.
6. Use a small context and concurrency of one for the first load. Increase context only after peak memory is known.
7. Stop an experiment if:
   - the vendor requires more GPUs than Pinnacles exposes per job;
   - runtime support requires an unreviewed patch;
   - weight storage would leave less than the required safety margin;
   - NCCL is unstable or throughput is unusably low across nodes; or
   - the job materially underutilizes scarce H200 resources.
8. Publish negative results and the exact reason. Do not recommend CPU offload as a core “fit” because it can transform a memory success into an unusably slow service.

**Exit criteria:** The README names a date-stamped largest verified configuration and clearly separates it from larger theoretical/unsupported candidates.

### Phase 8 — Student usability, evaluation, and release

**Goal:** Make the tutorial teachable and maintainable.

Tasks:

1. Run a clean-room exercise with at least one student who did not develop the scripts.
2. Time each section: login, probe, setup, download/staging, queue wait, startup, request, and cleanup.
3. Add expected-output snippets that avoid unstable details such as exact node names.
4. Add troubleshooting decision trees for:
   - `Invalid account or account/partition combination`;
   - `QOSMaxSubmitJobPerUserLimit` and long pending jobs;
   - missing/incorrect GRES;
   - gated-repository 401/403;
   - disk quota exceeded;
   - CUDA not visible or driver/runtime mismatch;
   - unsupported model architecture/kernel;
   - GPU OOM at load versus OOM during generation;
   - server readiness timeout;
   - failed SSH forwarding; and
   - jobs ending in `TIMEOUT`, `OUT_OF_MEMORY`, `FAILED`, or `CANCELLED`.
5. Include a “stop and clean up” section near the first runnable example, not only at the end.
6. Have CIRT or a knowledgeable UC Merced reviewer check resource and networking claims.
7. Tag `v0.1.0` only after all core acceptance criteria pass.

**Exit criteria:** A new student completes the core path without undocumented help, and maintainers can reproduce each verified model record.

---

## 10. Slurm serving design

The generic one-node job will follow this lifecycle:

```text
validate profile before submission
        |
        v
Slurm allocates an allowed GPU node
        |
        v
job records hardware/runtime metadata
        |
        v
server binds to loopback and starts as a child process
        |
        v
bounded readiness checks ---- failure ----> stop child, classify error, exit nonzero
        |
      ready
        |
        v
smoke test / authorized client requests
        |
        v
TERM, cancellation, wall time, or normal completion
        |
        v
stop child -> wait -> sanitize metrics -> exit
```

Representative directives will be generated from the profile, not copied blindly:

```bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
```

The implementation must verify whether Slurm accepts both `--gres` and any additional `--gpus-per-node` setting before documenting both. One unambiguous GPU request is preferable.

The server port must be user-configurable, in an approved high-port range, and checked for availability within the allocation. Readiness metadata will be written atomically. Only the allocating user may read any generated API credential.

---

## 11. Memory and capacity methodology

### 11.1 Pre-download estimate

For a first estimate:

```text
weight_bytes ≈ total_parameters × bytes_per_stored_weight + quantization metadata
required_vram ≈ loaded weights + KV cache + activations + runtime workspaces + safety margin
```

Approximate bytes per weight are 2 for BF16/FP16, 1 for FP8/INT8, and about 0.5 plus metadata for four-bit formats. Repository size is often a better starting measurement than parameter arithmetic, but neither proves runtime residency.

### 11.2 Runtime gate

No profile is considered to fit unless it passes a real load and generation test. The initial test will reserve no more than 85% of VRAM for the runtime/model where the backend provides such a control, use a small explicit context, and measure free/peak memory. Promotion requires at least 10% observed headroom after a representative request.

### 11.3 Context gate

Advertised context windows such as 256K, 1M, or 10M are model capabilities, not default Pinnacles settings. Core examples start at 8K. Context tiers are tested independently (8K, 32K, then larger if useful), and concurrency is recorded because KV cache cost scales with stored tokens and active requests.

### 11.4 Topology gate

Memory summed across devices is not automatically usable. A two-GPU same-node configuration and a four-GPU two-node configuration have different communication paths. A model that needs an eight-way tensor-parallel launch does not fit the public job shape merely because four H200s have enough aggregate bytes.

---

## 12. Verification strategy

### 12.1 Static/local checks

- YAML schema and invariant tests.
- Shell formatting/linting and Python formatting/type checks.
- Markdown lint and internal/external link checks.
- Reject private partitions and developer-specific absolute paths.
- Reject shell `eval`, embedded bearer tokens, and credential-like strings.
- Reject model weight extensions from Git.
- Validate every README model table row against `models.yaml` and the latest result record.

### 12.2 Cluster integration checks

- Slurm dry-run or minimal job submission under each access profile.
- CUDA visibility on each allowed GPU type.
- Environment import and tiny tensor calculation.
- Model download metadata and revision resolution.
- Server start, readiness, request, malformed request, and clean shutdown.
- Cancellation and wall-time behavior.
- Same-node two-GPU communication test before model sharding.
- Two-node NCCL test before any frontier model.

### 12.3 Model quality/safety smoke checks

This is not a leaderboard project, but a server that emits malformed chat, exposes hidden reasoning unexpectedly, or ignores its chat template is not working correctly. Each model receives a fixed, public test set for:

- basic factual response format;
- deterministic structured JSON where supported;
- system-role behavior;
- tool-call serialization without executing the call;
- multilingual output for advertised languages;
- multimodal input where supported;
- refusal/safety behavior using benign boundary tests; and
- absence of prior-request leakage across independent conversations.

Outputs are sampled and reviewed; benchmark scores are not treated as guarantees for Challenge tasks.

### 12.4 Performance checks

Measure cold startup separately from warm inference. At minimum record:

- time from job start to server readiness;
- first-token latency;
- prompt processing throughput;
- output throughput;
- peak VRAM;
- GPU utilization during a sustained request; and
- total Slurm elapsed time.

Performance comparisons use the same prompt token count, output limit, context setting, concurrency, runtime revision, and precision.

---

## 13. Security, privacy, licensing, and responsible use

1. **Cluster policy:** Students remain responsible for every command. The README will explicitly warn against running inference, compilation, or large downloads that consume heavy CPU on login nodes.
2. **Data classification:** Published CIRT policy says campus clusters do not support P3/P4 sensitive data. The tutorial will use only public synthetic prompts and will tell students to obtain Challenge-specific data handling guidance.
3. **Secrets:** GitHub tokens, Hugging Face tokens, API keys, and generated server credentials must never be committed or written to Slurm logs. Error reporting must redact authorization headers and query strings.
4. **Network exposure:** Loopback is the default. Any tunnel or non-loopback binding must be documented only after CIRT confirmation. Public exposure is out of scope.
5. **Model licenses:** Gemma and Llama may require manual acceptance; NVIDIA models use their stated model licenses; Mistral candidates listed here use Apache 2.0. The repository license covers project code and prose, not third-party weights.
6. **Supply chain:** Prefer developer-owned repositories and immutable revisions. A third-party quantization requires maintainer review of provenance, format, license, and checksum before becoming a core profile.
7. **Remote code:** `trust_remote_code=True` is not a harmless convenience. If a model requires it, pin the revision, inspect the code, document the decision, and isolate it from the default environment where practical.
8. **Generated actions:** Tool-call examples serialize proposals only. The tutorial will not let a model execute arbitrary shell commands or access student credentials.

---

## 14. Documentation standards

The README must be written for a student who knows basic Python but may not know Slurm, GPU memory, SSH tunneling, or model licensing.

Every command block will state:

- where it runs: local laptop, login node, interactive compute node, or batch script;
- what it changes;
- expected output or success signal;
- approximate storage/resource use;
- how to stop or undo it; and
- the next troubleshooting link if it fails.

Placeholders will use consistent angle-bracket notation such as `<UCM_USERNAME>`. Commands must never include a real username, personal email, home path, token, or node assignment.

Model documentation will distinguish:

- open source software from open-weight models;
- base from instruction-tuned checkpoints;
- total from active MoE parameters;
- stored precision from compute precision;
- official from third-party quantization;
- model context capability from the tested context; and
- theoretical fit from verified fit.

All volatile facts—cluster inventory, model revision, package version, and supported status—must carry a “last verified” date.

---

## 15. Maintenance and change policy

1. Pin repository code dependencies and model revisions for a release.
2. Review cluster state and model upstreams monthly during the Challenge preparation period.
3. Re-run probes after UC Merced maintenance, driver changes, runtime upgrades, or partition changes.
4. Add a new model through a pull request containing:
   - source and license review;
   - profile and schema tests;
   - storage/VRAM estimate;
   - successful cluster result or an explicit experimental label;
   - README update; and
   - rollback/removal note.
5. Never silently replace a checkpoint behind an existing profile. Add a revisioned profile or release note and revalidate.
6. Deprecate a broken model profile before removal and keep the failure explanation when educationally useful.
7. Treat web model rankings as discovery signals, not evidence. Prefer model cards, technical reports, runtime documentation, and observed Pinnacles results.

---

## 16. Proposed milestones

| Milestone | Deliverable | Depends on |
|---|---|---|
| M0 | Student account/resource decision record | actual Challenge account and/or CIRT response |
| M1 | Safe repository skeleton, CI, and probe scripts | M0 resource names |
| M2 | Reproducible environment and secure download flow | GPU runtime and network probe |
| M3 | Gemma 4 E2B vertical slice and first complete README path | M2 |
| M4 | Generic model profile launcher | M3 |
| M5 | Gemma 4 family and Nemotron 3.5 Lightning validated | M4 |
| M6 | Llama 4 Scout and Mistral Small 4 validated | license access and M4 |
| M7 | Largest-model experiment report | stable one-/two-GPU results and multi-node communication probe |
| M8 | Student clean-room trial and `v0.1.0` release | M0–M7 core criteria |

M0 is the only administrative blocker. Implementation of safe local structure and small test-partition examples can proceed in parallel, but long-queue/H200 claims cannot be finalized without it.

---

## 17. Open decisions for project review

1. What exact account/project association will LLNL Data Science Challenge students receive?
2. Does the workshop need only interactive demonstrations, or a server that remains available for hours/days?
3. Will CIRT provide a reservation and/or shared read-only model cache for the event?
4. Which client experience should be primary: command-line `curl`/Python, Jupyter, or an optional web UI?
5. Is multimodal input a core learning objective, or should the first release focus on text and tool-call formatting?
6. Should the repository optimize for vLLM as the common API server, or allow Transformers for the newest architectures until vLLM support stabilizes?
7. May students use SSH port forwarding to compute-node services, and what approved route should the tutorial show?
8. Which benchmark best represents the Challenge workload without exposing Challenge data?
9. Should third-party quantized checkpoints be permitted, or should all core paths use only developer-published artifacts?
10. Who will own monthly revalidation after the Challenge ends?

Defaults until these are answered: text-first, CLI/Python client, loopback-only service, vendor checkpoints, small explicit context, one GPU, and the `test` partition for access-neutral examples.

---

## 18. Immediate next actions after approval

1. Confirm the M0 account/access questions with an actual Challenge student account and CIRT.
2. Add the repository safety skeleton and capability probe scripts.
3. Run the GPU probe on A100, L40S, and H200 using short `test` allocations available to the maintainer, without using `dept.appliedmath`.
4. Select and pin the Python, PyTorch, Transformers, and serving-runtime versions based on observed driver/CUDA compatibility.
5. Implement the Gemma 4 E2B vertical slice.
6. Draft the student README from the commands and outputs that actually passed.
7. Add Nemotron 3.5 Lightning, Gemma 4 26B/31B, Llama 4 Scout, and Mistral Small 4 in that order.
8. Attempt multi-node/largest-model experiments only after the core tutorial is reliable and CIRT confirms the intended use is appropriate.

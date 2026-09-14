# Environment setup: detailed explanation and questions

[Return to the tutorial](../README.md#2-install-the-gemma-and-opencode-tooling).

An environment is the combination of interpreter, installed libraries,
executables, configuration, and system resources a program uses. For this
tutorial, installing Python packages is only one part: the model also needs
the correct checkpoint, serving runtime, GPU allocation, and client settings.

These questions anticipate what a reviewer might ask about the environment;
they are an interpretation of the mentor's feedback, not her exact questions.
Answers about the current setup come from the checked-in scripts and pins.
The explanations of why the choices are useful are technical rationale, not
a claim about undocumented historical decisions.

The main tutorial records the existing Pinnacles validation and its limits.
The future-project guidance below describes how to extend that setup; it is
not a newly validated PyTorch training or Matplotlib installation recipe.

## 1. Which environment are we setting up?

There are several cooperating parts:

| Part | What runs there | What it needs |
|---|---|---|
| Cluster platform | Slurm jobs and system software | Sponsored account, permitted partition, Linux tools, campus modules, GPU driver on GPU nodes |
| Model server | vLLM or llama.cpp, loading weights and answering requests | Its pinned runtime, checkpoint, sufficient RAM/VRAM, launch configuration |
| OpenCode or Codex client | Conversation and file/shell tools | Selected harness executable, provider configuration, access to the endpoint |
| Project execution | The Python programs and tests the harness runs | That project's interpreter and libraries |
| Benchmark evaluation, when used | Dataset preparation and grading | Suite-specific requirements described in the [benchmark guide](../benchmarks/README.md) |

**Why distinguish them?** An HTTP request connects the harness to the model;
it does not make the model server's Python packages available to every tool
process. Conversely, adding a plotting library to a project does not require
changing the software that serves the model. The small introductory exercise
can use the serving environment's Python because it needs only the standard
library. Larger projects should have their own environment.

Codex is an optional separate installation; see [the harness choice guide](../README.md#16-choose-codex-instead-of-opencode).
Its launcher reuses the model profiles and keeps Codex state under
`/data/$USER/pinnacles-agents/codex/PROFILE`. Project interpreter and dependency
guidance applies to both harnesses. The model setup scripts continue to install
OpenCode, and existing OpenCode-specific paths below describe that route.

## 2. What must already exist before setup?

The published route targets Pinnacles Linux x86-64, with Bash, Slurm commands,
the campus module system, and standard utilities such as Git, curl, tar, and
SHA-256 checking. The Python/vLLM setup scripts require Python **3.11**, supplied
by `anaconda3/2023.09-0`. Some quantized-profile scripts also use `flock` and
inspect Slurm job ownership and the compute process's cgroup.

Installation needs outbound access to the package and release sources used by
the scripts. Model downloads need access to Hugging Face; authentication or
license acceptance may depend on the model and account. Inference then uses
the downloaded checkpoint. Persistent data space and disposable scratch space
must be available; consult the selected lesson's budget before downloading.

The GGUF route additionally uses the campus CUDA/compiler toolchain and CMake;
see question 9. GPU inference needs a GPU compute allocation and a compatible
administrator-managed driver. A CPU setup allocation cannot establish that
GPU execution works.

**Why these prerequisites?** They match the repository's cluster-specific
scripts and existing lessons. The scripts use absolute `/data` and `/scratch`
paths and a Linux OpenCode release; they are not portable laptop installers.

## 3. Why load Anaconda but create a virtual environment with uv?

The campus module supplies the starting Python interpreter. The setup scripts
do not create a Conda environment or install project packages into Anaconda's
base environment. Instead, they create a small bootstrap environment with
Python's `venv`, install `uv==0.12.13` there, and use that tool to create and
synchronize the model-serving environments.

**Why this choice?** It uses an available campus interpreter while giving each
serving environment its own Python package directory. The bootstrap tool can
manage a target environment without being installed into that target. This
also avoids needing administrator access or modifying shared software.
Virtual environments isolate Python packages; they do not isolate files,
network access, or GPUs. See [Python's venv documentation](https://docs.python.org/3.11/library/venv.html).

Conda or a container could support a different deployment, but would require
its own dependency resolution and validation. They are not drop-in replacements
for the paths and installers in this tutorial.

## 4. Why Python 3.11 and these particular package versions?

Python 3.11 is the explicit major/minor version check in the vLLM setup scripts.
The package versions are the repository's recorded serving baseline, not a
recommendation to install whatever releases are newest.

All three current Python lock files contain these entries:

| Package | Recorded version | Role |
|---|---|---|
| `vllm` | `0.29.0` | Model serving, scheduling, API, and model-specific integration |
| `torch` | `2.13.0` | PyTorch tensor operations and execution used by the serving stack |
| `transformers` | `5.17.0` | Model/tokenizer integration used by the stack |
| `numpy` | `2.3.5` | Numerical array support |
| `huggingface-hub` | `1.31.0` | Model-repository access and downloads |

The full sources of truth are
[`gemma-requirements.lock`](gemma-requirements.lock),
[`muse-requirements.lock`](muse-requirements.lock), and
[`nemotron-requirements.lock`](nemotron-requirements.lock).
These contain many more packages, including CUDA-related components.

**Why pin the combination?** Serving depends on compatibility across model
support, compiled kernels, PyTorch, and CUDA libraries. Replacing one package
can affect the others. vLLM's installation documentation specifically explains
the relationship between its compiled components and the PyTorch/CUDA build.
See [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).
Upstream documentation can change; the repository files identify this
tutorial's baseline. A Python minor-version change also needs a fresh check
of wheel availability and runtime behavior.

## 5. Where does everything go, and why?

Paths below are relative to `/data/$USER/pinnacles-agents` unless stated
otherwise. The scripts expand `$USER` to the current account.

| Location | Contents and purpose |
|---|---|
| `bootstrap/` | Python environment containing the package-management tool |
| `envs/gemma-vllm/` | Shared runtime for Gemma BF16 and Gemma QAT |
| `envs/muse-vllm/` | Muse BF16 Python runtime |
| `envs/nemotron-vllm/` | Shared runtime for Nemotron Super and Lightning |
| `tools/opencode-v1.18.30/` | OpenCode executable, separate from Python packages |
| `tools/llama.cpp-<commit>/` | Compiled Muse GGUF runtime |
| `models/` | Downloaded model files; separate checkpoints consume separate space |
| `huggingface/` | Hugging Face state used by the Python model workflows |
| `opencode*/` | Profile-specific configuration, data, and session state selected by activation |
| `/scratch/$USER/pinnacles-agents/` | Installation/build caches, temporary files, and runtime caches, including profile subdirectories |
| Tutorial checkout in home | Small source files, scripts, configurations, and lock files |

**Why this layout?** Models and environments should survive the end of a job;
temporary build/cache files can be recreated. Keeping large artifacts outside
the Git checkout also avoids accidentally versioning model weights. The
[storage reference](../README.md#storage) explains quotas and scratch retention.

The Gemma and Nemotron quantized profiles share their family's Python runtime;
they do not each get a separate venv. Consequently, changing Gemma's packages
affects both Gemma lessons, and changing Nemotron's affects both NVIDIA lessons.
Separate configuration directories do not imply separate Python dependencies.

## 6. Is PyTorch missing from the requirements?

**No.** The distribution name used by Python package managers is `torch`, and
all three vLLM lock files already include `torch==2.13.0`. Students following
those setup scripts do not need a second PyTorch installation for model serving.
NumPy is already present too.

**Why not add another torch entry or upgrade it separately?** The lock already
specifies the serving dependency. A second generic requirement adds ambiguity,
and an independent upgrade can change the validated serving combination.
PyTorch being available in `gemma-vllm` also says nothing about whether it is
installed in a separate project interpreter or on a laptop.

If a future project imports PyTorch, record it as a direct dependency of that
project, even though the server separately uses it. If the project only sends
HTTP requests to the endpoint, it does not need PyTorch merely because the
remote model uses GPUs.

## 7. Should other dependencies be added now?

The current repair exercise imports `unittest` and the supplied `summary`
module. `unittest` comes with Python; it is not a missing pip requirement.
Matplotlib and pytest are absent from the serving locks, and neither is
required by that exercise. Other packages should be evaluated against the
actual project rather than added as a generic data-science bundle.

| Future task | Dependency to consider | Where it belongs |
|---|---|---|
| Save charts of results | `matplotlib` | Project/analysis environment |
| Manipulate tabular datasets | `pandas`, if used by the code | Project environment |
| Numerical or statistical routines | `numpy` or `scipy`, if imported | Project environment |
| Train or run a separate neural network | `torch`, plus only the domain libraries actually used | Project ML environment |
| Run tests written for pytest | `pytest` | Project test/development requirements |
| Work in notebooks | Chosen notebook frontend and kernel dependencies | Separate notebook/project setup |

**Why wait for a concrete task?** Each additional dependency adds installation
size and compatibility constraints. Keeping the serving baseline stable makes
model failures easier to reproduce. No extra mandatory requirements are needed
to complete the existing exercise. A future plotting lesson should introduce
its own tested dependency specification when its code is added.

## 8. Do I need to install CUDA, and is it the same as the GPU driver?

They are different components. The **driver** is system software through which
processes access the GPU. A **CUDA runtime/library set** supports GPU programs.
A **CUDA toolkit** includes development tools such as the `nvcc` compiler.
Python packages cannot replace the cluster's kernel driver.

The vLLM locks include CUDA-related packages, and the setup uses prebuilt
packages. Follow that lesson's existing setup rather than independently
installing a different toolkit. The GGUF build explicitly loads a campus CUDA
module because it compiles C++/CUDA code. GPU compatibility still depends on
the runtime build, driver, GPU architecture, and kernels used by the model.

**Why does this matter for future PyTorch work?** A project needs a PyTorch
build appropriate for its operating system and CPU/GPU target. Use the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
when designing that separate environment, then record the chosen build and
package source. Do not substitute its latest install command into the serving
setup. An installed CUDA-capable package does not allocate a GPU; Slurm does.

## 9. Why is Muse GGUF different from the other environments?

The [GGUF setup script](../scripts/setup-muse-gguf.sh) builds `llama-server`
from llama.cpp tag `b10353`, commit
`f8def7fe168bab245fbf15d3f18b26dbb1ef73c8`. It enables CUDA, targets architectures
`80;89`, disables the UI, and installs the compiled binaries and integrity
metadata under `tools/`. The exact module and campus CMake path are recorded
in [`muse-gguf-pins.sh`](muse-gguf-pins.sh).

**Why a different dependency description?** This path serves a GGUF checkpoint
through a compiled program, so a Python requirements file cannot describe its
whole build. It needs source and toolchain pins. It does not use PyTorch for
serving, although helper scripts still invoke Python for checks. Its activation
script adds the compiled tools to `PATH`; it does not select a project Python.
Adding Matplotlib would therefore still be a separate project concern.

## 10. Why install in a CPU allocation and serve in a GPU allocation?

The tutorial uses the login node for editing and job submission, a CPU compute
allocation for installation/downloads, and a GPU compute allocation for model
inference. Setup and downloads can consume significant CPU, memory, disk I/O,
and network bandwidth without requiring an allocated GPU.

**Why this separation?** It follows the tutorial's cluster workflow and avoids
holding a scarce GPU throughout dependency installation and large downloads.
Successful installation verifies only part of the setup. GPU checks and actual
generation belong in the requested GPU allocation. Similarly, agent-triggered
project tests on Pinnacles belong in a compute allocation, even if the tests
themselves only use CPUs.

## 11. What does activation do? Does it install anything?

For the vLLM profiles, the supplied `activate-*.sh` scripts prepend the selected
environment's `bin` directory and OpenCode directory to `PATH`. They export
storage/configuration variables and create the corresponding state directories.
They do not install dependencies or edit shell startup files.

**Why source the script?** `source` changes the current shell, so programs
launched from it inherit the settings. Executing it with `bash` would make
those environment changes only in the child process. A new shell needs its own
activation. The scripts are custom PATH setup, not the standard venv activation
script; they do not provide a matching `deactivate` function or reliably set
`VIRTUAL_ENV`. A prompt label is therefore not proof of the interpreter in use.

For diagnosis, the useful facts are the executable actually selected and
Python's `sys.executable`, rather than the environment name shown in a prompt.
An alias, a later module load, or a project activation can change command
selection. Prefer a fresh shell with the intended model activation when
switching workflows rather than stacking multiple profile activations.

## 12. Which Python will OpenCode use for my project?

OpenCode's shell tools execute on the machine where OpenCode runs. An ordinary
`python` command is resolved from that tool process's environment, including
its inherited PATH, unless the project or command explicitly selects another
interpreter. The supplied small exercise uses the activated model environment.

**Why does this cause missing-package errors?** Installing Matplotlib into one
venv does not install it into another. If OpenCode runs on your laptop through
a tunnel, packages installed on Pinnacles are not available to its laptop
commands. The tunnel carries API traffic, not a Python environment.

For an extended project, make its interpreter path and test command explicit
in that project's instructions, and verify them from the actual tool process.
Do not assume that activating an environment in one temporary shell tool call
will persist into later tool calls. The existing model launchers use explicit
serving executable paths, which helps keep project interpreter choices separate
from the server. A newly added project still needs an end-to-end tool check.

## 13. How should a future project add and record its requirements?

Use a separate environment owned by that project. The following is a design
and validation workflow, not an additional cluster installation lesson:

1. Identify the actual imports, Python version, test runner, and CPU/GPU needs.
   A plotting-only project and a GPU training project need different packages
   and different resource budgets.
2. Declare direct dependencies in the project's existing `pyproject.toml` or
   requirements input file. Put test/notebook extras in appropriate optional
   groups when the project's tooling supports them. Avoid competing sources
   of truth for the same environment.
3. Resolve those declarations in a fresh project environment on the intended
   platform. For PyTorch, record the CPU/CUDA build and wheel source as well as
   the version. Select versions by compatibility and tests, not by copying all
   packages from a serving lock.
4. Save the resolved dependency set in the project's lock format. With a
   requirements-based uv workflow, compilation resolves the input and syncing
   installs the resulting set. [uv's locking documentation](https://docs.astral.sh/uv/pip/compile/)
   explains these separate operations.
5. Check dependency consistency and run the actual workload: a plot that saves
   successfully, project tests, or a small intended PyTorch computation. GPU
   work must run in a suitable Slurm GPU allocation. Repeat the relevant task
   through OpenCode using the same project interpreter.
6. Recreate the project environment from its saved specification and repeat
   the checks. Commit the dependency declarations, lock, and tested usage
   instructions; keep the installed environment outside Git.

**Why both declarations and a lock?** Declarations explain what the project
needs directly; a resolved lock records the versions of dependencies pulled
in underneath them. A blanket freeze of the serving environment would mix in
unrelated server libraries and conceal the project's real requirements.

The current repository already has serving locks. It does not have a separately
validated general-purpose analysis/training environment; this guide does not
invent version pins for one before its workload exists.

## 14. Can I just install a new package into gemma-vllm?

That would modify the runtime shared by Gemma BF16 and QAT. Even when an
installation appears harmless, its dependency resolution may replace an
existing library. It also makes the installed state diverge from the lock.

The setup scripts call `uv pip sync` with an explicit target interpreter.
Syncing restores the package set described by the lock and can remove
additional packages that are absent from it. Thus an ad hoc Matplotlib install
could disappear the next time setup runs.
See [uv's environment synchronization semantics](https://docs.astral.sh/uv/pip/compile/).

**Why separate environments?** A project can change its plotting or ML stack
without changing the server. If a serving dependency itself must change,
prepare a candidate runtime and validate it before updating the published lock.
Do not rerun setup against an environment while one of your servers is using
it. Installation locks in some scripts do not protect running servers from
package changes, and not all installers share the same locking behavior.

## 15. Will pip work inside these environments?

The bootstrap environment has pip because the scripts use it to install uv.
The serving environments are created by uv, and their locks do not include
pip. Therefore `python -m pip` is not guaranteed to exist in a serving venv.
A bare `pip` command could even resolve to a different environment on PATH.

**Why use an explicit interpreter target?** The existing installers manage
packages with the bootstrap uv executable and its `--python` option, making
the destination unambiguous. A future project should use its chosen package
manager with an equally clear target. A missing pip module is not, by itself,
evidence that the model environment failed to install. See
[uv's environment selection documentation](https://docs.astral.sh/uv/pip/environments/).

## 16. How would plotting work without a desktop display?

For a future plotting project on a compute node, design scripts to save plots
to files. Matplotlib supports noninteractive backends such as Agg for PNG
output, and supports saving PDF/SVG as well. A GUI window is unnecessary for
that workflow. The `MPLBACKEND` setting can select a backend for a process;
`savefig` writes output. See [Matplotlib's backend documentation](https://matplotlib.org/stable/users/explain/figure/backends.html).

**Why choose file output?** Batch jobs and remote shells may have no display,
and saved figures can be inspected after the job ends. Validate that the output
file is nonempty and visually correct, not only that `import matplotlib`
succeeds. Choose a writable cache/config location if the default is unsuitable,
and save important figures outside disposable scratch. Ordinary plotting does
not require a GPU. Notebook widgets or interactive windows introduce additional
dependencies and should be treated as their own setup.

## 17. Can project PyTorch code share the model server's GPU?

A separate venv provides package isolation, not a separate GPU memory pool.
The model server can already occupy most of the allocation's VRAM, including
its reserved KV cache. A second PyTorch process may run out of memory even if
its own model seems small. CPU RAM requested with Slurm's `--mem` does not
increase VRAM.

**Why plan resources separately?** A CPU-only analysis task can coexist within
the job's CPU/RAM limits, but GPU training or inference needs an explicit memory
and scheduling plan. A separate GPU allocation, or stopping the server before
running the project workload, may be appropriate. Do not bypass Slurm's device
visibility or assume that creating another environment reserves hardware.

For a proposed GPU project, distinguish three checks: PyTorch imports, CUDA is
available to that process, and a representative GPU computation succeeds.
Only the last exercises the intended computation; none alone proves that it
can run concurrently with the serving workload.

## 18. How reproducible is the current setup?

The repository records Python major/minor, exact package versions, OpenCode's
version and archive checksum, model revisions, and profile-specific model or
runtime integrity information. The GGUF path also records a source commit and
build profile. These make the setup substantially more specific than an
unpinned installation command.

**What is not locked?** The Python requirement files are version-pinned package
lists, not complete machine images: they do not include per-wheel hashes or
encode the entire OS, driver, module installation, and hardware state. Python's
patch version is not explicitly enforced by the setup assertion. Available
package artifacts and campus software can also change. Identical package pins
do not promise bitwise-identical model output or identical performance.

**Why document the limits?** Reproduction needs both the software specification
and the execution context. For a validated change, record the repository
revision, Python version, package source/build choices, GPU and driver,
relevant modules, checkpoint revision, server options, project test command,
and observed results. Checksum checks cover only the artifacts that the scripts
actually verify; they are not a blanket guarantee for every dependency.

## 19. How do I tell whether setup really worked?

The existing Python installers synchronize the lock and run `uv pip check`.
That checks declared package consistency; it does not execute every library
or prove CUDA/model compatibility. The tutorial then checks the allocated
node/GPU, server startup, HTTP health and model identity, generation, and the
OpenCode file-edit/test round trip.

**Why this sequence?** Each stage answers a different question. An import can
succeed while a GPU kernel fails; a healthy endpoint can exist while a tool
parser is misconfigured. The exercise's initial tests deliberately fail until
the code is repaired, so those initial failures are expected, not evidence of
missing requirements. The final four tests must pass without changing the test
file. Follow the [existing Gemma lesson](../README.md#4-request-an-h200-compute-shell)
or the corresponding model lesson for the published commands.

For a future environment, add checks for its actual dependencies: project
imports and tests, a saved plot for Matplotlib, and an intended computation for
PyTorch. Revalidate the serving workflow if its own dependencies change. No new
Pinnacles runtime validation is claimed by this documentation update.

## 20. What are the most likely environment problems?

| Symptom | Likely explanation and next check | Why that check helps |
|---|---|---|
| Setup requests Python 3.11 | The campus module was not loaded, or another interpreter takes precedence; return to the lesson's setup shell/module step. | The script checks the interpreter actually resolved as `python3`. |
| A project reports `ModuleNotFoundError` | Compare the tool process's interpreter with the environment where the dependency was installed; then check the project's declared requirements. | Package availability is interpreter-specific. |
| `No module named pip` | The selected uv-created runtime may not contain pip; use its documented manager and explicit target. | A serving environment does not need pip installed inside it to be managed. |
| A package disappears after setup | It was installed outside the lock and sync restored the declared set. | Ad hoc additions are not durable requirements. |
| Dependency check succeeds but serving fails | Inspect the server log for model, driver, kernel, or configuration failures. | Package metadata checks do not execute the model. |
| CUDA is unavailable to a project | Check the real compute allocation, selected interpreter/build, and Slurm-visible devices. | Installing torch alone does not grant GPU access. |
| GPU out of memory | Check other processes in your allocation and the combined server/project memory requirement. | Separate Python environments still share allocated hardware. |
| Plotting fails with a display/backend error | Check whether the project expects a GUI; use a tested file-output design for batch work. | Rendering a file does not require a desktop session. |
| Setup/download fails for lack of space | Check personal data and scratch usage against the selected lesson's budget. | Filesystem-wide free space is not your personal quota. |
| Commands worked in one terminal but fail in another | Re-enter the appropriate allocation and apply its activation in that shell. | Shell settings and node-local endpoints do not follow every new terminal. |

For a useful issue report, include the failed command, interpreter path,
relevant versions, allocation/GPU details, and the pertinent error excerpt.
This helps distinguish a project dependency problem from a serving or cluster
problem without guessing at a package upgrade.

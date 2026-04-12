# smallctl Bundle

This directory is a portable install bundle for `smallctl`.

## Install

```bash
cd smallctl
./install.sh
```

The installer will create a virtual environment in `./.venv` by default and install the package in editable mode.

## Activate

```bash
source .venv/bin/activate
```

## Run

```bash
smallctl --help
```

Example task run:

```bash
smallctl --task "Read README.md and summarize it"
```

## Configure

Copy `.env.example` to `.env` and adjust the values for your environment.

```bash
cp .env.example .env
```

Useful rollout toggle:

- `SMALLCTL_STAGED_REASONING=true` or `--staged-reasoning` enables the staged-reasoning strategy flag without changing the default runtime behavior.

LM Studio notes:

- Prefer `first_token_timeout_sec: 45` or higher for slower local generations.
- `provider_profile: lmstudio` will now try LM Studio's native unload API automatically.
- Configure `backend_restart_command` if you want restart fallback when unload does not recover.
- A common setup is:

```yaml
provider_profile: lmstudio
first_token_timeout_sec: 45
backend_restart_command: "systemctl --user restart lmstudio.service"
```

## Notes

- The bundle includes the `smallctl` source code and installer.
- Logs, temp files, caches, and secret `.env` files are intentionally excluded.
- The current tool path is: pending tool call creation -> dispatch -> persisted tool execution record -> artifact creation -> prompt assembly.
- `dispatch_tools()` and `persist_tool_results()` are the main execution/persistence seam, while `PromptAssembler.build_messages()` is the prompt seam.

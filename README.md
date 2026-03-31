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

## Notes

- The bundle includes the `smallctl` source code and installer.
- Logs, temp files, caches, and secret `.env` files are intentionally excluded.

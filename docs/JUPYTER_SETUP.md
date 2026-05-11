# Remote JupyterLab + VS Code setup

End-to-end instructions for running the SSTCa2 pipeline on a Linux server
through a persistent JupyterLab kernel, edited from your local machine via
VS Code's Remote-SSH extension.

**Goal architecture**
- Code, data, conda env, and the running kernel all live on the server.
- JupyterLab listens on the server's loopback interface only (`127.0.0.1:8889`),
  protected by a token.
- The kernel is started inside a `tmux` or `screen` session so it survives
  browser closes, SSH drops, and laptop sleep.
- Your local VS Code window (via Remote-SSH) edits server files and attaches
  to the same kernel that a browser would use.

**What this setup does *not* cover** (out of scope by design)
- Auto-restart after a server reboot (would need systemd).
- Multi-user access / proxy auth (single-user, token-gated, loopback-only).
- Checkpoint-and-resume after a kernel crash.

---

## Part A — One-time server-side setup

Assumes you already have miniforge installed on the server and SSH access.
Replace `<server>` and `<user>` with your hostname/username throughout.

### A1. Create the conda env

SSH into the server and run:

```bash
ssh <user>@<server>
mamba create -n sstca2 python=3.11 \
    jupyterlab ipykernel ipywidgets \
    numpy pandas matplotlib scipy scikit-learn statsmodels \
    xarray scikit-image seaborn opencv \
    umap-learn networkx tqdm pyyaml optuna
mamba activate sstca2
```

Cross-check that no imports are missing:

```bash
cd /Users/vsekulic/code/sstca2
grep -hE "^(import|from) " *.py | sort -u
```

Install any missing packages with `mamba install -n sstca2 <pkg>` (prefer
`mamba`/`conda-forge`; fall back to `pip` only if a package isn't on
conda-forge).

### A2. Register the env as a Jupyter kernel

```bash
mamba activate sstca2
python -m ipykernel install --user --name sstca2 \
    --display-name "SSTCa2 (sstca2 env)"
```

Verify:

```bash
jupyter kernelspec list
```

You should see `sstca2` in the list.

### A3. Generate a Jupyter config and a token

```bash
jupyter server --generate-config
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the printed hex string — that's your auth token. Keep it private.

Edit `~/.jupyter/jupyter_server_config.py` and set the following (the file
already contains commented-out versions of every line; uncomment and edit
in place, or just append a new block):

```python
c.ServerApp.ip                 = '127.0.0.1'   # loopback only
c.ServerApp.port               = 8889          # any free port
c.ServerApp.open_browser       = False
c.ServerApp.root_dir           = '/Users/vsekulic/code/sstca2'
c.ServerApp.token              = '<paste-the-hex-token-here>'
c.ServerApp.password           = ''            # rely on token
c.ServerApp.disable_check_xsrf = False
# Never auto-cull idle kernels (the pipeline holds large state)
c.MappingKernelManager.cull_idle_timeout = 0
c.MappingKernelManager.cull_connected    = False
# Don't shut down the server when no client is connected
c.ServerApp.shutdown_no_activity_timeout = 0
```

> **Security note.** Because the server binds to `127.0.0.1`, the only way
> for traffic to reach it is through the SSH tunnel (Part C). The token is
> a second layer of protection in case the tunnel is misconfigured. Do
> **not** change `ip` to `0.0.0.0` — that would expose the server to the
> network.

### A4. (Optional) Convenience launch script

A reusable launcher is provided at [`scripts/start_jupyter.sh`](../scripts/start_jupyter.sh).
Make sure it's executable on the server:

```bash
chmod +x /Users/vsekulic/code/sstca2/scripts/start_jupyter.sh
```

You can symlink it onto your PATH if you like:

```bash
mkdir -p ~/bin
ln -sf /Users/vsekulic/code/sstca2/scripts/start_jupyter.sh ~/bin/start_jupyter.sh
```

---

## Part B — Starting the kernel under a terminal multiplexer

Either `tmux` or `screen` works. Pick one. Both keep the JupyterLab process
alive after you log out.

### Option B1 — tmux

```bash
ssh <user>@<server>
tmux new -s jupyter
# inside the new tmux pane:
~/bin/start_jupyter.sh
# (or: mamba activate sstca2 && cd /Users/vsekulic/code/sstca2 && jupyter lab)
```

Detach with `Ctrl-b d`. The Jupyter server keeps running.

Useful commands:

| Action            | Command                          |
|-------------------|----------------------------------|
| Reattach          | `tmux attach -t jupyter`         |
| List sessions     | `tmux ls`                        |
| Kill session      | `tmux kill-session -t jupyter`   |

### Option B2 — screen

```bash
ssh <user>@<server>
screen -S jupyter
# inside the screen pane:
~/bin/start_jupyter.sh
```

Detach with `Ctrl-a d`.

Useful commands:

| Action            | Command                          |
|-------------------|----------------------------------|
| Reattach          | `screen -r jupyter`              |
| List sessions     | `screen -ls`                     |
| Kill session      | `screen -X -S jupyter quit`      |

Optional `~/.screenrc` to make `screen` nicer:

```
escape ^Bb        # remap escape to Ctrl-b so it doesn't clash with bash Ctrl-a
defscrollback 10000
```

### Confirm the server is up

On the server, in a fresh shell:

```bash
pgrep -af jupyter
# Expected: a line ending in `jupyter-lab` (or `jupyter-server`)
```

The first time `jupyter lab` starts it prints a URL like
`http://127.0.0.1:8889/lab?token=...`. Note the token matches the one in
your config file. From here on you can use either that URL or the bare
hostname form below.

---

## Part C — Connecting from your local machine

### C1. SSH tunnel

On your **local** machine, in a terminal:

```bash
ssh -N -L 8889:127.0.0.1:8889 <user>@<server>
```

Leave it running. `-N` means "no remote command, just forward ports". The
local port `8889` now forwards to the server's loopback `8889`.

Optionally make the tunnel automatic by adding to `~/.ssh/config`:

```
Host sstca2-tunnel
    HostName <server>
    User <user>
    LocalForward 8889 127.0.0.1:8889
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then `ssh -N sstca2-tunnel` opens the tunnel.

### C2. Browser access (sanity check)

Open in your local browser:

```
http://127.0.0.1:8889/lab?token=<your-token>
```

You should see the JupyterLab UI. The file browser shows the contents of
`/Users/vsekulic/code/sstca2` (the `root_dir` you set in step A3).

### C3. VS Code Remote-SSH (recommended)

This is the preferred editing environment. VS Code runs locally; everything
it touches (editor buffers, terminal, kernel) runs on the server.

1. Install the **Remote - SSH** extension in local VS Code
   (`ms-vscode-remote.remote-ssh`).
2. Install the **Python** and **Jupyter** extensions
   (`ms-python.python`, `ms-toolsai.jupyter`). After connecting via
   Remote-SSH, you'll be prompted to install them on the server side
   too — accept.
3. `Cmd-Shift-P` (or `Ctrl-Shift-P`) → **Remote-SSH: Connect to Host...** →
   pick `<user>@<server>` (or the `sstca2-tunnel` alias from your SSH
   config).
4. In the remote VS Code window: **File → Open Folder...** →
   `/Users/vsekulic/code/sstca2`.
5. Open `run_pipeline.ipynb`.
6. Top-right of the notebook editor: **Select Kernel** → **Existing Jupyter
   Server...** → paste `http://127.0.0.1:8889/?token=<your-token>` → pick
   **SSTCa2 (sstca2 env)** from the kernel list.

VS Code now talks to the same kernel the browser does. The notebook reads
the project layout *on the server*; edits to `.py` files in adjacent tabs
save to the server's filesystem; `%autoreload 2` in the notebook picks up
the edits without re-running the data-load cell.

> **One-host note.** With Remote-SSH, VS Code already opens its own SSH
> connection to the server. The Jupyter extension will reuse that
> connection's port forwarding, so you can skip the manual `ssh -L` from
> step C1 if you only ever connect via VS Code. Keep the manual tunnel for
> browser access.

---

## Part D — Daily workflow

Once the one-time setup is done:

1. **On the server**, make sure the Jupyter process is alive:
   ```bash
   ssh <user>@<server> 'pgrep -af jupyter'
   ```
   If nothing prints, reattach (`tmux attach -t jupyter` or `screen -r
   jupyter`) and re-run `start_jupyter.sh`.
2. **On your local machine**, open the SSH tunnel (or use VS Code
   Remote-SSH, which does this automatically):
   ```bash
   ssh -N sstca2-tunnel
   ```
3. **Open VS Code** → Remote-SSH connect → open the workspace folder →
   open `run_pipeline.ipynb` → select the existing kernel.
4. **Step through the notebook**:
   - Cell 1: `import` modules.
   - Cell 2: build `cfg = PipelineConfig(...)`.
   - Cell 3 (heavy, ~10 min): `ds = SSTCa2_loader.load_all_mice(...)`.
   - Cell 4 (instant): `globals().update(vars(ds))` — exposes `TFC_cond`,
     `Test_B`, `engram_id`, etc. as top-level names.
   - Subsequent cells: call `pipe.run_*(ds, cfg)` for big pipelines, or
     write inline scaffolding code against the top-level names.
5. **Iterate on a module** (e.g. edit `SSTCa2_population.py`): save the
   file → `%autoreload 2` picks it up → re-run only the affected analysis
   cell. `ds` stays loaded.
6. **Disconnect freely**: close VS Code, suspend the laptop, lose WiFi —
   the kernel and `ds` survive. Reconnect and resume.

---

## Part E — Troubleshooting

**`http://127.0.0.1:8889` shows "connection refused" in the browser**
- Tunnel not running. Check the local terminal where `ssh -N ...` was
  started. If it exited, restart it.
- Server-side Jupyter not running. SSH in and `pgrep -af jupyter`.

**JupyterLab loads but says "Invalid credentials"**
- Token mismatch. Re-read the URL in your tmux/screen pane (or
  `~/.jupyter/jupyter_server_config.py`) and paste the correct token.

**Kernel says "Dead" or VS Code can't connect to it**
- Kernel ran out of memory or hit an unhandled exception. Reattach the
  multiplexer pane to read the traceback. Restart the kernel from the
  notebook UI and re-run from the load cell.

**`%autoreload 2` not picking up changes**
- The codebase uses several `from X import *` patterns, which can defeat
  autoreload. Run an explicit reload cell, bottom-up by dependency:
  ```python
  import importlib
  for m in ['SSTCa2_utilities','SSTCa2_sessions','SSTCa2_engram',
           'SSTCa2_analysis','SSTCa2_spatial','SSTCa2_decoder',
           'SSTCa2_population','SSTCa2_isomap','SSTCa2_epoch_analysis',
           'SSTCa2_pipeline','SSTCa2_loader','SSTCa2_config']:
      importlib.reload(__import__(m))
  ```
  Then re-run any cell whose imports came `from X import *`.

**Server reboot — everything is gone**
- Expected (this setup intentionally does not auto-restart). SSH in,
  start a new tmux/screen session, re-run `start_jupyter.sh`, re-open
  the notebook in VS Code, re-run from the top.

**Token leaked / want to rotate**
- Generate a new one with
  `python -c "import secrets; print(secrets.token_hex(32))"`, update
  `~/.jupyter/jupyter_server_config.py`, kill and restart the Jupyter
  process inside the multiplexer pane.

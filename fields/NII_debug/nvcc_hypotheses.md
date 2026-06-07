# run_01 crash — `PermissionError [Errno 13] Permission denied: 'nvcc'`

## Root cause (in one line)
Our container ships **no `nvcc`** (it doesn't need one for training). When `torch.compile` looks up
`nvcc`, the container has none of its own, so it picks up the **host's `nvcc`** — which on the NII
node is **not executable** (blocked by the host's security setup) → crash. On our home box there is
no host `nvcc` in the way, so the same run works.

*(Technical detail: a non-executable nvcc raises `PermissionError`, which PyTorch does not catch; a
simply-absent nvcc IS caught and ignored — which is why home is fine. The error is `EACCES`/errno 13
= "found but not executable", distinct from a missing file or a full disk.)*

> Note: the `kernel cache directory could not be created` warnings in the log (lines 9416–9434) are
> **unrelated and harmless** — PyTorch just disables one on-disk cache and recompiles instead. Not
> nvcc, not the crash.

## 3 hypotheses for *why the host nvcc is non-executable*
1. **`noexec` mount (most likely).** The host CUDA dir is bind-mounted into the container but its
   filesystem is mounted `noexec` — standard security hardening on shared/scratch/NFS mounts.
   `execve` → EACCES.
2. **Permission bits / `root_squash`.** The mount allows exec, but the nvcc file's mode/owner
   (e.g. `0744`, owned by a different account under root-squashed NFS) gives the job user no
   execute permission.
3. **Host `$PATH` leak.** Singularity inherits the host environment, so a host CUDA dir
   (e.g. from `module load cuda`) is on `PATH` inside the container, which is how the host nvcc gets
   picked instead of "not found". (Usually combined with 1 or 2 as the reason it's non-exec.)

## What the NII host can run to tell us which case it is
Inside the same container/session that failed:
```bash
which -a nvcc                                         # is the path inside or outside the container?
N=$(readlink -f "$(which nvcc)"); echo "$N"; ls -l "$N"
findmnt -T "$N" -o TARGET,SOURCE,FSTYPE,OPTIONS       # 'noexec' in OPTIONS, or a host SOURCE  -> H1
stat -c 'mode=%A owner=%U:%G' "$N"; id                # no o+x / owner mismatch (mount is exec)  -> H2
echo "PATH=$PATH"; env | grep -iE 'CUDA_HOME|MODULEPATH|LMOD'   # a foreign CUDA dir on PATH    -> H3
nvcc --version                                        # reproduces the EACCES
```
Reading it: `noexec` in `findmnt` → **H1**; exec mount but no `o+x` / foreign owner → **H2**; a
non-container CUDA dir on `PATH` → **H3** (and 1/2 explains the non-exec part).

## Could a full `/tmp` disk cause this? — No.
Checked and ruled out (3 independent reviews): a full disk raises `ENOSPC` (errno 28) from *write*
calls; it cannot make `execve` return "permission denied" (errno 13). `run_01.log` has **zero**
out-of-space messages and successfully wrote several GB to `/tmp` (model + checkpoint) right up to
the crash. The "No space left on device" you saw was a **different** training run.

## Our fix — ship our own nvcc so we don't depend on the host (SHIPPED in v2.1)
The image now bundles a real, self-contained `nvcc` (the matched CUDA-13.0 pip wheels) at
`/opt/fields/bin/nvcc` and puts it **first on `PATH`** (baked, and re-pinned at launch), so the
container always uses ours and never the host's — regardless of `--nv` or host `PATH`. Verified
end-to-end with a non-executable nvcc bound at the NII vector. No change is requested on the NII side.
Image: `olmo-sft-v2.1-allsm.sif` / `chankhavu/olmo3-sft-v2.1:cu130-allsm`.

**Immediate unblock (no new image needed):** add to the `singularity run` command
```
--env PATH=/usr/local/nvidia/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```
This drops the CUDA dir from `PATH`, so nvcc resolves to "not found" → PyTorch's graceful path.
It keeps `torch.compile` and activation-checkpointing budget mode fully working.

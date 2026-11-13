# Kernel and processes

nano-rlm's workspace behaves like a Linux box an agent is logged into. A command can be stopped when it hangs, a crashed kernel comes back without losing the conversation, and what the agent deliberately leaves running keeps running after it is done, so a task that asks for a server gets graded with the server up.

## Timeouts and cancellation

| | Times out after | Then |
|---|---|---|
| `ipython` cell | `exec_timeout` (300 s), or the call's `timeout`, at most 600 s | The kernel is interrupted and the output so far is returned with a notice. A kernel that does not respond is restarted ([below](#kernel-crash-recovery)) |
| `bash` command | `exec_timeout` | Its process group is killed and the output so far is returned with the error |
| `rlm.shell` job | its own `timeout`, if given | Its process group is killed and `timed_out` is set |

A cancelled turn stops a running `bash` command the same way. A cancelled `ipython` call interrupts its cell, and its kernel is recovered if needed.

`bash` commands read an empty stdin, so a command that waits for input sees end-of-file at once rather than hanging or reading the harness's own streams. `rlm.shell` jobs have stdin closed too.

## Which processes outlive what

| Process | Kernel restart | Timeout or cancelled turn | Session end |
|---|---|---|---|
| Started from a cell and still attached to the kernel (`!cmd &`, `subprocess.Popen`) | stopped | the cell is interrupted | stopped |
| A `bash` tool command | unaffected | stopped, partial output kept | stopped |
| A `bash` tool command's background job (`cmd &`, `nohup cmd &`) | runs on | runs on | runs on |
| An `rlm.shell` job and the processes in its group | runs on | runs on (its own `timeout` applies) | stopped with its owner |
| An `rlm.services` service | runs on | runs on | runs on, until `stop()`; in a population episode its owner's retirement, stop or episode end stops it |
| A daemon in a session of its own (`setsid`, `start_new_session=True`) | runs on | runs on | runs on |

- **The kernel's session.** The kernel leads a session of its own. A restart or the end of the session stops every process still in that session, the way a closing terminal does, while processes that started a session of their own are left alone.
- **Bash commands.** Each `bash` command, whether a tool call or `await bash(...)` from a cell, leads its own process group. A timeout or cancellation kills that group while the command is still running. Once bash itself has exited, a job it put in the background outlives the call, even one that kept the call's output open until the timeout. A `bash` command run from a cell is in the kernel's session, so a restart stops it with the kernel.
- **Shell jobs and services.** The supervisor owns these. A job's leftover processes are killed when the job ends, and its owner's end ends it. A service runs in its own session and only `stop()` ends it ([Services](python-api.md#services)).

What keeps running after a session is the orchestrator's to clean up, for example by tearing down the container or VM.

## Kernel crash recovery

If the kernel dies during a cell, rlm starts a fresh kernel under the same agent and resumes the conversation with a recovery notice. A dead kernel found before a cell starts is restarted too, and that cell is returned unexecuted so the model can rebuild its Python state first. A kernel restarted after an unresponsive timeout or interrupt produces the same notice; an interrupt that leaves the kernel alive does not claim variables were lost.

Python variables, imports and in-kernel tasks are lost. The conversation, history, child agents, shell jobs, services, watches and inbox, read state included, belong to the supervisor and stay; `rlm.agent.list`/`get` and `rlm.shell.list`/`get` recover their handles. The interrupted cell is never replayed: its file writes or accepted spawns, sends and jobs may already have happened, so the model inspects them before retrying.

Recovery notices are `kernel_recovery` records in the history and messages in the context, so truncated tool output cannot hide them. rlm makes at most three recovery attempts without a completed cell in between. A failed start is reported, and a later `ipython` call may retry within that budget; once it is spent, IPython is unavailable but the agent can still answer or wait. Kernel death is noticed during a cell or at the next `ipython` call; nothing polls the kernel in the background. A crashed supervisor is not recovered.

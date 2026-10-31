# The verification gate: what "gate decision" means here

The full BeepSME agent is private, and the gate is not a separate module that
can be lifted out, so this file describes it instead of shipping code.

## The rule

A task is one user command. The agent runs it as a loop of steps (model call,
parse actions, execute, look again). Any part of that loop can record a failure
reason on the task: an action that raised, an unknown action name, the model
stalling, an HTTP error from the model proxy, a send that could not be
verified. When the loop ends, the agent decides:

```python
task_succeeded = not self._task_failed_reason
obs.task_finished(task_succeeded, self._task_failed_reason, len(action_log))
# prints:  [task] verdict=ok steps=N
#     or:  [task] verdict=failed steps=N reason=<first 80 characters>
```

The same boolean drives what the user sees: the task card shows "Done!" or the
failure reason, and the widget is told `success=True/False`. So the verdict
line is the agent's own report of whether the task worked. This is the
`gate_decision` column (1 = ok, 0 = failed).

## What the gate does not check

The verdict is "nothing raised a failure", not "the result is correct". A
turn that runs cleanly but produces the wrong thing, or that politely declines
an impossible request, ends with `verdict=ok`. A turn that delivered the right
content and then failed on a later step ends with `verdict=failed`.

## Failure reasons seen in the 31 scored trials

From `results/results.json` (`reasons_on_false_negatives`,
`reasons_on_true_negatives`):

- `ai_http_413`: the request to the model proxy exceeded its size limit. In
  this run it came from a very large teaching prompt. It occurred both on
  tasks whose content had already been delivered (false negatives) and on
  tasks that never produced their output (true negatives).
- `read_clipboard raised: unknown action 'read_clipboard'`: the model emitted
  an action name that does not exist.
- `I got stuck waiting for something that wasn't happening...`: the stall
  detector stopped a task that kept waiting without acting.

## The "logged claim" is a different thing

The research log (`turns.jsonl`) has an `outcome` field that the
pre-registered analysis used as the agent's claim. The logging call after a
normal exit passes the constant `outcome="success"`; only a crash writes
`"error"`. It never reads `_task_failed_reason`. That is why the logged claim
is 1 on all 31 rows even where the verdict line says `failed`.

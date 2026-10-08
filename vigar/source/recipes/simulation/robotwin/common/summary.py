"""Single-task evaluation summaries shared by policy adapters."""

import json
import time


def write_summary(output, args, records, errors, started):
    finished = [r for r in records if r["status"] == "evaluated"]
    successes = sum(r["success"] for r in finished)
    result = dict(
        task=args.task,
        task_config=args.task_config,
        requested=args.episodes,
        evaluated=len(finished),
        successes=successes,
        success_rate=successes / len(finished) if finished else None,
        complete=len(finished) == args.episodes and not errors,
        rejected_expert_seeds=sum(r["status"] == "expert_rejected" for r in records),
        errors=errors,
        seconds=time.monotonic() - started,
        records=records,
    )
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    return result

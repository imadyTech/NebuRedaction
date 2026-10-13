"""Teacher labelling (NR-0003 §2.2).

Laya answers a seven-class `choice` question per unit (plan §4.4); CredSweeper's
ML score is a second opinion (§2.2 item 2). The run is batched, checkpointed and
resumable, because at NR-0000's measured CPU throughput it is an overnight job
and EXECUTION.md §4 requires one to survive interruption.
"""

VERSION = "0.1.0"

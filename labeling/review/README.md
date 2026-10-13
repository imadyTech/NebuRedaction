# Reviewing the label queue

One page, as NR-0003 §2.4 item 4 requires. **Frank does this review.** Budget
one to two hours; the queue is capped at 500 units.

## Start

```bash
uv run --group eval python -m labeling.review
```

That is all. The queue has already been built, and the tool picks up wherever
you left off — stop any time with `q` and re-run the same command.

## What you are deciding

Each unit is one log attribute. The question is **what kind of value is this**,
in seven classes. You are shown everything the three systems think:

```
[12/500]  synth-train-004217   source=synth   family=generic.oauth_client_secret
reason: synthetic:teacher_generator_disagree   (DISAGREEMENT)
------------------------------------------------------------------------------
  key      : client_secret
  value    : Hq8mZ2vKpT4wRn6yBx1cFj9sLd3gAe7u
  template : OAuth exchange for <*> completed
  scope    : otel.attribute
------------------------------------------------------------------------------
  generator: api_key
  rules    : none  (low, none)
  teacher  : none=0.712  api_key=0.139  token=0.061   entropy=1.482
------------------------------------------------------------------------------
  current  : api_key

  [Enter] confirm    [n] none  [p] password  [t] token  [a] api_key
                     [k] private_key  [s] connection_string  [c] cookie
  [v] full value   [u] undo   [s] skip   [q] save and quit
```

`current` is the label that stands if you just press **Enter**. For a synthetic
unit that is the generator's label, which is correct by construction unless the
generator itself has a bug. For an own-log unit there is no prior label, so it
is the teacher's choice.

## Keys

| key | does |
|---|---|
| **Enter** | confirm `current` — the common case |
| `n` `p` `t` `a` `k` `s` `c` | set the class directly |
| `v` | show the whole value (PEM blocks are truncated by default) |
| `u` | undo: go back to the previous unit and decide again |
| `s` | skip, if you genuinely cannot tell |
| `q` | save and quit; re-run to continue |

Every keystroke is written to `data/labels/review_v0.jsonl` immediately, so
nothing is lost if the terminal closes. An undo appends a new record rather than
erasing the old one, so your changes of mind stay auditable.

## What to look for, and the one thing to flag

Most of the queue is **disagreements between the teacher and the generator**,
sorted first, then by how confused the teacher is. Three things worth knowing:

1. **The teacher is not calibrated and is confidently wrong in a specific way.**
   It reads the message template. NR-0000 saw it label a user name as a password
   at 0.972 because the template said `Failed password for invalid user`; this
   run saw it label `OrderId = 29` a bearer token at 0.979. When the teacher
   disagrees with the generator and the template contains a credential word,
   the teacher is usually the one that is wrong.

2. **Judge the value, not the key name.** That is the project's whole subject.
   `PasswordPolicy = strict` is `none`. `PartitionKey = users-2026` is `none`.

3. **If you think the *generator* is wrong, say so in a note rather than just
   correcting the label.** A generator defect affects every unit of that family,
   not the one in front of you, so §2.3 requires fixing the generator code and
   recording the defect — correcting a single label would leave the rest wrong
   and hide the cause. The family name is on screen for this reason. (Noting is
   not yet wired to a key; tell me the unit id and I will record it.)

## Afterwards

```bash
uv run --group eval python -m labeling.review --summary   # what you decided
uv run --group eval python -m labeling.merge              # produce train/val
```

## Rebuilding the queue

Only needed if the teacher labels or the calibration change:

```bash
uv run --group eval python -m labeling.review --build
```

Rebuilding does not discard your decisions — they are keyed by unit id, and any
unit you have already decided is skipped.

## Why a terminal and not a web page

§2.4 item 1 allows either and asks the executor to explain. Reviewing 500 units
in two hours is eight to fourteen seconds each, which is a single-keystroke
task; a browser puts a mouse and a round trip into every decision. A localhost
server would also be an open port and a running process in a project whose
subject is credential leakage, and the values on this screen are the most
sensitive data the project handles. Resumption here is an append-only file read
back at startup, with no server state to reconcile. The one thing a page would
do better is render a multi-line PEM block, which is why `v` exists.

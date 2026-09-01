# HN Launch Kit — 2026-08-30

How to get 1bit.MONSTER onto Hacker News. HN does **not** crawl the web: a
human submits a URL, and the **title** + **first comment** decide everything
that happens next. This kit is the submission.

## The URL to submit

```
https://1bit.monster/1bit-post-npu-reversal.html
```

Why this page, not the homepage or the repo:
- HN front page rewards a **story**, not a product index. "I reverse-engineered
  AMD's NPU stack in 4 days" is the hook; the homepage is a card grid.
- The post now links the full 31k-word `docs/journey.md` (receipts) and
  today's HRX news (momentum) — HN readers dig one hop to evidence.
- The repo link is in the footer; HN rules discourage "here's my repo" as the
  main submission for this kind of story.

## Title options (submit exactly one; HN allows no "Show HN:" prefix for
## a link post unless it's a genuine Show HN — this is a story, so skip it)

Recommended (factual, curiosity, no hype — HN mods rewrite marketing titles):
```
I reverse-engineered AMD's NPU stack in 4 days
```

Alternates (pick one, don't submit multiple):
```
AMD shipped a 50 TOPS NPU nobody could use. We reverse-engineered it in 4 days
Reverse-engineering AMD's locked-down XDNA 2 NPU: 4 days, a disassembler, no docs
```

Avoid: "One engine any model zero Python" (marketing), "Huge news!!" (hype),
anything with "!" or "🚀". HN kills those titles and the post with them.

## The first comment (this is 50% of the outcome)

Post this as the submitter's own comment immediately after submitting. It is
the story HN actually reads. Keep it first-person, specific, and honest about
what's broken — that is the site's culture and it is why the "corrections"
section of the post exists.

```
Author here. TL;DR of the 4 days:

Day 1: catalogued the 22 proprietary .so files and 209 xclbin bitstreams of
AMD's FastFlowLM NPU runtime. Everything is opaque — firmware is RSA-2048
signed, bitstreams are undocumented.

Day 2: traced dispatch with dynamic_debug + ftrace + bpftrace until we could
drive the NPU's mailbox protocol by hand.

Day 3: first real GEMM through raw ioctls — bit-exact, outside FLM's binary.

Day 4: replaced the whole stack with open C++.

Since then: 100% of HuggingFace's arch-bearing checkpoints map to an engine
token (317,310 / 317,310), one C++26 binary, zero Python at runtime, MIT.

The part I'm most proud of is the honesty section of the post. We quarantined
unsourced throughput figures, retracted a wrong efficiency claim, and
disproved our own speculative-decode projection end-to-end (572 tok/s
projected → 0.1-0.2 tok/s measured). The full journal is public:
github.com/1bit-MONSTER/1bit-MONSTER/blob/main/docs/journey.md

Today it also runs AMD's experimental IREE runtime as an in-process decode
lane — two llama.cpps in one process, dlopen'd, ~2x HIP warm decode on a 30B
MoE GGUF: https://1bit.monster/1bit-post-hrx-engine.html

Happy to answer anything — especially "why would anyone use this" questions.
```

## Timing (US East)

- Best: Tue–Thu, 07:30–09:30 ET or 16:00–17:30 ET (HN's two engagement peaks)
- Never: Friday after 14:00 ET, weekends, or US holidays (post dies in the void)
- If it dies without comments in the first ~20 minutes: it is dead, delete and
  re-submit at the next window (HN allows one resubmit)

## What not to do

- Don't upvote-brigade / ask friends to upvote from same-IP (HN's ring
  detector bans the whole thing)
- Don't answer your own "why would anyone use this" with "because it's cool"
- Don't mention "we need upvotes" anywhere

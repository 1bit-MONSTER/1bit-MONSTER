# Traffic attribution — what a "clone spike" is made of

Written 2026-09-15 for [#2378](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/2378)
("Traffic spike: 3027 clones (392 unique) yesterday"), which asked for the source
of a GitHub clone spike and a README/docs update if one was warranted. The
answer is: **the clone series tracks Actions run volume almost one-for-one**,
so the spike was this repo's own CI, not outside interest. No README change is
warranted; this document is the durable record, and
`.github/workflows/traffic-alert.yml` now puts the numbers in the alert itself.

## Why the question is not answerable from the clone API

GitHub exposes clone counts and unique cloners per day
(`GET /repos/{owner}/{repo}/traffic/clones`), but **no source, referrer or user
agent for clones** — only for views (`/traffic/popular/referrers`,
`/traffic/popular/paths`). So a clone spike cannot be attributed directly. It can
be attributed *by comparison*: GitHub's own docs say the clone figures include
automated traffic, and this repo knows how much automation it generates per day.

## Method

`GET /repos/{owner}/{repo}/actions/runs?per_page=1&created=>=YYYY-MM-DDT00:00:00Z`
returns `total_count`, the number of workflow runs created on or after that
instant. The count for one day is the difference between consecutive days:

```
runs(day D) = total_count(created >= D) - total_count(created >= D+1day)
```

The endpoint is readable **unauthenticated** for a public repo (no token scope
needed, which matters because `traffic-alert.yml` deliberately avoids elevating
its token — see issue #159), so the workflow can compute this for free.

## Measured, 2026-08-31 → 2026-09-13 (fetched 2026-09-15 ~04:55Z)

| day | clones | unique | workflow runs | clones / run |
|---|---:|---:|---:|---:|
| 08-31 | 1801 | 357 | 405 | 4.4 |
| 09-01 | 861 | 109 | 294 | 2.9 |
| 09-02 | 1417 | 143 | 428 | 3.3 |
| 09-03 | 460 | 81 | 175 | 2.6 |
| 09-04 | 944 | 124 | 255 | 3.7 |
| 09-05 | 1019 | 132 | 394 | 2.6 |
| 09-06 | 578 | 112 | 115 | 5.0 |
| 09-07 | 434 | 122 | 91 | 4.8 |
| 09-08 | 284 | 85 | 137 | 2.1 |
| 09-09 | 550 | 89 | 222 | 2.5 |
| 09-10 | 1025 | 223 | 290 | 3.5 |
| 09-11 | 2277 | 244 | **758** | 3.0 |
| 09-12 | **3027** | 392 | **776** | 3.9 |
| 09-13 | **3063** | 516 | **670** | 4.6 |

Totals over the window: 17,740 clones, 5,010 workflow runs — **3.54 clones per
run**. Pearson correlation between the two series is **r = 0.948**; the least
squares fit is `clones ≈ 3.84 × runs − 108`.

The spike day is not an outlier against that line. What moved on
2026-09-11..13 is the **run count** (758 / 776 / 670 against a 91–428
baseline). Grouping the 2026-09-12 runs by workflow shows what that volume is
made of — CI 190, Scope Guard 138, CodeQL 135, PR Agent 134, Release 52,
Post SEO 41, Deploy Site 41 — a burst of PR/push activity and its release/SEO
tail, against a 2026-09-08 baseline of 137 runs (PR Agent 81, CI 12, Scope
Guard 11, CodeQL 10). Clones followed because every workflow that checks the
repo out clones it.

Views are a separate series and are **not** CI-driven: over the same period the
top referrer was `1bit.monster` (242 views, 6 uniques) and `github.com`
(235 views, 25 uniques), with the repo root / pulls / issues / actions pages as
the top paths. That is site traffic, and it is what a human-interest spike would
look like.

## Reading the next spike

- **Clone spike with a proportional run count** → CI volume. Nothing to do;
  this is the common case and the reason the threshold is a trailing average.
- **Clone spike with a flat run count** (clones/run well above the ~3.5
  baseline) → something outside is cloning repeatedly; start with the top
  referrers and paths for the same window, and note that a full clone of this
  repo is large, so a crawler hitting it repeatedly is plausible.
- **Views/referrers spike with flat clones** → human interest, e.g. a launch
  post. That is the series to watch for the README/docs question the alert was
  originally written around.

## What the alert now does

`traffic-alert.yml` computes the run count for the same day as the clone figure
and puts both the CI number and the clones-per-run ratio into the spike issue
body and the job summary (falling back to the previous message if the public
API is unavailable). A future issue of this shape therefore arrives with its
most likely explanation already on it.

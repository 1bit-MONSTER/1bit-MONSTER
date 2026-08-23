# Fluxer community ops

Idempotent scripts for managing the official `1bit.MONSTER` Fluxer community
(`1540334785874886656`, invite `https://fluxer.gg/USDRI0qz`).

They read all config from the environment or `./.env`. **Commit nothing secret** —
`.env` is gitignored; set `FLUXER_TOKEN`, `FLUXER_TOKEN_TYPE`, `FLUXER_API_BASE_URL`,
and `FLUXER_HOME_GUILD_ID` locally. These scripts never need the token in git.

## Scripts

`add-support-channel.mjs`
: Create the `#support` channel (idempotent) under the Text Channels category and
  seed a one-time welcome message. `--fresh` recreates the channel.

```bash
node add-support-channel.mjs            # create + seed (no-op if present)
node add-support-channel.mjs --fresh    # delete + recreate
```

`theme-roles.mjs`
: Apply the site brand palette to the guild roles — Admin `#1779e1`, Moderator
  `#008048`, everyone `#0e1217`. Dry-run by default.

```bash
node theme-roles.mjs                    # dry run (prints what it would do)
node theme-roles.mjs --apply            # write the colors
```

Both reuse the same `.env` precedence as `fluxer-mcp` (env var, then `./.env`),
so the token is never passed on the command line or committed to git.

# Mempack Deep Test Report

Date: January 22, 2026
Repo: wordtolatex-server

## Scope
This document captures the full set of mempack operations executed and the findings about retrieval behavior and CLI/MCP usage.

## Environment
- MCP server configured in `/Users/sujendragharat/.codex/config.toml`:
  - command: `mem`
  - args: `mcp --allow-write --write-mode ask`
  - enabled tools: `mempack.get_context`, `mempack.explain`, `mempack.add_memory`, `mempack.checkpoint`
- CLI binary located at: `/Users/sujendragharat/go/bin/mem`

## Operations Performed
### MCP operations
- `mempack.get_context` used with multiple queries
- `mempack.add_memory` (confirmed) created a test memory
- `mempack.checkpoint` (confirmed) created a checkpoint state
- `mempack.explain` used to verify retrieval scoring
- `list_mcp_resources` and `list_mcp_resource_templates` returned empty

### CLI operations
- `mem get`, `mem explain`, `mem threads`, `mem thread`, `mem show`
- `mem ingest-artifact` to add chunk(s) from `AGENTS.md`

## Records Created
### Thread
- `mempack-test-2026-01-22`

### Memory IDs
- `M-20260122-231553-04fa37ae` — title: "Mempack test entry"
- `M-20260122-231609-4faa9eb2` — title: "Checkpoint"

### Chunk ID
- `C-20260122-231957-64d6c405` from `AGENTS.md#L1-L10`

### Checkpoint State
```json
{"note":"checkpoint test","date":"2026-01-22","repo":"wordtolatex-server"}
```

## Findings
### 1) Retrieval is phrase-sensitive
- Queries that match exact phrases in the memory title/summary retrieve results.
- Natural-language or re-phrased queries often return no matches.

Examples (CLI `mem explain`):
- Works:
  - "mempack test entry"
  - "test entry"
  - "mempack test"
  - "retrieve"
  - "operations"
- Does not work:
  - "test retrieve"
  - "test retrieve operations"
  - "do a deeper test save something retreive and do all operations"

Observed implication: short, exact phrase queries are more reliable; misspellings (e.g., "retreive") and loosely related text reduce matches.

### 2) Query choice drives MCP `get_context` results
- `mempack.get_context` with the exact phrase "Mempack test entry" returns the created memory and checkpoint state.
- The same call with a broader request-style query returns no memories.

### 3) CLI flag ordering for `mem thread`
- `mem thread --limit 20 mempack-test-2026-01-22` works.
- `mem thread mempack-test-2026-01-22 --limit 20` fails with "thread not found".

### 4) Ingested chunks are retrievable by exact text
- `mem ingest-artifact --thread mempack-test-2026-01-22 AGENTS.md` added 1 chunk.
- Querying "Mempack Agent Policy" returns the chunk via both CLI and MCP.

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.get_context
# mempack.add_memory
# mempack.checkpoint
# mempack.explain

# CLI
mem get "Mempack test entry"
mem explain "Mempack test entry"
mem threads
mem thread --limit 20 mempack-test-2026-01-22
mem show M-20260122-231553-04fa37ae
mem ingest-artifact --thread mempack-test-2026-01-22 AGENTS.md
```

## Recommendations
- Use short exact phrases when querying mempack (title/summary substrings).
- When planning future searches, include those exact phrases in memory titles or summaries.
- Use `mem explain` to debug unexpected empty results.
- Keep `mem thread` flags before positional args to avoid CLI parsing errors.

## Cleanup (Optional)
If you want to remove the test data:
```bash
mem forget M-20260122-231553-04fa37ae
mem forget M-20260122-231609-4faa9eb2
```
Note: Chunk removal for ingested artifacts was not tested.

---

## Addendum: Comprehensive Mempack Test Sweep (January 25, 2026)
Repo: wordtolatex-server

## Scope
Comprehensive validation of Mempack.md claims across MCP, CLI, ingestion, workspaces, tokenization, and Git orphan filtering. Embeddings and clustering were checked for availability.

## Environment
- CLI: mempack v0.2.0 (dev)
- MCP: mem mcp status reports "not running" (Codex MCP tools available)
- Config: /Users/sujendragharat/.config/mempack/config.toml (no embedding_provider configured)
- Repo IDs:
  - p_4a1edff0 (wordtolatex-server)
  - r_841f36b8 (temp acceptance repo at /private/tmp/mempack_accept_20260125.d51oBv)

## Records Created
### Threads
- mempack-test-2026-01-25-cli
- mempack-test-2026-01-25-mcp
- mempack-test-2026-01-25-ingest
- mempack-accept-jan25 (temp repo)

### Memory IDs (CLI, wordtolatex-server)
- M-20260125-205724-03769864 - "CLI Tie A: Tiebreak Jan25 Phi"
- M-20260125-205731-7c40a1d9 - "CLI Tie B: Tiebreak Jan25 Phi"
- M-20260125-205735-de6394be - "CLI Summary Only"
- M-20260125-205742-4d90a065 - "CLI Case Token"
- M-20260125-205746-8e3d1d49 - "CLI Order A"
- M-20260125-205750-56f2318b - "CLI Order B"
- M-20260125-205755-264e6d96 - "CLI Punct"
- M-20260125-205802-6650e651 - "CLI Supersede A: Nova Jan25" (superseded)
- M-20260125-205806-9d5a9378 - "CLI Supersede B: Nova Jan25"

### Memory IDs (MCP, wordtolatex-server)
- M-20260125-210136-42d32d08 - "MCP Test A: Pulsar Jan25"

### Checkpoint Memory IDs (MCP)
- M-20260125-210143-72a7f345 - reason "Jan25 MCP checkpoint"
- State ID: S-20260125-210143-7065420c

### Memory IDs (Acceptance Repo r_841f36b8)
- M-20260125-210309-0aa9de89 - "Acceptance B: Delta 99"
- M-20260125-210403-1a8704a8 - "Acceptance A: Alpha 11"
- M-20260125-210300-781ecc75 - "Acceptance A: Delta 99" (forgotten)

### Chunk IDs (Representative)
- C-20260125-205948-2cda979b - file:mempack_ingest_test/include.txt#L1-L2
- C-20260125-205948-cba78912 - file:mempack_ingest_test/ignored.txt#L1-L2
- C-20260125-205948-a53a74fe - file:mempack_ingest_test/long.txt#L246-L285
- C-20260125-205948-e3aeae74 - file:mempack_ingest_test/long.txt#L211-L250

### Test Artifacts (Temp)
- /private/tmp/mempack_accept_20260125.d51oBv (git repo with commits A/B)
- /tmp/mempack_ingest_test_jan25 (ingestion outside repo)

## Findings
### 1) MCP output contract gaps
- `mempack.get_context` responses did not include `search_meta` (even with results).
- `format=prompt` returned the same JSON payload; no prompt string surfaced.
- `mempack.get_initial_context` tool is not exposed in this MCP configuration.
- `mempack.add_memory` requires `confirmed=true`; `confirmed=false` returns "write_mode=ask requires confirmed=true after user approval."

### 2) Tokenization and retrieval
- Recency breaks ties: "tiebreak-jan25-phi" returned B before A with a tiny recency bonus in `mempack.explain`.
- Summary-only and case-insensitive retrieval worked ("summary-only-jan25", "camelcasejan25").
- Word order appears to affect results: "orderflip-jan25 first second" vs "orderflip-jan25 second first" returned distinct memories.
- Punctuation handling: "kiwi 4201", "kiwi-4201", and "kiwi_4201" matched; "kiwi4201" returned no hits.

### 3) Workspaces and state
- `mem get --workspace other` returned memories but an empty state object (workspace isolation for memories not enforced).

### 4) Supersede
- `mem supersede` created a new memory and set `superseded_by` on the old entry; the superseded memory still appears but ranks after the newer one.

### 5) Ingestion behavior
- Outside-repo ingest returned 0 files/0 chunks.
- Repo-local ingest: files_ingested=3, chunks_added=11, files_skipped=6; re-ingest added 0 chunks.
- `.mempackignore` respected (ignore-02 not retrievable).
- Root `.gitignore` respected for `logs/` (ignore-logs-01 not retrievable).
- `.gitignore` inside the artifact was not respected (ignored.txt retrievable).
- `.log` file skipped (ignore-log-01 not retrievable).
- Duplicate chunks were not collapsed in output; `include-01` returned two chunk entries and no `sources[]` field.

### 6) Git orphan filtering + acceptance proof
- In temp repo r_841f36b8, commit B query "delta 99" returned the commit B memory.
- After checkout to commit A, "delta 99" returned no memories; `orphans_filtered=1`.
- `--include-orphans` returned the commit B memory as expected.
- `delta99` did not rewrite to "delta 99" (no results); rewrite behavior not observed.

### 7) CLI behavior
- `mem thread <id> --limit 20` now succeeds (previously failed in earlier report); `memory_count` still reports 0 in thread metadata.
- Missing `--summary` still errors: `mem add ...` returns "missing --summary".

### 8) Repo detection / anchoring
- `mem doctor --json` reports `has_git=false` for wordtolatex-server; newly created CLI memories have empty `anchor_commit`.
- In the temp repo, `anchor_commit` was populated and `mem get` showed `head`/`branch`.

### 9) Embeddings and clustering
- CLI has no `mem embed` command; clustering flags not exposed; vector/hybrid tests blocked.

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.get_context (budgeted + format=prompt)
# mempack.add_memory (confirmed true/false)
# mempack.checkpoint
# mempack.explain

# CLI (wordtolatex-server)
mem --version
mem doctor --json
mem mcp status
mem add --thread mempack-test-2026-01-25-cli --title "CLI Tie A: Tiebreak Jan25 Phi" --summary "tiebreak-jan25-phi"
mem supersede M-20260125-205802-6650e651 --title "CLI Supersede B: Nova Jan25" --summary "supersede-jan25-nova"
mem get "tiebreak-jan25-phi" --debug
mem get "kiwi-4201" --debug
mem get "kiwi4201" --debug
mem thread mempack-test-2026-01-25-cli --limit 20
mem thread --limit 20 mempack-test-2026-01-25-cli
mem ingest-artifact /tmp/mempack_ingest_test_jan25 --thread mempack-test-2026-01-25-ingest
mem ingest-artifact mempack_ingest_test --thread mempack-test-2026-01-25-ingest
mem get "include-01" --debug
mem get "ignore-02" --debug
mem get "ignore-01" --debug

# CLI (acceptance repo)
mem init --no-agents
mem add --thread mempack-accept-jan25 --title "Acceptance B: Delta 99" --summary "delta 99 acceptance B"
mem get "delta 99" --debug
mem get "delta 99" --include-orphans --debug
```

## Cleanup (Optional)
To remove the 2026-01-25 test data in wordtolatex-server:
```bash
mem forget M-20260125-205724-03769864
mem forget M-20260125-205731-7c40a1d9
mem forget M-20260125-205735-de6394be
mem forget M-20260125-205742-4d90a065
mem forget M-20260125-205746-8e3d1d49
mem forget M-20260125-205750-56f2318b
mem forget M-20260125-205755-264e6d96
mem forget M-20260125-205802-6650e651
mem forget M-20260125-205806-9d5a9378
mem forget M-20260125-210136-42d32d08
mem forget M-20260125-210143-72a7f345
```

To remove the acceptance repo memories:
```bash
mem forget M-20260125-210309-0aa9de89
mem forget M-20260125-210403-1a8704a8
```

To remove temp artifacts:
```bash
rm -rf /private/tmp/mempack_accept_20260125.d51oBv /tmp/mempack_ingest_test_jan25
```

---

## Addendum: Mempack Checklist Deep Test (January 24, 2026)
Repo: wordtolatex-server

## Scope
Full checklist coverage across MCP tooling, retrieval/ranking, ingestion, embeddings, state/health, and CLI/config.

## Records Created
### Threads
- `mempack-test-2026-01-24-checklist`
- `mempack-test-2026-01-24-cli`
- `mempack-test-2026-01-24-ingest` (chunks only)

### Memory IDs (MCP)
- `M-20260124-215508-65012aa6` — title: "Checklist MCP A: Orion Drift"
- `M-20260124-215512-6375669a` — title: "Checklist MCP B: Orion Drift"
- `M-20260124-215515-3c2acb2c` — title: "Checklist MCP C: Summary Token Only"
- `M-20260124-215522-c3965f74` — title: "Checklist MCP D: Alpha Beta"
- `M-20260124-215526-e8847563` — title: "Checklist MCP E: Beta Alpha"

### Checkpoint Memory IDs (MCP)
- `M-20260124-215531-bcf16dfe` — reason: "Checklist baseline state"
- `M-20260124-215536-0ecc216e` — reason: "Checklist invalid JSON state"
- `M-20260124-215618-86e69ca4` — reason: "Checklist restore state"

### Memory IDs (CLI)
- `M-20260124-231157-231661d3` — title: "CLI Supersede B: Nova Key"
- `M-20260124-231149-0e11dfa5` — title: "CLI Supersede A: Nova Key" (forgotten)

### Checkpoint Memory IDs (CLI)
- `M-20260124-231314-d253f81d` — reason: "CLI invalid json"
- `M-20260124-231344-240a8a54` — reason: "CLI state file json"
- `M-20260124-231453-0038a474` — reason: "CLI state file md"
- `M-20260124-231521-e8f9eb7e` — reason: "Checklist final state"

### State IDs (Representative)
- `S-20260124-231521-565abb84` — state: "checklist-final"

### Chunk IDs (Representative)
- `C-20260124-215958-840124e4` — locator: "file:mempack_ingest_test/include.txt#L1-L2"
- `C-20260124-215958-1875909d` — locator: "file:mempack_ingest_test/ignored.txt#L1-L2"
- `C-20260124-215958-3f0a11ae` — locator: "file:mempack_ingest_test/long.txt#L211-L250"
- `C-20260124-215958-75dfa417` — locator: "file:mempack_ingest_test/long.txt#L246-L285"

### Test Artifacts (Repo)
- `mempack_ingest_test/` (test fixtures for ingestion)

### Test Artifacts (Temp)
- `/tmp/mempack_ingest_test/` (ingestion outside repo returned 0 ingested)
- `/tmp/mempack_state.json`
- `/tmp/mempack_state.md`

## Findings
### 1) MCP tooling and error paths
- `mempack.add_memory` with `confirmed=false` returns "write_mode=ask requires confirmed=true".
- `confirmed=true` succeeded, indicating allow-write is enabled.
- `mempack.checkpoint` accepted invalid JSON strings without error and stored `state.raw` until a valid JSON checkpoint replaced it.
- Missing-param errors are not reachable via MCP tool schema; CLI covers this case.
- `format=prompt` still returns JSON (no prompt-format output observed).

### 2) Retrieval & ranking
- "orion drift" returns B before A; `mempack.explain` shows identical BM25/FTS with tiny recency bonus.
- Natural-language query "please retrieve orion drift baseline" returns no memories (no query rewrite observed).
- Budget truncation: `budget=20` returned no memories; `budget=80` truncated "mcp" to 2 results (previously 6 at higher budget).
- Supersede down-ranking works: superseded memory has `superseded=true` and final_score ~2.6 vs ~7.6; `link_trail` stayed empty.
- `--include-orphans` produced identical results to default (no orphans observed).
- `mempack.explain` surfaces BM25/FTS only; no embedding/hybrid fields present.

### 3) Ingestion behavior
- `mem ingest-artifact /tmp/mempack_ingest_test` (outside repo) ingested 0 files.
- Repo-local `mempack_ingest_test` ingest: `files_ingested: 3`, `chunks_added: 11`, `files_skipped: 4`.
- `.mempackignore` respected (ignored.mempack not retrievable); `.gitignore` inside artifact dir not respected (ignored.txt ingested).
- Root `.gitignore` pattern `logs/` appears respected (logs/skip.txt not retrievable).
- `.bin` file skipped (bin-01 not retrievable); size limits not hit in this run.
- Chunking shows overlap and locator format: `file:<path>#Lx-Ly` (e.g., L211-L250 and L246-L285).
- Dedupe works: re-ingest added 0 chunks.

### 4) Embeddings
- No embedding provider config found in `~/.config/mempack/config.toml` and no CLI commands for embed/backfill/worker; hybrid scoring and min-similarity filters were not testable in this build.

### 5) State & health
- `mem init --no-agents` completed successfully (no AGENTS changes requested).
- `mem checkpoint --state-file` with JSON yields parsed state; Markdown yields `state.raw`.
- Invalid JSON accepted by both MCP and CLI checkpoints.
- `mem doctor` reports `schema` user_version 8 vs current_version 4; FTS ok; `--repair` made no changes.
- `mem mcp status` reports not running.
- State fallback from repo files was not exercised (no reset/clear path used).

### 6) CLI & config
- `mem --version` reports `mempack v0.2.0 (dev)`.
- `mem repos` lists two repos; `mem use` successfully switched to `p_2feaaf42` and back.
- `mem get --workspace other` returned memories but empty state (workspace isolation not enforced for memories).
- `mem thread` still reports `memory_count: 0` despite listing memories.
- `mem show` includes `superseded_by`; `mem forget` works as expected.
- CLI error path confirmed: `mem add --thread ... --title ...` fails with "missing --summary".

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.add_memory
# mempack.checkpoint
# mempack.get_context (budgeted + prompt format)
# mempack.explain

# CLI
mem --version
mem repos
mem use p_2feaaf42
mem use p_4a1edff0
mem init --no-agents
mem doctor --json
mem doctor --repair --verbose
mem mcp status
mem threads
mem thread mempack-test-2026-01-24-checklist --limit 20
mem add --thread mempack-test-2026-01-24-cli --title "CLI Supersede A: Nova Key" --summary "..."
mem supersede M-20260124-231149-0e11dfa5 --title "CLI Supersede B: Nova Key" --summary "..."
mem explain "nova key"
mem forget M-20260124-231149-0e11dfa5
mem ingest-artifact mempack_ingest_test --thread mempack-test-2026-01-24-ingest
mem get "include-01"
mem get "chunk-line-250" --debug
mem get "alpha beta" --include-orphans
mem get "orion drift" --workspace other
mem checkpoint --reason "CLI state file json" --state-file /tmp/mempack_state.json --thread mempack-test-2026-01-24-cli
mem checkpoint --reason "CLI state file md" --state-file /tmp/mempack_state.md --thread mempack-test-2026-01-24-cli
mem checkpoint --reason "Checklist final state" --state-json '{"note":"checklist final","date":"2026-01-24","repo":"wordtolatex-server","phase":"checklist-final"}' --thread mempack-test-2026-01-24-cli
```

## Cleanup (Optional)
If you want to remove the 2026-01-24 test data:
```bash
# MCP checklist memories
mem forget M-20260124-215508-65012aa6
mem forget M-20260124-215512-6375669a
mem forget M-20260124-215515-3c2acb2c
mem forget M-20260124-215522-c3965f74
mem forget M-20260124-215526-e8847563
mem forget M-20260124-215531-bcf16dfe
mem forget M-20260124-215536-0ecc216e
mem forget M-20260124-215618-86e69ca4

# CLI checklist memories
mem forget M-20260124-231157-231661d3
mem forget M-20260124-231314-d253f81d
mem forget M-20260124-231344-240a8a54
mem forget M-20260124-231453-0038a474
mem forget M-20260124-231521-e8f9eb7e
```
Note: `M-20260124-231149-0e11dfa5` was already forgotten. Chunk removal for ingested artifacts was not tested.

If you want to delete local test fixtures:
```bash
rm -rf mempack_ingest_test /tmp/mempack_ingest_test /tmp/mempack_state.json /tmp/mempack_state.md
```

---

## Addendum: Embeddings Test (Ollama) (January 24, 2026)
Repo: wordtolatex-server

## Scope
Ollama embeddings validation as a separate test: provider config, queue/backfill behavior, worker execution, and vector metadata visibility.

## Setup
- Ollama available at `/opt/homebrew/bin/ollama` with `nomic-embed-text:latest` present (`ollama list`).
- Config updated to:
  - `embedding_provider = "ollama"`
  - `embedding_model = "nomic-embed-text"`
  - `embedding_min_similarity = 0.6`
- Config backup created at:
  - `/Users/sujendragharat/.config/mempack/config.toml.bak-embeddings-20260124-183733`
- `OLLAMA_HOST` not set; default base URL used (`http://localhost:11434`).

## Records Created
### Thread
- `mempack-test-2026-01-24-embed`

### Memory IDs
- `M-20260124-233820-376c45e3` — title: "Embedding Test A: Lumen Orbit"
- `M-20260124-233836-2d28cb8b` — title: "Embedding Test B: Silver River"
- `M-20260124-233858-f45c159f` — title: "Embedding Test C: Redwood Trail"
- `M-20260124-234017-dcf08e45` — title: "Embedding Test MCP D: Aurora Drift"

## Findings
### 1) CLI embedding commands are missing in this build
- `mem embed --help` returns "unknown command: embed".
- No `mem embed status` or `mem embed --kind` available for backfill/coverage checks.

### 2) Embedding queue and table remain empty
- After adding 3 CLI memories and 1 MCP memory with `embedding_provider=ollama`, `embedding_queue` and `embeddings` counts remained `0`.
- Indicates `maybeEmbedMemory` is not invoked in this build, or embedding queueing is disabled.

### 3) Vector metadata not visible in explain output
- `mem explain "lumen orbit"` only returns BM25 fields; no vector ranks/scores or vector status block.

### 4) MCP daemon start did not stay running
- `mem mcp start` logged a read-only server start (`tools=2`) but no `mcp.pid` remained.
- `mem mcp stop` reported "not running"; no evidence that the embedding worker loop stayed active.

### 5) Embedding validation blocked by build limitations
- Without queueing/backfill and a persistent worker, vector search, min similarity filtering, and hybrid ranking could not be validated.

## Commands Used (Representative)
```bash
ollama list
mem mcp start
mem add --thread mempack-test-2026-01-24-embed --title "Embedding Test A: Lumen Orbit" --summary "..."
mem add --thread mempack-test-2026-01-24-embed --title "Embedding Test B: Silver River" --summary "..."
mem add --thread mempack-test-2026-01-24-embed --title "Embedding Test C: Redwood Trail" --summary "..."
mem explain "lumen orbit"
```

## Cleanup (Optional)
If you want to remove the embeddings test data:
```bash
mem forget M-20260124-233820-376c45e3
mem forget M-20260124-233836-2d28cb8b
mem forget M-20260124-233858-f45c159f
mem forget M-20260124-234017-dcf08e45
```

To revert embedding config:
```bash
cp /Users/sujendragharat/.config/mempack/config.toml.bak-embeddings-20260124-183733 /Users/sujendragharat/.config/mempack/config.toml
```

---

## Addendum: Embeddings Test (Ollama) Follow-up (CLI Backfill) (January 24, 2026)
Repo: wordtolatex-server

## Scope
Use newer `mem` CLI from the mempack repo (`/Users/sujendragharat/Library/CloudStorage/GoogleDrive-sgharat298@gmail.com/My Drive/MacExternalCloud/Documents/Projects/memory/mem`) to run `mem embed` and validate vector/hybrid retrieval.

## Records Created
- No new memories; embeddings table populated for existing data.

## Findings
### 1) Embed status available in newer CLI
- `mem embed status` reports provider `ollama`, model `nomic-embed-text`, enabled true.
- Before backfill: memories missing 110, chunks missing 13.
- After backfill: memories with embeddings 110/110, chunks 13/13.

### 2) Backfill succeeded via `mem embed --kind all`
- Completed in ~31s and reported: `Embedded memories=110 chunks=13`.

### 3) Vector mode and fallback metadata appear in `mem get`
- `mem get "lumen orbit" --debug` shows `search_meta.mode=vector`, `vector_used=true`, and `fallback_reason=bm25_empty` with warning `bm25_empty_vector_fallback`.

### 4) Explain now includes vector fields
- `mem explain "alpha echo"` shows BM25 + vector ranks/scores and RRF (hybrid output).
- `mem explain "orion drift"` returns vector-only results with `vector_score` and `vector_rank`.

### 5) Min-similarity appears not enforced for vector-only results
- `min_similarity` is 0.6, but vector-only results with scores below 0.6 (e.g., 0.56, 0.53) were still included.
- Indicates the threshold may not be applied to vector-only fallback in this build.

## Commands Used (Representative)
```bash
./mem embed status
./mem embed --kind all
./mem get "lumen orbit" --debug
./mem explain "lumen orbit"
./mem explain "alpha echo"
./mem explain "orion drift"
```

---

## Addendum: MCP Deep Test Follow-up (January 24, 2026)
Repo: wordtolatex-server

## Scope
Read-only MCP verification using existing memories (no new writes). Confirmed retrieval behavior for phrase matching, order sensitivity, punctuation tokenization, summary indexing, and state handling.

## Records Created
- None (read-only).

## Findings
### 1) Recency still breaks ties between identical matches
- Query "alpha echo" returns B before A; `mempack.explain` shows identical BM25/FTS with a tiny recency bonus difference.
- Query "zeta kappa mix" returns writepath B before A for the same reason.

### 2) Summary-only tokens and case-insensitive matches work
- "summary-only-77" and "camelcase99" return memory C (summary-only token + case-insensitive match).

### 3) Word order remains respected
- "first second" returns memory D; "second first" returns memory E.

### 4) Punctuation/number tokenization is permissive; separators required
- "delta 42", "delta-42", and "delta_42" match memory E.
- "delta42" returns no memories.

### 5) State always returns even when no memories match
- Queries "zz-nope" and "nonsense-zz" returned zero memories but included state: `phase` = "writepath-2".

### 6) Prompt format appears to return JSON; no rewrite behavior observed
- `format=prompt` for query "alpha echo" returned the standard JSON structure (no prompt-formatted text observed).
- Natural-language query "please retrieve alpha echo baseline" returned no memories, consistent with phrase sensitivity.

### 7) Budget parameter accepted; truncation not observed at 120 tokens
- `budget=120` set `target_total` to 120 with `used_total` 118 for query "mcp" and returned 6 memories (no explicit truncation signal).

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.get_context "alpha echo"
# mempack.get_context "summary-only-77"
# mempack.get_context "camelcase99"
# mempack.get_context "first second"
# mempack.get_context "second first"
# mempack.get_context "delta 42"
# mempack.get_context "delta-42"
# mempack.get_context "delta_42"
# mempack.get_context "delta42"
# mempack.get_context "zeta kappa mix"
# mempack.get_context "zz-nope"
# mempack.get_context "nonsense-zz"
# mempack.explain "alpha echo"
# mempack.explain "zeta kappa mix"
# mempack.get_context "alpha echo" --format prompt
# mempack.get_context "please retrieve alpha echo baseline"
# mempack.get_context "mcp" --budget 120
```

---

## Addendum: MCP Write-Path Deep Test (January 24, 2026)
Repo: wordtolatex-server

## Scope
MCP write-path validation for add/checkpoint flows plus retrieval behavior and CLI thread metadata.

## Records Created
### Thread
- `mempack-test-2026-01-23-writepath`

### Memory IDs
- `M-20260124-014557-e88622ac` — title: "MCP Write Test A: Zeta Kappa Mix"
- `M-20260124-014559-30513c87` — title: "MCP Write Test B: Zeta Kappa Mix"
- `M-20260124-014602-bc524507` — title: "MCP Write Test C: Summary Token Only"
- `M-20260124-014608-a94814d9` — title: "MCP Write Test D: One Two"
- `M-20260124-014610-ff9bc8d0` — title: "MCP Write Test E: Two One"

### Checkpoint Memory IDs
- `M-20260124-014605-412cbc6d` — reason: "Writepath baseline state"
- `M-20260124-014616-fb12a2f5` — reason: "Writepath updated state"

### State IDs
- `S-20260124-014605-fe9af404` — state: "writepath-1"
- `S-20260124-014616-949d917d` — state: "writepath-2"

### Checkpoint States
```json
{"note":"writepath baseline","date":"2026-01-23","repo":"wordtolatex-server","phase":"writepath-1"}
```
```json
{"note":"writepath updated","date":"2026-01-23","repo":"wordtolatex-server","phase":"writepath-2"}
```

## Findings
### 1) Recency breaks ties between identical matches
- Query "zeta kappa mix" returned B before A.
- `mempack.explain` showed identical BM25/FTS scores; recency bonus ordered B first.

### 2) Summary-only tokens are indexed and case-insensitive
- Querying "summary-only-88" and "camelcasetoken42" both returned memory C.

### 3) Word order is respected for phrase-like queries
- "one two" returned D; "two one" returned E.

### 4) Punctuation/number tokenization is permissive
- "delta 99", "delta-99", and "delta_99" all matched memory E.

### 5) Separatorless tokens do not match
- "delta99" returned no matches.

### 6) State always returns; latest checkpoint wins
- "zz-nope" returned no memories but included the latest state.

### 7) CLI `mem thread` memory_count looks stale
- `memory_count` reported `0` even when the returned `memories` list had entries.

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.add_memory (5 memories in thread mempack-test-2026-01-23-writepath)
# mempack.checkpoint (2 checkpoints)
# mempack.get_context "zeta kappa mix"
# mempack.get_context "summary-only-88"
# mempack.get_context "camelcasetoken42"
# mempack.get_context "one two"
# mempack.get_context "two one"
# mempack.get_context "delta 99"
# mempack.get_context "delta-99"
# mempack.get_context "delta_99"
# mempack.get_context "delta99"
# mempack.get_context "zz-nope"
# mempack.explain "zeta kappa mix"

# CLI
mem thread --limit 10 mempack-test-2026-01-23-writepath
```

## Cleanup (Optional)
If you want to remove the 2026-01-24 write-path test data:
```bash
mem forget M-20260124-014557-e88622ac
mem forget M-20260124-014559-30513c87
mem forget M-20260124-014602-bc524507
mem forget M-20260124-014608-a94814d9
mem forget M-20260124-014610-ff9bc8d0
mem forget M-20260124-014605-412cbc6d
mem forget M-20260124-014616-fb12a2f5
```

---

## Addendum: MCP Deep Test (January 23, 2026)
Repo: wordtolatex-server

## Scope
MCP-only validation for ranking ties, summary indexing, order sensitivity, punctuation tokenization, and state updates.

## Records Created
### Thread
- `mempack-test-2026-01-23-deep`

### Memory IDs
- `M-20260123-173514-54985952` — title: "MCP Deep Test A: Alpha Echo Density"
- `M-20260123-173516-81401f9f` — title: "MCP Deep Test B: Alpha Echo Baseline"
- `M-20260123-173519-d988b61e` — title: "MCP Deep Test C: Summary Token Only"
- `M-20260123-173522-6aab22d6` — title: "MCP Deep Test D: First Second"
- `M-20260123-173525-59a16d36` — title: "MCP Deep Test E: Second First"

### Checkpoint Memory IDs
- `M-20260123-173529-7241e456` — reason: "Deep MCP test baseline state"
- `M-20260123-173615-7d740097` — reason: "Deep MCP test updated state"

### State IDs
- `S-20260123-173529-8b63d199` — state: "deep-1"
- `S-20260123-173615-fe21a894` — state: "deep-2"

### Checkpoint States
```json
{"note":"mcp deep test baseline","date":"2026-01-23","repo":"wordtolatex-server","phase":"deep-1"}
```
```json
{"note":"mcp deep test updated","date":"2026-01-23","repo":"wordtolatex-server","phase":"deep-2"}
```

## Findings
### 1) Recency breaks ties between identical matches
- Query "alpha echo" returned both A and B, with B first.
- `mempack.explain` showed identical BM25/FTS scores; a slightly higher recency bonus ordered B before A.

### 2) Summary-only tokens are indexed and case-insensitive
- Querying "summary-only-77" (summary-only) and "camelcase99" (case-insensitive) both returned memory C.

### 3) Word order is respected for phrase-like queries
- "first second" returned D; "second first" returned E.

### 4) Punctuation/number tokenization is permissive
- "delta 42", "delta-42", and "delta_42" all matched memory E.

### 5) State always returns; latest checkpoint wins
- "nonsense-zz" returned no memories but included state after each checkpoint.
- The second checkpoint replaced the earlier state in subsequent calls.

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.add_memory (5 memories in thread mempack-test-2026-01-23-deep)
# mempack.checkpoint (2 checkpoints)
# mempack.get_context "alpha echo"
# mempack.get_context "summary-only-77"
# mempack.get_context "camelcase99"
# mempack.get_context "first second"
# mempack.get_context "second first"
# mempack.get_context "delta 42"
# mempack.get_context "delta-42"
# mempack.get_context "delta_42"
# mempack.get_context "nonsense-zz"
# mempack.explain "alpha echo"
# mempack.explain "delta 42"
```

## Cleanup (Optional)
If you want to remove the 2026-01-23 deep test data:
```bash
mem forget M-20260123-173514-54985952
mem forget M-20260123-173516-81401f9f
mem forget M-20260123-173519-d988b61e
mem forget M-20260123-173522-6aab22d6
mem forget M-20260123-173525-59a16d36
mem forget M-20260123-173529-7241e456
mem forget M-20260123-173615-7d740097
```

---

## Addendum: MCP Edge Case Test (January 23, 2026)
Repo: wordtolatex-server

## Scope
Focused MCP edge-case validation plus a few CLI behavior checks for regressions.

## Records Created
### Thread
- `mempack-test-2026-01-23`

### Memory IDs
- `M-20260123-162233-bc3b7c66` — title: "MCP Edge Test: Alpha-Beta_01"
- `M-20260123-162236-7f423604` — title: "Checkpoint"

### Chunk ID
- `C-20260123-162301-07495e09` from `AGENTS.md#L1-L10`

### Checkpoint State
```json
{"note":"mcp edge checkpoint","date":"2026-01-23","repo":"wordtolatex-server","case":"mixedCase"}
```

## Findings
### 1) Punctuation/case is tokenized permissively
- Queries like "alpha beta_01" and "alpha beta 01" match "Alpha-Beta_01".

### 2) Misspellings and natural-language queries still fail
- "alfa beta 01" returns no results.
- "please retrieve the edge test entry with alpha beta" returns no results.

### 3) Empty retrieval still returns state
- `mempack.get_context` with a nonsense phrase returns no memories/chunks, but includes the latest state.

### 4) Ingested chunks can surface duplicates across threads
- Querying "Mempack Agent Policy" returned both the 2026-01-23 and 2026-01-22 chunks.

### 5) CLI `mem thread` flag ordering no longer fails
- Both `mem thread --limit 5 <thread>` and `mem thread <thread> --limit 5` succeeded.

### 6) CLI `mem thread` memory_count looks stale
- `memory_count` reported `0` even when the returned `memories` list had entries.

### 7) Invalid thread still errors
- `mem thread --limit 5 mempack-test-2026-01-23-missing` exits non-zero with "thread not found".

## Commands Used (Representative)
```bash
# MCP (via tool calls)
# mempack.get_context
# mempack.add_memory
# mempack.checkpoint
# mempack.explain

# CLI
mem ingest-artifact --thread mempack-test-2026-01-23 AGENTS.md
mem thread --limit 5 mempack-test-2026-01-23
mem thread mempack-test-2026-01-23 --limit 5
mem get "alpha beta 01"
```

## Cleanup (Optional)
If you want to remove the 2026-01-23 test data:
```bash
mem forget M-20260123-162233-bc3b7c66
mem forget M-20260123-162236-7f423604
```
Note: Chunk removal for ingested artifacts was not tested.

---

## Addendum: Comprehensive Mempack Test Sweep (January 26, 2026)
Repo: wordtolatex-server

## Scope
Comprehensive validation of Mempack.md claims across MCP, CLI, ingestion, workspaces, tokenization, and Git orphan filtering using the new mem binary.

## Environment
- CLI binary: /Users/sujendragharat/Library/CloudStorage/GoogleDrive-sgharat298@gmail.com/My Drive/MacExternalCloud/Documents/Projects/memory/mem
- CLI version: mempack v0.2.0 (dev)
- CLI config override (sandbox-safe):
  - XDG_CONFIG_HOME=/Users/sujendragharat/Library/CloudStorage/GoogleDrive-sgharat298@gmail.com/My Drive/MacExternalCloud/Documents/Projects/wordtolatex-server/.config-mempack-jan26
  - MEMPACK_DATA_DIR=/Users/sujendragharat/Library/CloudStorage/GoogleDrive-sgharat298@gmail.com/My Drive/MacExternalCloud/Documents/Projects/wordtolatex-server/.mempack_data_jan26
- MCP (CLI status): `mem mcp status` reports "not running" (Codex MCP tools still available)
- Repo IDs:
  - p_4a1edff0 (wordtolatex-server)
  - r_18ddbf5a (temp acceptance repo at /private/tmp/mempack_accept_20260126.r6RpsI)
- Note: CLI tests used local config/data overrides due to sandbox write restrictions; MCP tool calls used the Codex MCP server and its state (later updated to the Jan 26 MCP checkpoint restart test).

## Records Created
### Threads
- mempack-test-2026-01-26-cli
- mempack-test-2026-01-26-mcp
- mempack-test-2026-01-26-mcp2
- mempack-test-2026-01-26-ingest
- mempack-accept-jan26 (temp repo)

### Memory IDs (CLI, wordtolatex-server)
- M-20260126-044132-9e1fffc0 - "CLI Tie A: Tiebreak Jan26 Phi"
- M-20260126-044133-b2800de5 - "CLI Tie B: Tiebreak Jan26 Phi"
- M-20260126-044133-7a91d590 - "CLI Summary Only"
- M-20260126-044133-12ebf4b0 - "CLI Case Token"
- M-20260126-044134-e4df3a33 - "CLI Order A"
- M-20260126-044134-8098cfc7 - "CLI Order B"
- M-20260126-044134-4365723d - "CLI Punct"
- M-20260126-044134-43332878 - "CLI Supersede A: Nova Jan26" (superseded)
- M-20260126-044134-05d29bd1 - "CLI Supersede B: Nova Jan26"

### Memory IDs (MCP, wordtolatex-server)
- M-20260126-044356-0fe8df25 - "MCP Test A: Pulsar Jan26"
- M-20260126-044919-89a9b5ba - "MCP Test B: Quasar Jan26"

### Checkpoint Memory IDs (MCP)
- M-20260126-044420-2e932961 - reason "Jan26 MCP checkpoint"
- State ID: S-20260126-044420-dc009331
- M-20260126-044939-98f43b09 - reason "Jan26 MCP checkpoint (restart test)"
- State ID: S-20260126-044939-b564e08b

### Memory IDs (Acceptance Repo r_18ddbf5a)
- M-20260126-044137-bb55aaa2 - "Acceptance A: Alpha 26"
- M-20260126-044137-fa028f9c - "Acceptance B: Delta 99"

### Chunk IDs (Representative)
- C-20260126-044136-9d8802e3 - file:mempack_ingest_test/include.txt#L1-L2
- C-20260126-044136-8f36563a - file:mempack_ingest_test/ignored.txt#L1-L2

### Test Artifacts (Temp / Local)
- /private/tmp/mempack_accept_20260126.r6RpsI (git repo with commits A/B)
- /tmp/mempack_ingest_test_jan26 (ingestion outside repo)
- .config-mempack-jan26 (local config override)
- .mempack_data_jan26 (local data override)
- .config-mempack-jan26-mcp (local MCP config override)
- .mempack_data_jan26_mcp (local MCP data override)
- /tmp/mempack_mcp_jan26.log (MCP server log)

## Findings
### 1) MCP output contract gaps (unchanged)
- `mempack.get_context` responses did not include `search_meta`.
- `format=prompt` returned the same JSON payload; no prompt string surfaced.
- `mempack.get_initial_context` tool is not exposed in this MCP configuration.
- `mempack.add_memory`/`mempack.checkpoint` require `confirmed=true`; `confirmed=false` returns "write_mode=ask requires confirmed=true after user approval."

### 2) Tokenization and retrieval (updated)
- Summary-only and case-insensitive retrieval worked ("summary-only-jan26", "camelcasejan26").
- Word order did **not** change results: both "orderflip-jan26 first second" and "orderflip-jan26 second first" returned the same two memories (A then B).
- Punctuation handling: "kiwi 4202", "kiwi-4202", and "kiwi_4202" matched; "kiwi4202" rewrote to "kiwi 4202" and matched.
- Rewrite metadata now surfaces in `search_meta` (e.g., `rewrites_applied` for "delta99" and "kiwi4202").

### 3) Tie/recency scoring
- "tiebreak-jan26-phi" returned A before B. `mempack.explain` showed identical BM25/FTS with a tiny recency bonus for B, but A retained a higher final score due to FTS rank.

### 4) Workspaces
- `mem get --workspace other` returned **no memories** for "summary-only-jan26" (workspace isolation appears enforced). State was empty.

### 5) Supersede
- `mem supersede` created a new memory and set `superseded_by` on the old entry; the superseded memory still appears after the new one in thread listing.

### 6) Ingestion behavior
- Outside-repo ingest returned 0 files/0 chunks.
- Repo-local ingest: `files_ingested=3`, `chunks_added=11`, `files_skipped=6`; re-ingest added 0 chunks.
- `.mempackignore` respected ("ignore-02" not retrievable).
- Root `.gitignore` respected ("ignore-logs-01" not retrievable).
- `.gitignore` inside the artifact was not respected (ignored.txt retrievable).
- `.log` file skipped ("ignore-log-01" not retrievable).
- Duplicate chunk output **collapsed** with `sources[]` populated ("include-01" returned one chunk with sources).

### 7) Git orphan filtering + acceptance proof
- In temp repo r_18ddbf5a, commit B query "delta 99" returned the commit B memory.
- After checkout to commit A, "delta 99" returned no memories; debug showed `orphans_filtered=1`.
- `--include-orphans` returned the commit B memory as expected.
- `delta99` rewrote to "delta 99" and matched (rewrite observed in `search_meta`).

### 8) CLI behavior
- Both `mem thread <id> --limit 20` and `mem thread --limit 20 <id>` succeeded; `memory_count` reported 9.
- Missing `--summary` still errors: `mem add ...` returns "missing --summary".
- CLI `mem get --format prompt` returned a prompt-style response string.

### 9) Repo detection / anchoring
- `mem doctor --json` reports `has_git=false` for wordtolatex-server; newly created CLI memories have empty `anchor_commit`.
- In the temp repo, `anchor_commit` was populated and `mem get` reported `head`/`branch`.

### 10) Embeddings and clustering
- `mem embed status` exists and reports provider `auto` (effective `ollama`), but embeddings are unavailable (Ollama not reachable). No clustering tests were run.

### 11) Sandbox/config note
- CLI tests ran with repo-local config/data overrides to avoid sandbox write restrictions; MCP tool calls used the Codex MCP server and its state (updated to the Jan 26 MCP checkpoint restart test).

### 12) MCP restart test (new binary)
- Started `mem mcp` with the new binary using repo-local config/data overrides and repo ID `p_4a1edff0`.
- MCP tool calls still returned JSON-only payloads for `format=prompt` and omitted `search_meta`.
- MCP state now reflects the latest checkpoint: `note="jan26 mcp checkpoint restart test"` (January 26, 2026).
- Could not verify that Codex MCP tool calls were routed to the new server (no Codex MCP config change possible in sandbox), so results reflect the configured MCP server.

## Commands Used (Representative)
```bash
# CLI (new binary + local config/data overrides)
MEM="/Users/sujendragharat/Library/CloudStorage/GoogleDrive-sgharat298@gmail.com/My Drive/MacExternalCloud/Documents/Projects/memory/mem"
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM init --no-agents
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM add --thread mempack-test-2026-01-26-cli --title "CLI Tie A: Tiebreak Jan26 Phi" --summary "tiebreak-jan26-phi"
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM supersede M-20260126-044134-43332878 --title "CLI Supersede B: Nova Jan26" --summary "supersede-jan26-nova"
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM get "tiebreak-jan26-phi" --debug
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM get "kiwi4202" --debug
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM ingest-artifact mempack_ingest_test --thread mempack-test-2026-01-26-ingest
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM get "include-01" --debug
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" $MEM embed status

# MCP server (new binary, local overrides)
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26-mcp" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26_mcp" $MEM init --no-agents
nohup env XDG_CONFIG_HOME="$PWD/.config-mempack-jan26-mcp" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26_mcp" $MEM mcp --repo p_4a1edff0 --allow-write --write-mode ask > /tmp/mempack_mcp_jan26.log 2>&1 &
# (best-effort) kill <pid>

# MCP (via tool calls)
# mempack.get_context (baseline + format=prompt)
# mempack.add_memory (confirmed true/false)
# mempack.checkpoint (confirmed true/false)
# mempack.explain
```

## Cleanup (Optional)
To remove the 2026-01-26 test data (using the local config/data overrides):
```bash
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044132-9e1fffc0
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044133-b2800de5
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044133-7a91d590
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044133-12ebf4b0
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044134-e4df3a33
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044134-8098cfc7
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044134-4365723d
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044134-43332878
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044134-05d29bd1
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044356-0fe8df25
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044919-89a9b5ba
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044420-2e932961
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044939-98f43b09
```

To remove the acceptance repo memories (same overrides):
```bash
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044137-bb55aaa2
XDG_CONFIG_HOME="$PWD/.config-mempack-jan26" MEMPACK_DATA_DIR="$PWD/.mempack_data_jan26" mem forget M-20260126-044137-fa028f9c
```

To remove temp artifacts and local overrides:
```bash
rm -rf /private/tmp/mempack_accept_20260126.r6RpsI /tmp/mempack_ingest_test_jan26
rm -rf .config-mempack-jan26 .mempack_data_jan26 .config-mempack-jan26-mcp .mempack_data_jan26_mcp /tmp/mempack_mcp_jan26.log
```

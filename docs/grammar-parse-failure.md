# `failed to parse grammar` — tool schemas that break the backend

Claude Code gets this back through the middleware:

```
✻ 502 LM Studio returned 400: {"error":{"code":400,"message":
"Failed to initialize samplers: failed to parse grammar","type":"invalid_request_error"}}
```

**This is not an llm-search bug.** No request ever reaches the model, and
nothing in `anthropic_adapter.py` / `tool_loop.py` is at fault. The failure
is inside the llama.cpp-family backend when it tries to compile the
grammar it derives from the *tool schemas the client sent*. Any client
that sends a tool whose JSON Schema contains a `pattern` can trigger it.

This document exists because the symptom points at the middleware (the
error surfaces as a `502 LM Studio returned 400`, i.e. llm-search merely
relaying), and because diagnosing it requires reading the backend's log,
which has a Windows-specific trap (see [Reading the log](#reading-the-log)).

- [Background](#background)
- [Root cause](#root-cause)
- [Why it only triggers on some tools](#why-it-only-triggers-on-some-tools)
- [Which schemas trigger it](#which-schemas-trigger-it)
- [Diagnosing](#diagnosing)
- [The fix](#the-fix)
- [Verification](#verification)

## Background

When a request carries `tools` and `tool_choice` is `auto`, llama.cpp does
not just pass the schemas to the model — it *compiles them into a GBNF
grammar* that constrains decoding, so the model physically cannot emit a
malformed tool call. The chain:

| Step | Location |
|---|---|
| Mark the grammar lazy because tools are present | `common/chat.cpp:1094` |
| Convert JSON Schema → GBNF | `common/json-schema-to-grammar.cpp`, `common/peg-parser.cpp` |
| Parse the GBNF | `src/llama-grammar.cpp` (`llama_grammar_parser::parse`) |
| Return `nullptr` on parse failure | `src/llama-grammar.cpp:1222` |
| Throw | `common/sampling.cpp:275` — `throw std::runtime_error("failed to parse grammar")` |
| Wrap for HTTP | `tools/server/server-context.cpp:2127` — `"Failed to initialize samplers: " + e.what()` |

Because grammar construction happens at *sampler init*, the request fails
before prefill. A grammar error is total: one bad character class in one
tool takes down the entire request, not just that tool.

## Root cause

The schema→GBNF converter and the GBNF parser disagree about escaping
inside character classes.

Given this `pattern` (from Claude Code's `ArtifactData` tool, on
`writes[].collection` / `writes[].doc_id`):

```json
"pattern": "^(?!\\.\\.?(?:\\/|$))[A-Za-z0-9_\\-\\.~:@+]{1,200}$"
```

…the converter emits the character class **verbatim**:

```
tool-ArtifactData-arg-writes-schema-item-doc-id-1 ::= [A-Za-z0-9_\-\.~:@+]
```

But `parse_char()` in `src/llama-grammar.cpp:162` accepts only this escape
set — everything else throws `unknown escape at …`:

```
\x  \u  \U  \t  \r  \n  \\  \"  \[  \]  \-   (and . / on this fork — see below)
```

So `\-` and `\.` are fatal. The error message names whichever comes first:

```
parse: error parsing grammar: unknown escape at \-.~:@+]
```

The escapes are *correct regex*, not garbage. `\-` between `_` and `.`
is precisely what stops `_-.` from being read as a range, so dropping the
backslash is not a valid fix.

This is an upstream llama.cpp bug, not a TurboQuant regression. Upstream
`master` accepts `\-` but still rejects `\.` and `\/`. The fork this stack
runs (`C:\code\llama-cpp-turboquant`) was additionally missing `\-`.

## Why it only triggers on some tools

The pattern is only converted when the property sits somewhere the
converter descends into. A **top-level** parameter is treated as "raw" and
delegated to the built-in `xml-arg-string` rule — its `pattern` is
discarded entirely and can never break anything.

That is why an isolated test of the same pattern passes while the real
request fails, and why the bug looks arbitrary:

```
top-level  doc_id  {pattern: ...}   → delegated to xml-arg-string → 200 OK
nested     writes[].doc_id          → converted to a char class   → 400
```

Any tool with a patterned string *nested* in an object/array item is a
candidate.

## Which schemas trigger it

Anything whose `pattern` uses a redundant escape inside a character class
— `\-`, `\.`, `\/` being the common ones. In the captured traffic, the
only offender was Claude Code's `ArtifactData` tool, which is sent on
essentially every Claude Code request, so it broke *all* of them.

Note that `responses_adapter.py:_sanitize_params()` defaults
`type`/`properties`/`additionalProperties` but does **not** strip
`pattern`, and `anthropic_adapter.py:_anthropic_tools_to_openai()` passes
`input_schema` through raw. So both the Anthropic and Responses paths are
exposed. There is no middleware-side sanitization for this today.

## Diagnosing

### Reading the log

The backend logs the full generated grammar on failure. On Windows,
PowerShell's `>` redirection writes **UTF-16LE**, so `grep` finds nothing
and it looks like the log is empty:

```bash
iconv -f UTF-16LE -t UTF-8 server.err.log > server.err.utf8.log
```

### Finding the offending rule

```bash
# every distinct character class the converter emitted
grep -oE '::= \[[^]]*\]' server.err.utf8.log | sort -u

# the actual parse error and the class that caused it
grep -n "unknown escape" server.err.utf8.log
```

A class containing a backslash that GBNF does not accept is the culprit.
The rule name points straight at the tool and property —
`tool-ArtifactData-arg-writes-schema-item-doc-id-1` is `ArtifactData` →
`writes[]` → `doc_id`.

### Minimal reproduction

Send a tool with the pattern **nested**, not at the top level:

```python
import json, urllib.request

params = {"type":"object","properties":{"writes":{"type":"array","items":{
    "type":"object","properties":{"doc_id":{
        "type":"string","pattern":r"^[a-z\-]+$"}}}}}}
tool = {"type":"function","function":{"name":"T","description":"d","parameters":params}}
payload = {"model":"x","messages":[{"role":"user","content":"hi"}],
           "max_tokens":1,"tools":[tool]}

req = urllib.request.Request("http://127.0.0.1:3080/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
try:
    urllib.request.urlopen(req, timeout=180); print("200")
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode())   # 400 failed to parse grammar
```

`^[a-z\-]+$` fails; `^[a-z-]+$` (hyphen last, no escape) passes. That
one-character difference is the whole bug.

## The fix

In `src/llama-grammar.cpp`, `parse_char()`, add the redundant escapes to
the self-escaping group:

```cpp
case '\\':
case '"':
case '[':
case ']':
// Regex metacharacters escaped inside a JSON-schema character class,
// e.g. [A-Za-z0-9_\-\.~:@+] as emitted for Claude Code's ArtifactData
// tool. The schema->GBNF converter passes such classes through
// verbatim, so these redundant-but-valid escapes must be accepted
// here or the entire grammar fails to parse.
case '-':
case '.':
case '/':
          return std::make_pair(src[1], src + 2);
```

Each is returned as the plain character, so `\-` stays a literal hyphen
rather than becoming a range — the semantics the pattern author intended
are preserved.

> **Local divergence:** only `case '-'` matches upstream. The `'.'` and
> `'/'` cases are additions this fork needs (upstream still rejects `\.`
> and `\/`). Carry this patch forward when rebasing the fork.

Rebuild — `llama-grammar.cpp` links into `llama.dll`, so the server must
be stopped first or the link fails with `LNK1104: cannot open file
'llama.dll'`:

```powershell
# stop the running llama-server, then:
cd C:\code\llama-cpp-turboquant
cmake --build build --config Release --target llama-server
```

(With the Ninja generator this is the same; with the Visual Studio
generator do **not** pass `-- /m` from Git Bash — MSYS rewrites `/m` into
a path and MSBuild dies with `MSB1008: Only one project can be specified`.)

### Rejected alternatives

- **Strip `pattern` in the middleware.** Works, but silently discards a
  constraint the client asked for, and leaves every other client of the
  backend broken. It also can't be done naively: rewriting `\-` → `-`
  turns a literal hyphen into a range and changes the pattern's meaning.
- **Rewrite escapes to hex form (`\-` → `\x2D`) in the middleware.**
  Semantically safe, since `\x` *is* accepted — but it's string surgery on
  arbitrary user patterns, and it only fixes the paths that go through the
  middleware.

Fixing the parser accepts what the generator emits, once, for every client.

## Verification

Against a nested `writes[]` schema, before and after the patch:

| Pattern | Before | After |
|---|---|---|
| `^(?!\.\.?(?:\/|$))[A-Za-z0-9_\-\.~:@+]{1,200}$` (real `ArtifactData`) | 400 | 200 |
| `^[a-z\-]+$` | 400 | 200 |
| `^a\/b$` | 400 | 200 |
| `^[a-z-]+$` | 200 | 200 (no regression) |
| `^[a-z]+$` | 200 | 200 (no regression) |

The before/after rows were measured with the repro above against a live
server; the real-world `ArtifactData` row was reproduced with the exact
schema Claude Code sends. Confirmed end-to-end by running Claude Code
through the middleware on the patched build.

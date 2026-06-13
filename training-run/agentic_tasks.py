"""Agentic training tasks — Hermes/OpenClaw capability coverage."""

from __future__ import annotations

import itertools
import random

# Hand-authored seeds across capability areas.
_SEEDS: list[dict] = [
    # file_ops
    {
        "category": "file_ops",
        "prompt": "Read config.json and notes.txt. Write summary.txt containing the project name from config and the word count of notes.",
        "workspace": {"files": {
            "config.json": '{"project": "benchmax-agent", "version": 2}',
            "notes.txt": "alpha beta gamma delta",
        }},
        "enabled_tools": ["read_file", "write_file", "list_files", "exec"],
        "success": {"type": "file_contains", "path": "summary.txt", "substring": "benchmax-agent"},
        "min_tool_calls": 2,
    },
    {
        "category": "file_ops",
        "prompt": "Patch bug.py: replace 'return a - b' with 'return a + b'.",
        "workspace": {"files": {"bug.py": "def add(a, b):\n    return a - b\n"}},
        "enabled_tools": ["read_file", "patch_file"],
        "success": {"type": "file_contains", "path": "bug.py", "substring": "return a + b"},
        "min_tool_calls": 1,
    },
    # research
    {
        "category": "research",
        "prompt": "Who founded the company described in the indexed pages? Answer after </think>.",
        "web_pages": {
            "https://docs.example.com/about": "Acme Corp was founded by Jane Rivera in 2019.",
            "https://docs.example.com/products": "Acme sells developer tools.",
        },
        "enabled_tools": ["web_search", "web_fetch"],
        "success": {"type": "answer_contains", "substring": "Jane Rivera"},
        "min_tool_calls": 1,
    },
    # math
    {
        "category": "math",
        "prompt": "Compute (17 * 23) + (144 / 12) using tools. Give the numeric result after </think>.",
        "enabled_tools": ["calculate", "run_python"],
        "success": {"type": "answer_contains", "substring": "403"},
        "min_tool_calls": 1,
    },
    # memory
    {
        "category": "memory",
        "prompt": "Store the code 'ALPHA-7742' in memory under key 'deploy_token'. Recall it and put it in your final answer.",
        "enabled_tools": ["memory_store", "memory_recall"],
        "success": {"type": "answer_contains", "substring": "ALPHA-7742"},
        "min_tool_calls": 2,
    },
    # skills
    {
        "category": "skills",
        "prompt": "Read the debugging skill and follow it to find the bad line in app.log, then report the line in your final report.",
        "workspace": {"files": {"app.log": "INFO start\nERROR null pointer at line 42\nINFO end"}},
        "skills": {"debugging": "When analyzing logs, grep for ERROR and report the first error line verbatim."},
        "enabled_tools": ["read_skill", "read_file", "exec"],
        "success": {"type": "answer_contains", "substring": "line 42"},
        "min_tool_calls": 2,
    },
    # messaging
    {
        "category": "messaging",
        "prompt": "Send the deployment summary 'build 88 passed' to channel #releases.",
        "enabled_tools": ["send_message"],
        "success": {"type": "message_sent", "channel": "#releases", "substring": "build 88 passed"},
        "min_tool_calls": 1,
    },
    # planning
    {
        "category": "planning",
        "prompt": (
            "Create todos for: (1) read data.csv (2) write report.txt. Mark both done after "
            "you complete them. data.csv contains 'sales=100'."
        ),
        "workspace": {"files": {"data.csv": "sales=100"}},
        "enabled_tools": ["todo_write", "todo_list", "read_file", "write_file"],
        "success": {"type": "todos_done"},
        "min_tool_calls": 2,
    },
    # browser
    {
        "category": "browser",
        "prompt": "Use browser_snapshot to read the page and report the headline in your final report.",
        "browser_url": "https://news.example.com",
        "web_pages": {"https://news.example.com": "HEADLINE: Quantum CPUs ship in Q3"},
        "enabled_tools": ["browser_snapshot"],
        "success": {"type": "answer_contains", "substring": "Quantum CPUs"},
        "min_tool_calls": 1,
    },
    # sessions
    {
        "category": "sessions",
        "prompt": "List sessions and report how many messages are in session 'support-1'.",
        "workspace": {"sessions": {"support-1": ["hi", "help"], "support-2": ["ping"]}},
        "enabled_tools": ["sessions_list"],
        "success": {"type": "answer_contains", "substring": "2"},
        "min_tool_calls": 1,
    },
    # multi_tool
    {
        "category": "multi_tool",
        "prompt": (
            "Search the web for 'pricing', fetch the pricing page, store the cheapest plan "
            "price in memory as 'cheapest', then answer with that price."
        ),
        "web_pages": {
            "https://acme.com/pricing": "Basic $9, Pro $29, Enterprise $99",
        },
        "enabled_tools": ["web_search", "web_fetch", "memory_store", "memory_recall"],
        "success": {"type": "answer_contains", "substring": "9"},
        "min_tool_calls": 3,
    },
    # openclaw_style
    {
        "category": "openclaw",
        "prompt": (
            "You have workspace files only — no priors. Read README.md and follow its "
            "instruction for the final answer."
        ),
        "workspace": {"files": {
            "README.md": "Instruction: the answer is 'gateway-ok'. Put that after </think>.",
        }},
        "enabled_tools": ["read_file"],
        "success": {"type": "answer_contains", "substring": "gateway-ok"},
        "min_tool_calls": 1,
    },
    # hermes_style
    {
        "category": "hermes",
        "prompt": (
            "Research task: use web_search then web_fetch to determine the API rate limit "
            "documented for the service."
        ),
        "web_pages": {
            "https://api.example.com/docs": "Rate limit: 120 requests per minute per key.",
        },
        "enabled_tools": ["web_search", "web_fetch"],
        "success": {"type": "answer_contains", "substring": "120"},
        "min_tool_calls": 2,
    },
]


def _expand_variants(seed: int) -> list[dict]:
    """Generate category variants from templates."""
    rng = random.Random(seed)
    out: list[dict] = []

    for _ in range(8):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        out.append({
            "category": "math",
            "prompt": f"Use calculate to compute {a} * {b} + {a - b}. Give your final answer after </think>.",
            "enabled_tools": ["calculate"],
            "success": {"type": "answer_contains", "substring": str(a * b + (a - b))},
            "min_tool_calls": 1,
        })
    for _ in range(12):
        a, b = rng.randint(5, 50), rng.randint(5, 50)
        expr = f"({a} + {b}) * 2"
        val = (a + b) * 2
        out.append({
            "category": "math",
            "prompt": f"Use run_python to evaluate: {expr}. Put the result after </think>.",
            "enabled_tools": ["run_python", "calculate"],
            "success": {"type": "answer_contains", "substring": str(val)},
            "min_tool_calls": 1,
        })

    names = ["apollo", "nova", "helix", "orbit", "pulse", "vertex", "cipher", "flux"]
    for name in names:
        out.append({
            "category": "file_ops",
            "prompt": f"Write {name}/status.txt with content 'ready'.",
            "enabled_tools": ["write_file", "list_files"],
            "success": {"type": "file_contains", "path": f"{name}/status.txt", "substring": "ready"},
            "min_tool_calls": 1,
        })
    for i in range(10):
        fname = f"module_{i}.py"
        out.append({
            "category": "file_ops",
            "prompt": (
                f"List all files, then read {fname} if it exists and summarize first line in your final report."
            ),
            "workspace": {"files": {fname: f"# module {i}\nprint({i})"}},
            "enabled_tools": ["list_files", "read_file", "exec"],
            "success": {"type": "answer_contains", "substring": f"module {i}"},
            "min_tool_calls": 2,
        })

    topics = ["kubernetes", "postgres", "redis", "graphql", "oauth", "webhooks"]
    for topic in topics:
        out.append({
            "category": "research",
            "prompt": f"What does the docs say about {topic}? Answer in one sentence after </think>.",
            "web_pages": {f"https://docs.example.com/{topic}": f"{topic}: production-ready feature."},
            "enabled_tools": ["web_search", "web_fetch"],
            "success": {"type": "answer_contains", "substring": topic},
            "min_tool_calls": 1,
        })
    extra_topics = ["grpc", "kafka", "terraform", "prometheus", "nginx", "tls", "jwt", "sqlite"]
    for topic in extra_topics:
        out.append({
            "category": "research",
            "prompt": f"Search for '{topic} limits' and report the numeric limit from docs.",
            "web_pages": {
                f"https://docs.example.com/{topic}": f"{topic} limits: max {rng.randint(10, 999)} units."
            },
            "enabled_tools": ["web_search", "web_fetch"],
            "success": {"type": "answer_contains", "substring": "units"},
            "min_tool_calls": 1,
        })

    for n in range(6):
        token = f"TOK-{1000 + n}"
        out.append({
            "category": "memory",
            "prompt": f"Store '{token}' under key 'auth' and recall it in your answer.",
            "enabled_tools": ["memory_store", "memory_recall"],
            "success": {"type": "memory_has", "key": "auth", "value": token},
            "min_tool_calls": 2,
        })

    channels = ["#general", "#alerts", "#deploys", "#incidents"]
    for ch in channels:
        out.append({
            "category": "messaging",
            "prompt": f"Notify {ch} with text 'ping from agent'.",
            "enabled_tools": ["send_message"],
            "success": {"type": "message_sent", "channel": ch, "substring": "ping from agent"},
            "min_tool_calls": 1,
        })

    skill_names = ["deploy", "incident-response", "code-review", "testing", "security-audit"]
    for sk in skill_names:
        out.append({
            "category": "skills",
            "prompt": f"Read the '{sk}' skill and follow its final instruction in your final report.",
            "skills": {sk: f"Skill {sk}: final instruction — respond with '{sk}-ok' in your final report."},
            "enabled_tools": ["read_skill"],
            "success": {"type": "answer_contains", "substring": f"{sk}-ok"},
            "min_tool_calls": 1,
        })

    for _ in range(8):
        out.append({
            "category": "planning",
            "prompt": (
                "Create 2 todos (step-a, step-b), mark step-a done after writing done.txt with 'a', "
                "then mark step-b done after writing b.txt with 'b'."
            ),
            "enabled_tools": ["todo_write", "write_file", "todo_list"],
            "success": {"type": "file_contains", "path": "b.txt", "substring": "b"},
            "min_tool_calls": 2,
        })

    for i in range(5):
        url = f"https://app{i}.example.com"
        out.append({
            "category": "browser",
            "prompt": "Snapshot the browser page and extract the status word in your final report.",
            "browser_url": url,
            "web_pages": {url: f"STATUS: online-{i}"},
            "enabled_tools": ["browser_snapshot"],
            "success": {"type": "answer_contains", "substring": f"online-{i}"},
            "min_tool_calls": 1,
        })

    for i in range(5):
        sid = f"thread-{i}"
        out.append({
            "category": "sessions",
            "prompt": "List sessions and report the name of the session with the most messages.",
            "workspace": {"sessions": {sid: ["m"] * (i + 2), f"other-{i}": ["x"]}},
            "enabled_tools": ["sessions_list"],
            "success": {"type": "answer_contains", "substring": sid},
            "min_tool_calls": 1,
        })

    for i in range(6):
        out.append({
            "category": "multi_tool",
            "prompt": (
                "Read input.txt, store the value in memory as 'v', recall it, write output.txt, "
                "answer with the value."
            ),
            "workspace": {"files": {"input.txt": f"value-{i}"}},
            "enabled_tools": ["read_file", "memory_store", "memory_recall", "write_file"],
            "success": {"type": "file_contains", "path": "output.txt", "substring": f"value-{i}"},
            "min_tool_calls": 3,
        })

    for i in range(5):
        out.append({
            "category": "openclaw",
            "prompt": "Use exec to cat policy.txt and follow the ONLY instruction in that file.",
            "workspace": {"files": {"policy.txt": f"ONLY: answer with claw-{i}"}},
            "enabled_tools": ["exec", "read_file"],
            "success": {"type": "answer_contains", "substring": f"claw-{i}"},
            "min_tool_calls": 1,
        })

    for _ in range(5):
        out.append({
            "category": "hermes",
            "prompt": (
                "Multi-step research: search 'tier', fetch doc, calculate tier*10, answer with result."
            ),
            "web_pages": {"https://api.example.com/tier": "tier=7"},
            "enabled_tools": ["web_search", "web_fetch", "calculate"],
            "success": {"type": "answer_contains", "substring": "70"},
            "min_tool_calls": 2,
        })

    return out


def build_all_tasks(seed: int = 42) -> list[dict]:
    tasks = [dict(t) for t in _SEEDS]
    tasks.extend(_expand_variants(seed))
    # Deduplicate by prompt hash
    seen: set[str] = set()
    unique: list[dict] = []
    for t in tasks:
        key = t["prompt"]
        if key in seen:
            continue
        seen.add(key)
        t.setdefault("requires_tools", True)
        t.setdefault("max_tool_calls", 12)
        unique.append(t)
    return unique


AGENTIC_TASKS: list[dict] = build_all_tasks()

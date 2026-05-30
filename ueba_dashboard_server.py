#!/usr/bin/env python3
"""
ueba_dashboard_server.py
─────────────────────────
CyberSentinel UEBA — Dashboard API Server

Serves ueba_alerts.jsonl data as JSON endpoints for the dashboard.

Run:
    python3 ueba_dashboard_server.py
    python3 ueba_dashboard_server.py --config ueba_config.yaml
    python3 ueba_dashboard_server.py --port 3026 --alerts /path/to/ueba_alerts.jsonl

Configuration is sourced (in priority order) from:
    1. CLI flags
    2. The `dashboard:` block in ueba_config.yaml
    3. Built-in defaults (matches the historical hard-coded values)

The Anthropic API key for the AI Security Analyst is taken from the
ANTHROPIC_API_KEY environment variable. If unset, the dashboard falls back to
a locally-generated summary.
"""

import argparse
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # config file is optional

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # AI proxy will degrade gracefully


# ── Defaults (used when config file or keys are missing) ──────────────────────
_DEFAULTS = {
    "host":            "0.0.0.0",
    "port":            3026,
    "alerts_file":     "/root/NEW_DRIVE/aditya_ueba/ueba_alerts.jsonl",
    "agents_registry": "/root/NEW_DRIVE/aditya_ueba/agents.json",
    "false_positives_file": "/root/NEW_DRIVE/aditya_ueba/false_positives.json",
    "fp_patterns_file":     "/root/NEW_DRIVE/aditya_ueba/fp_patterns.json",
    "max_alerts":      10000,
    "cache_ttl_secs":  10,
    "ai_analyst": {
        "enabled":     True,
        "model":       "claude-sonnet-4-6",
        "max_tokens":  1000,
        "timeout_secs": 20,
    },
    # Agent auto-discovery — periodically reads the Wazuh manager's
    # `client.keys` to keep agents.json in sync with the SOC inventory.
    #
    # Reading the bind-mounted client.keys requires NO permission changes
    # on the SOC server: the file is plain-text and world-readable by the
    # `soc` user. If you ever want to switch to `agent_control -l`
    # (richer: includes connection status), set remote_command to
    # `docker exec cybersentinel-manager /var/ossec/bin/agent_control -l`
    # and grant docker access. The poller auto-detects the format.
    "agent_sync": {
        "enabled":            True,
        "poll_interval_secs": 60,
        "ssh_host":           "soc@localhost",
        "ssh_port":           2222,
        "ssh_key":            "/root/.ssh/id_ueba",
        "remote_command":     "cat /var/ossec/etc/client.keys",
        "ssh_timeout_secs":   20,
    },
}


def load_dashboard_config(config_path: str | None) -> dict:
    """Load the `dashboard:` block from ueba_config.yaml, layered on defaults."""
    cfg = dict(_DEFAULTS)
    cfg["ai_analyst"] = dict(_DEFAULTS["ai_analyst"])
    if not config_path or not yaml:
        return cfg
    p = Path(config_path)
    if not p.exists():
        return cfg
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[dashboard] failed to read {config_path}: {e}")
        return cfg
    dash = (raw.get("dashboard") or {}) if isinstance(raw, dict) else {}
    for k, v in dash.items():
        if k == "ai_analyst" and isinstance(v, dict):
            cfg["ai_analyst"].update(v)
        elif k == "agent_sync" and isinstance(v, dict):
            cfg.setdefault("agent_sync", dict(_DEFAULTS["agent_sync"]))
            cfg["agent_sync"].update(v)
        else:
            cfg[k] = v
    return cfg


_SERVER_STARTED_AT = datetime.now(timezone.utc).isoformat()


# ── App + runtime config (populated in main()) ────────────────────────────────
# The dashboard is served from one of two places, in priority order:
#   1. dashboard/dist/         — built by `cd dashboard && npm run build`
#   2. dashboard/index.html    — legacy single-file fallback
# The decision is per-request so flipping between builds doesn't need a restart.
DASHBOARD_DIR     = Path(__file__).resolve().parent / "dashboard"
DASHBOARD_DIST    = DASHBOARD_DIR / "dist"
DASHBOARD_LEGACY  = DASHBOARD_DIR / "index.html"

app = Flask(__name__, static_folder=None)
CORS(app)

CFG: dict = dict(_DEFAULTS)
CFG["ai_analyst"] = dict(_DEFAULTS["ai_analyst"])
ALERTS_FILE: Path     = Path(_DEFAULTS["alerts_file"])
AGENTS_REGISTRY: Path = Path(_DEFAULTS["agents_registry"])
FP_FILE: Path         = Path(_DEFAULTS["false_positives_file"])
FP_PATTERNS_FILE: Path = Path(_DEFAULTS["fp_patterns_file"])

# ── Alert cache — avoids reading the full file on every API request ──
_alert_cache: list = []
_alert_cache_mtime: float = 0.0

# ── False-positive store — in-memory dict, persisted to FP_FILE atomically ──
# Maps event_id -> {"event_id": ..., "reason": "...", "marked_at": "..."}
_fp_lock = __import__("threading").Lock()
_fp_dict: dict = {}

# ── False-positive pattern store ─────────────────────────────────────────────
# Each pattern auto-suppresses any future alert whose rule description
# (security.signature) matches `rule_description`. Comparison is whitespace-
# trimmed and case-insensitive so analysts don't have to worry about minor
# string normalisation differences between the SIEM source and the alert
# stream. Persisted to FP_PATTERNS_FILE atomically.
_fp_pat_lock = __import__("threading").Lock()
_fp_patterns: list = []  # list of dicts: {id, rule_description, reason, marked_at}


def _norm_desc(s) -> str:
    """Normalise a rule description for matching: trim + lowercase."""
    return (str(s or "")).strip().lower()


def _parse_iso(ts) -> datetime | None:
    """Parse an ISO-8601 timestamp; return None if missing/unparseable."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None

# ── Agent sync state (populated by the background poller) ────────────────────
_agent_sync_last_at: str = ""           # ISO timestamp of last successful poll
_agent_sync_last_error: str = ""        # last error message (for /api/health)
_agent_sync_lock = __import__("threading").Lock()


def _parse_client_keys(stdout: str) -> list[dict]:
    """Parse a Wazuh `client.keys` file into [{id, name, ip}].

    Format is one agent per line: `<id> <name> <ip|"any"> <key>`. Lines
    starting with `#` or `!` (revoked) are skipped. IDs `000` / `0` are
    the manager itself and are skipped too.
    """
    out: list[dict] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        agent_id, name, ip = parts[0], parts[1], parts[2]
        if agent_id in ("000", "0"):
            continue
        # `any` is Wazuh's placeholder when no fixed IP was configured.
        if ip.lower() == "any":
            ip = ""
        else:
            ip = ip.split("/")[0]   # drop /netmask if present
        out.append({
            "id":     agent_id,
            "name":   name,
            "ip":     ip,
            "status": "",            # client.keys doesn't carry status
        })
    return out


def _parse_agent_control_l(stdout: str) -> list[dict]:
    """Parse the output of `agent_control -l` into a list of {id, name, ip} dicts.

    Wazuh's typical output format:

        Wazuh agent_control. List of available agents:
           ID: 000, Name: cybersentinel-manager (server), IP: 127.0.0.1, Active/Local
           ID: 001, Name: Sparta1, IP: 10.200.11.106/any, Active
           ID: 002, Name: KAFKA_ME_P, IP: 10.200.10.174, Disconnected
        ...

    Server-self entry (ID 000) is skipped — it isn't a real endpoint.
    """
    import re as _re
    line_re = _re.compile(
        r"ID:\s*(?P<id>\S+)\s*,\s*"
        r"Name:\s*(?P<name>.+?)(?:\s*\([^)]+\))?\s*,\s*"
        r"IP:\s*(?P<ip>[^,\s]+)\s*,\s*"
        r"(?P<status>.+?)\s*$"
    )
    out: list[dict] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or not line.lstrip().startswith("ID:"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        agent_id = m.group("id").strip()
        if agent_id in ("000", "0"):
            continue                                # skip the manager itself
        ip = m.group("ip").split("/")[0]            # strip /any or /netmask
        out.append({
            "id":     agent_id,
            "name":   m.group("name").strip(),
            "ip":     ip,
            "status": m.group("status").strip().lower(),
        })
    return out


def _fetch_agents_from_soc() -> tuple[list[dict] | None, str]:
    """SSH to the SOC server and run `agent_control -l`, parse the result.

    Returns (agents, error_message). On success error_message is "".
    On failure agents is None and error_message describes the problem.
    """
    cfg = CFG.get("agent_sync") or _DEFAULTS["agent_sync"]
    cmd = [
        "ssh",
        "-p", str(cfg["ssh_port"]),
        "-i", cfg["ssh_key"],
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(cfg['ssh_timeout_secs'])}",
        cfg["ssh_host"],
        cfg["remote_command"],
    ]
    try:
        import subprocess
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=int(cfg["ssh_timeout_secs"]) + 5,
        )
    except Exception as e:
        return None, f"ssh failed: {e}"
    if r.returncode != 0:
        # Surface stderr (first line) so the operator can fix permissions.
        err = (r.stderr or "").strip().splitlines()[:1]
        return None, f"remote command rc={r.returncode}: {' / '.join(err)}"
    # Auto-detect the output format. client.keys lines start with the
    # agent id followed by a space; agent_control lines start with "ID:".
    stdout = r.stdout
    agents = _parse_client_keys(stdout)
    if not agents:
        agents = _parse_agent_control_l(stdout)
    if not agents:
        return None, "remote command returned no parseable agents"
    return agents, ""


def _merge_agents(fresh: list[dict], existing_path: Path) -> list[dict]:
    """Merge `fresh` (from agent_control) with existing agents.json entries.

    Preserves fields the CLI doesn't return — most importantly `os` — by
    keying on agent id. New agents picked up from Wazuh just won't have
    those fields until the operator fills them in (or we extend the
    poller to call `agent_control -i <id>` for OS info).
    """
    by_id: dict = {}
    if existing_path.exists():
        try:
            for rec in json.loads(existing_path.read_text(encoding="utf-8") or "[]"):
                if isinstance(rec, dict) and rec.get("id"):
                    by_id[str(rec["id"])] = dict(rec)
        except Exception:
            pass
    fresh_ids: set = set()
    out: list[dict] = []
    for a in fresh:
        aid = str(a["id"])
        fresh_ids.add(aid)
        merged = dict(by_id.get(aid, {}))
        # Always trust the manager for id + name. For ip/status, only
        # overwrite when fresh has a real value — client.keys can return
        # empty strings (e.g. when an agent's IP is configured as "any"
        # or status isn't available from the file).
        merged["id"]   = aid
        merged["name"] = a["name"]
        if a.get("ip"):
            merged["ip"] = a["ip"]
        elif "ip" not in merged:
            merged["ip"] = ""
        if a.get("status"):
            merged["status"] = a["status"]
        out.append(merged)
    # Sort by numeric id for stable ordering.
    out.sort(key=lambda r: (int(r["id"]) if str(r["id"]).isdigit() else 9999, r["id"]))
    return out


def _write_agents_atomically(path: Path, agents: list[dict]) -> None:
    """Write agents.json with a tmp+rename so a reader never sees a half file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(agents, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _agent_sync_poll() -> None:
    """One round of the agent sync — runs in the background thread."""
    global _agent_sync_last_at, _agent_sync_last_error
    fresh, err = _fetch_agents_from_soc()
    with _agent_sync_lock:
        if err:
            _agent_sync_last_error = err
            print(f"[agent-sync] failed: {err}", flush=True)
            return
        try:
            merged = _merge_agents(fresh, AGENTS_REGISTRY)
            _write_agents_atomically(AGENTS_REGISTRY, merged)
            _agent_sync_last_at    = datetime.now(timezone.utc).isoformat()
            _agent_sync_last_error = ""
            print(f"[agent-sync] wrote {len(merged)} agents to {AGENTS_REGISTRY}", flush=True)
        except Exception as e:
            _agent_sync_last_error = f"write failed: {e}"
            print(f"[agent-sync] write failed: {e}", flush=True)


def _start_agent_sync_thread() -> None:
    """Spawn the agent-sync background thread (daemon, won't block shutdown)."""
    import threading
    cfg = CFG.get("agent_sync") or _DEFAULTS["agent_sync"]
    if not cfg.get("enabled"):
        print("[agent-sync] disabled in config")
        return
    interval = max(15, int(cfg.get("poll_interval_secs", 60)))

    def _loop():
        # Do one fetch immediately so first request after startup is fresh.
        _agent_sync_poll()
        while True:
            time.sleep(interval)
            _agent_sync_poll()

    t = threading.Thread(target=_loop, name="agent-sync", daemon=True)
    t.start()
    print(f"[agent-sync] polling every {interval}s — `{cfg['remote_command']}`")


def load_fps() -> None:
    """Load FPs from disk into _fp_dict. Safe to call multiple times."""
    global _fp_dict
    if not FP_FILE.exists():
        _fp_dict = {}
        return
    try:
        data = json.loads(FP_FILE.read_text(encoding="utf-8") or "[]")
        if isinstance(data, list):
            _fp_dict = {r["event_id"]: r for r in data if isinstance(r, dict) and r.get("event_id")}
        else:
            _fp_dict = {}
    except Exception as e:
        print(f"[dashboard] failed to load FP file {FP_FILE}: {e}")
        _fp_dict = {}


def save_fps() -> None:
    """Atomically write _fp_dict to FP_FILE."""
    tmp = FP_FILE.with_suffix(FP_FILE.suffix + ".tmp")
    try:
        FP_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(list(_fp_dict.values()), indent=2), encoding="utf-8")
        tmp.replace(FP_FILE)
    except Exception as e:
        print(f"[dashboard] failed to persist FPs to {FP_FILE}: {e}")


def load_fp_patterns() -> None:
    """Load FP patterns from disk into _fp_patterns.

    Accepts both the new shape (`rule_description`) and the legacy shape
    (`signature_id` + optional `user`/`agent`). Legacy entries are dropped
    with a warning since they can't be migrated without alert context — the
    analyst will need to re-mark a representative alert FP to recreate them.
    """
    global _fp_patterns
    if not FP_PATTERNS_FILE.exists():
        _fp_patterns = []
        return
    try:
        data = json.loads(FP_PATTERNS_FILE.read_text(encoding="utf-8") or "[]")
        if isinstance(data, list):
            kept, dropped = [], 0
            for p in data:
                if isinstance(p, dict) and (p.get("rule_description") or "").strip():
                    kept.append(p)
                elif isinstance(p, dict) and p.get("signature_id"):
                    dropped += 1
            _fp_patterns = kept
            if dropped:
                print(f"[dashboard] dropped {dropped} legacy sig-based FP pattern(s) — re-mark to recreate")
        else:
            _fp_patterns = []
    except Exception as e:
        print(f"[dashboard] failed to load FP patterns {FP_PATTERNS_FILE}: {e}")
        _fp_patterns = []


def save_fp_patterns() -> None:
    """Atomically write _fp_patterns to FP_PATTERNS_FILE."""
    tmp = FP_PATTERNS_FILE.with_suffix(FP_PATTERNS_FILE.suffix + ".tmp")
    try:
        FP_PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(_fp_patterns, indent=2), encoding="utf-8")
        tmp.replace(FP_PATTERNS_FILE)
    except Exception as e:
        print(f"[dashboard] failed to persist FP patterns to {FP_PATTERNS_FILE}: {e}")


def _alert_matches_pattern(alert: dict, pattern: dict) -> bool:
    """Return True iff alert's rule description matches the pattern's
    rule_description (trim + case-insensitive) AND the alert arrived after
    the pattern was created. The time guard keeps historical alerts visible
    — patterns only suppress NEW alerts going forward.
    """
    sec = alert.get("security", {}) or {}
    if _norm_desc(sec.get("signature")) != _norm_desc(pattern.get("rule_description")):
        return False
    # Time guard: pattern.marked_at must exist; alert must have been *emitted
    # by the engine* strictly later. Compare against ueba.processed_at (engine
    # time) rather than event_time (source time) — a backlog replay can produce
    # alerts whose event_time is days old but whose processed_at is "now".
    p_at = _parse_iso(pattern.get("marked_at"))
    ueba = alert.get("ueba", {}) or {}
    a_at = _parse_iso(ueba.get("processed_at") or alert.get("event_time"))
    if p_at is None or a_at is None:
        return False
    return a_at > p_at


def _matching_pattern(alert: dict):
    """Return the first FP pattern matching this alert, or None."""
    if not _fp_patterns:
        return None
    for p in _fp_patterns:
        if _alert_matches_pattern(alert, p):
            return p
    return None


def _pattern_from_alert(alert: dict, reason: str) -> dict | None:
    """Build an FP pattern record from an alert's rule description.

    The pattern suppresses every future alert whose rule description (the
    human-readable rule name, `security.signature`) matches — regardless of
    user, host, or even signature_id. This is the broad "this kind of activity
    is benign here" semantic.

    Returns None if the alert has no rule description.
    """
    sec = alert.get("security", {}) or {}
    desc = (str(sec.get("signature") or "")).strip()
    if not desc:
        return None
    return {
        "id":               "fpp-" + uuid.uuid4().hex[:10],
        "rule_description": desc,
        "reason":           reason or "auto-suppress from analyst FP mark",
        "marked_at":        datetime.now(timezone.utc).isoformat(),
    }


def _add_pattern_if_new(candidate: dict | None):
    """Append candidate to _fp_patterns unless an identical rule_description
    already exists. Returns (record, was_new). If candidate is None or has no
    rule_description, returns (None, False).
    """
    if not candidate or not (candidate.get("rule_description") or "").strip():
        return None, False
    cand_norm = _norm_desc(candidate["rule_description"])
    with _fp_pat_lock:
        for p in _fp_patterns:
            if _norm_desc(p.get("rule_description")) == cand_norm:
                return p, False
        _fp_patterns.append(candidate)
        save_fp_patterns()
        return candidate, True


def _remove_patterns_for_descriptions(descriptions) -> int:
    """Drop every FP pattern whose rule_description matches any in the given
    iterable (whitespace-trimmed, case-insensitive). Persists once if anything
    changed. Returns the number of patterns removed.

    Used by the FP restore flow so that un-marking an alert also tears down
    the auto-suppression pattern that was created when it was marked. Without
    this, Restore only frees the single event from the FP filter while the
    pattern keeps suppressing every other alert with the same signature —
    surprising the analyst and starving the engine of those alerts.
    """
    targets = {_norm_desc(d) for d in descriptions if d}
    if not targets:
        return 0
    removed = 0
    with _fp_pat_lock:
        keep = []
        for p in _fp_patterns:
            if _norm_desc(p.get("rule_description")) in targets:
                removed += 1
                continue
            keep.append(p)
        if removed:
            _fp_patterns[:] = keep
            save_fp_patterns()
    return removed


def _read_alerts_from_disk(n: int | None = None) -> list:
    """Refresh the cache from disk and return the raw alert list (no FP filtering)."""
    global _alert_cache, _alert_cache_mtime
    if n is None:
        n = CFG["max_alerts"]
    if not ALERTS_FILE.exists():
        return []
    try:
        mtime = ALERTS_FILE.stat().st_mtime
    except Exception:
        return _alert_cache
    if _alert_cache and mtime == _alert_cache_mtime and \
            (time.time() - _alert_cache_mtime) < CFG["cache_ttl_secs"]:
        return _alert_cache
    alerts = []
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                alerts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception:
        return _alert_cache
    _alert_cache = alerts
    _alert_cache_mtime = mtime
    return alerts


def load_alerts(n: int | None = None, include_fp: bool = False) -> list:
    """Load alerts. By default filters out any alert that is FP-marked, either
    by event_id (per-instance) or by a stored fingerprint pattern
    (signature_id + user + agent). Pass include_fp=True to opt out (used by
    the "show FPs" feed view and by /api/false-positives).
    """
    raw = _read_alerts_from_disk(n)
    if include_fp:
        return raw
    if not _fp_dict and not _fp_patterns:
        return raw
    out = []
    for a in raw:
        if a.get("event_id") in _fp_dict:
            continue
        if _matching_pattern(a) is not None:
            continue
        out.append(a)
    return out


def _wants_include_fp() -> bool:
    """Read ?include_fp=1 from the current request."""
    v = (request.args.get("include_fp") or "").strip().lower()
    return v in ("1", "true", "yes", "y")


def get_user(alert: dict) -> str:
    """Resolve the most meaningful username from an alert.
    Priority: subject.name > raw_event user fields > object.name
              > subject.ip > ae model (if personal) > host.name > 'unknown'
    """
    sub  = alert.get("subject", {}) or {}
    obj  = alert.get("object", {}) or {}
    ueba = alert.get("ueba", {}) or {}
    host = alert.get("host", {}) or {}
    raw  = (alert.get("context", {}) or {}).get("raw_event", {}) or {}
    win  = (raw.get("data", {}) or {}).get("win", {}) or {}
    wed  = (win.get("eventdata", {}) or {})
    raw_data = raw.get("data", {}) or {}

    name = sub.get("name")
    if name and name not in ("", None):
        return name

    for field in ("subjectUserName", "targetUserName", "subjectDomainName"):
        v = wed.get(field)
        if v and v not in ("", "-", None):
            return v

    for field in ("srcuser", "dstuser", "user", "username", "accountName"):
        v = raw_data.get(field)
        if v and v not in ("", "-", None):
            return v

    name = obj.get("name")
    if name and name not in ("", None):
        return name

    ip = sub.get("ip")
    if ip and ip not in ("", None):
        return ip

    model = ueba.get("raw_scores", {}).get("autoencoder", {}).get("model_used", "")
    if model and model not in ("global", "unknown", "", None):
        return model

    hname = host.get("name")
    if hname and hname not in ("", None):
        return hname

    return "unknown"


@app.route("/api/stats")
def stats():
    """Overall stats for the stats overview panel."""
    alerts = load_alerts(include_fp=_wants_include_fp())
    if not alerts:
        return jsonify({
            "total_alerts": 0, "highly_anomalous": 0, "anomalous": 0,
            "suspicious": 0, "campaigns": 0, "suppressed_noise": 0,
            "unique_users": 0, "alert_rate_1h": 0, "top_reasons": []
        })

    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)

    total        = len(alerts)
    highly       = sum(1 for a in alerts if a.get("ueba", {}).get("risk_verdict") == "highly_anomalous")
    anomalous    = sum(1 for a in alerts if a.get("ueba", {}).get("risk_verdict") == "anomalous")
    suspicious   = sum(1 for a in alerts if a.get("ueba", {}).get("risk_verdict") == "suspicious")
    campaigns    = len({a.get("ueba", {}).get("campaign_id") for a in alerts
                        if a.get("ueba", {}).get("campaign_id")})
    unique_users = len({get_user(a) for a in alerts})

    recent = []
    for a in alerts:
        try:
            t = datetime.fromisoformat(a.get("ueba", {}).get("processed_at", ""))
            if t >= one_hour_ago:
                recent.append(a)
        except Exception:
            pass

    reason_counts: dict = defaultdict(int)
    for a in alerts:
        for r in (a.get("ueba", {}).get("anomaly_reasons") or []):
            reason_counts[r] += 1
    top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]

    return jsonify({
        "total_alerts":    total,
        "highly_anomalous": highly,
        "anomalous":       anomalous,
        "suspicious":      suspicious,
        "campaigns":       campaigns,
        "unique_users":    unique_users,
        "alert_rate_1h":   len(recent),
        "top_reasons":     [{"reason": r, "count": c} for r, c in top_reasons],
    })


def _alert_to_feed_item(a: dict) -> dict:
    """Map a raw alert (as written by the engine) to the compact dict shape
    consumed by the dashboard feed table. Used by both /api/feed and the
    SSE /api/stream endpoint.
    """
    ueba = a.get("ueba", {}) or {}
    sec  = a.get("security", {}) or {}
    host = a.get("host", {}) or {}
    eid  = a.get("event_id")
    return {
        "event_id":     eid,
        "event_time":   a.get("event_time"),
        "processed_at": ueba.get("processed_at"),
        "user":         get_user(a),
        "verdict":      ueba.get("risk_verdict"),
        "score":        ueba.get("combined_score"),
        "reasons":      ueba.get("anomaly_reasons", []),
        "campaign_id":  ueba.get("campaign_id"),
        "signature":    (sec.get("signature") or "")[:80],
        "signature_id": sec.get("signature_id"),
        "severity":     sec.get("severity"),
        "host":         host.get("name"),
        "host_ip":      host.get("ip"),
        "mitre_tactic": (a.get("context", {}) or {}).get("raw_event", {}).get("rule", {}).get("mitre", {}).get("tactic", []),
        "evidence":     a.get("evidence", {}),
        "fp":           _fp_dict.get(eid) if eid else None,
        "fp_pattern":   _matching_pattern(a),
    }


@app.route("/api/feed")
def feed():
    """Live alert feed — all loaded alerts, newest first.
    FP-marked alerts are filtered out unless ?include_fp=1 is passed.
    """
    alerts = load_alerts(include_fp=_wants_include_fp())
    return jsonify([_alert_to_feed_item(a) for a in reversed(alerts)])


@app.route("/api/users")
def users():
    """User risk leaderboard — top 20 riskiest users."""
    alerts = load_alerts(include_fp=_wants_include_fp())
    user_stats: dict = defaultdict(lambda: {
        "count": 0, "max_score": 0.0, "verdicts": defaultdict(int),
        "reasons": defaultdict(int), "last_seen": "", "hosts": set()
    })

    for a in alerts:
        user  = get_user(a)
        ueba  = a.get("ueba", {}) or {}
        score = ueba.get("combined_score", 0) or 0
        verdict = ueba.get("risk_verdict", "")
        host = (a.get("host", {}) or {}).get("name", "")

        user_stats[user]["count"] += 1
        user_stats[user]["max_score"] = max(user_stats[user]["max_score"], score)
        user_stats[user]["verdicts"][verdict] += 1
        user_stats[user]["last_seen"] = ueba.get("processed_at", "")
        if host:
            user_stats[user]["hosts"].add(host)
        for r in (ueba.get("anomaly_reasons") or []):
            user_stats[user]["reasons"][r] += 1

    leaderboard = []
    for user, s in user_stats.items():
        if user in ("unknown", "global", ""):
            continue
        risk = s["max_score"] * 0.5 + (s["count"] / 100) * 0.3 + \
               (s["verdicts"].get("highly_anomalous", 0) / max(s["count"], 1)) * 0.2
        leaderboard.append({
            "user":             user,
            "alert_count":      s["count"],
            "max_score":        round(s["max_score"], 3),
            "risk_index":       round(risk, 3),
            "top_verdict":      max(s["verdicts"], key=s["verdicts"].get) if s["verdicts"] else "",
            "top_reason":       max(s["reasons"], key=s["reasons"].get) if s["reasons"] else "",
            "last_seen":        s["last_seen"],
            "hosts":            list(s["hosts"])[:3],
            "highly_anomalous": s["verdicts"].get("highly_anomalous", 0),
        })

    leaderboard.sort(key=lambda x: -x["risk_index"])
    return jsonify(leaderboard[:20])


@app.route("/api/campaigns")
def campaigns():
    """Campaign timeline data."""
    alerts = load_alerts(include_fp=_wants_include_fp())
    campaign_data: dict = defaultdict(lambda: {
        "alerts": [], "users": set(), "hosts": set(),
        "first_seen": "", "last_seen": "", "verdicts": defaultdict(int),
        "reasons": defaultdict(int), "signatures": set()
    })

    for a in alerts:
        ueba = a.get("ueba", {}) or {}
        cid  = ueba.get("campaign_id")
        if not cid:
            continue
        user = get_user(a)
        host = (a.get("host", {}) or {}).get("name", "")
        t    = ueba.get("processed_at", "")
        sig  = (a.get("security", {}) or {}).get("signature", "")[:60]

        cd = campaign_data[cid]
        cd["users"].add(user)
        if host:
            cd["hosts"].add(host)
        cd["verdicts"][ueba.get("risk_verdict", "")] += 1
        for r in (ueba.get("anomaly_reasons") or []):
            cd["reasons"][r] += 1
        if sig:
            cd["signatures"].add(sig)
        if not cd["first_seen"] or t < cd["first_seen"]:
            cd["first_seen"] = t
        if t > cd["last_seen"]:
            cd["last_seen"] = t
        cd["alerts"].append({
            "time":    t,
            "user":    user,
            "score":   ueba.get("combined_score", 0),
            "verdict": ueba.get("risk_verdict", ""),
        })

    result = []
    for cid, cd in campaign_data.items():
        result.append({
            "campaign_id":   cid,
            "alert_count":   len(cd["alerts"]),
            "users":         list(cd["users"]),
            "hosts":         list(cd["hosts"]),
            "first_seen":    cd["first_seen"],
            "last_seen":     cd["last_seen"],
            "top_verdict":   max(cd["verdicts"], key=cd["verdicts"].get) if cd["verdicts"] else "",
            "top_reason":    max(cd["reasons"], key=cd["reasons"].get) if cd["reasons"] else "",
            "signatures":    list(cd["signatures"])[:3],
            "timeline":      sorted(cd["alerts"], key=lambda x: x["time"])[-20:],
            "highly_anomalous": cd["verdicts"].get("highly_anomalous", 0),
        })

    result.sort(key=lambda x: x["last_seen"], reverse=True)
    return jsonify(result)


@app.route("/api/agents")
def agents():
    """Agent (endpoint) summary — all registered agents, merged with alert stats."""
    registry: list = []
    if AGENTS_REGISTRY.exists():
        try:
            registry = json.loads(AGENTS_REGISTRY.read_text())
        except Exception:
            pass

    alerts = load_alerts(include_fp=_wants_include_fp())
    agent_stats: dict = defaultdict(lambda: {
        "alert_count": 0, "max_score": 0.0,
        "verdicts": defaultdict(int), "users": set(),
        "reasons": defaultdict(int), "last_seen": "",
        "first_seen": "", "ip": "",
    })

    for a in alerts:
        host = (a.get("host", {}) or {}).get("name", "")
        if not host:
            continue
        ueba    = a.get("ueba", {}) or {}
        score   = ueba.get("combined_score", 0) or 0
        verdict = ueba.get("risk_verdict", "")
        user    = get_user(a)
        t       = ueba.get("processed_at", "")
        ip      = (a.get("host", {}) or {}).get("ip", "")

        s = agent_stats[host]
        s["alert_count"] += 1
        s["max_score"] = max(s["max_score"], score)
        s["verdicts"][verdict] += 1
        s["users"].add(user)
        if ip:
            s["ip"] = ip
        for r in (ueba.get("anomaly_reasons") or []):
            s["reasons"][r] += 1
        if not s["first_seen"] or t < s["first_seen"]:
            s["first_seen"] = t
        if t > s["last_seen"]:
            s["last_seen"] = t

    seen = set()
    result = []
    for reg in registry:
        name = reg["name"]
        seen.add(name)
        s = agent_stats.get(name, {})
        verdicts = s.get("verdicts", {})
        result.append({
            "agent":            name,
            "ip":               s.get("ip") or reg.get("ip", ""),
            "os":               reg.get("os", ""),
            "agent_id":         reg.get("id", ""),
            "alert_count":      s.get("alert_count", 0),
            "max_score":        round(s.get("max_score", 0), 3),
            "top_verdict":      max(verdicts, key=verdicts.get) if verdicts else "",
            "top_reason":       max(s["reasons"], key=s["reasons"].get) if s.get("reasons") else "",
            "users":            list(s.get("users", set()))[:10],
            "last_seen":        s.get("last_seen", ""),
            "first_seen":       s.get("first_seen", ""),
            "highly_anomalous": verdicts.get("highly_anomalous", 0),
            "anomalous":        verdicts.get("anomalous", 0),
            "suspicious":       verdicts.get("suspicious", 0),
        })

    for name, s in agent_stats.items():
        if name in seen:
            continue
        verdicts = s["verdicts"]
        result.append({
            "agent":            name,
            "ip":               s["ip"],
            "os":               "",
            "agent_id":         "",
            "alert_count":      s["alert_count"],
            "max_score":        round(s["max_score"], 0),
            "top_verdict":      max(verdicts, key=verdicts.get) if verdicts else "",
            "top_reason":       max(s["reasons"], key=s["reasons"].get) if s["reasons"] else "",
            "users":            list(s["users"])[:10],
            "last_seen":        s["last_seen"],
            "first_seen":       s["first_seen"],
            "highly_anomalous": verdicts.get("highly_anomalous", 0),
            "anomalous":        verdicts.get("anomalous", 0),
            "suspicious":       verdicts.get("suspicious", 0),
        })

    result.sort(key=lambda x: (-x["highly_anomalous"], -x["max_score"]))
    return jsonify(result)


@app.route("/api/agent/<agent_name>")
def agent_alerts(agent_name):
    """All alerts for a specific agent, newest first."""
    alerts = load_alerts(include_fp=_wants_include_fp())
    out = []
    for a in reversed(alerts):
        if ((a.get("host", {}) or {}).get("name", "")) != agent_name:
            continue
        out.append(_alert_to_feed_item(a))
    return jsonify(out)


@app.route("/api/user/<path:username>")
def user_alerts(username):
    """All alerts for a specific user, newest first."""
    alerts = load_alerts(include_fp=_wants_include_fp())
    out = []
    for a in reversed(alerts):
        if get_user(a) != username:
            continue
        out.append(_alert_to_feed_item(a))
    return jsonify(out)


# ── False-positive management ────────────────────────────────────────────────
@app.route("/api/false-positives")
def list_false_positives():
    """Return all FP records, each enriched with the original alert payload."""
    # Always read the raw alert list (include_fp=True) so we can look up the
    # underlying alert even if it's been FP-filtered.
    raw = load_alerts(include_fp=True)
    by_id = {a.get("event_id"): a for a in raw if a.get("event_id")}
    out = []
    for eid, rec in _fp_dict.items():
        a = by_id.get(eid)
        out.append({**rec, "alert": _alert_to_feed_item(a) if a else None})
    out.sort(key=lambda r: r.get("marked_at", ""), reverse=True)
    return jsonify(out)


@app.route("/api/false-positive", methods=["POST"])
def mark_false_positive():
    """Mark a single alert as FP. Also auto-creates a suppression pattern
    derived from the alert's (signature_id, user, agent) fingerprint so future
    similar alerts are filtered out of the feed. The per-event record stays
    around as an audit trail (which alert prompted the pattern).
    """
    body = request.get_json(silent=True) or {}
    eid = (body.get("event_id") or "").strip()
    reason = (body.get("reason") or "").strip()
    if not eid:
        return jsonify({"ok": False, "error": "event_id required"}), 400
    rec = {
        "event_id":  eid,
        "reason":    reason,
        "marked_at": datetime.now(timezone.utc).isoformat(),
    }
    with _fp_lock:
        _fp_dict[eid] = rec
        save_fps()

    # Auto-suppress similar alerts: derive a pattern from the underlying alert.
    raw = _read_alerts_from_disk()
    alert = next((a for a in raw if a.get("event_id") == eid), None)
    pat, pat_new = (None, False)
    if alert:
        candidate = _pattern_from_alert(alert, reason)
        pat, pat_new = _add_pattern_if_new(candidate)
    return jsonify({
        "ok":          True,
        "record":      rec,
        "pattern":     pat,       # None if alert missing or no signature_id
        "pattern_new": pat_new,   # False if pattern already existed (dedupe)
    })


@app.route("/api/false-positive/<path:event_id>", methods=["DELETE"])
def unmark_false_positive(event_id):
    """Restore an FP-marked alert AND tear down the auto-suppression pattern
    that was created when it was marked. The mark and the pattern are created
    together by mark_false_positive — symmetry says they should be torn down
    together too.
    """
    with _fp_lock:
        existed = _fp_dict.pop(event_id, None)
        if existed:
            save_fps()

    # Resolve the alert's rule description and drop any matching pattern.
    patterns_removed = 0
    if existed:
        raw = _read_alerts_from_disk()
        alert = next((a for a in raw if a.get("event_id") == event_id), None)
        if alert:
            desc = (alert.get("security", {}) or {}).get("signature")
            patterns_removed = _remove_patterns_for_descriptions([desc])

    return jsonify({
        "ok":               True,
        "removed":          bool(existed),
        "patterns_removed": patterns_removed,
    })


def _event_ids_in_campaign(campaign_id: str) -> list[str]:
    """Scan the raw (unfiltered) alert list and return every event_id whose
    ueba.campaign_id equals campaign_id."""
    raw = _read_alerts_from_disk()
    return [
        a["event_id"]
        for a in raw
        if a.get("event_id")
        and (a.get("ueba", {}) or {}).get("campaign_id") == campaign_id
    ]


@app.route("/api/false-positive/campaign/<path:campaign_id>", methods=["POST"])
def mark_campaign_false_positive(campaign_id):
    """Mark every alert in a campaign as FP. Also auto-creates one pattern per
    unique rule description within the campaign so the suppression follows
    similar alerts forward in time."""
    body = request.get_json(silent=True) or {}
    reason = (body.get("reason") or "").strip() \
             or f"campaign {campaign_id} marked as false positive"
    eids = _event_ids_in_campaign(campaign_id)
    if not eids:
        return jsonify({"ok": False, "error": "no alerts for campaign"}), 404
    now = datetime.now(timezone.utc).isoformat()
    with _fp_lock:
        for eid in eids:
            _fp_dict[eid] = {"event_id": eid, "reason": reason, "marked_at": now}
        save_fps()

    # Auto-suppress: one pattern per unique fingerprint in the campaign.
    raw = _read_alerts_from_disk()
    by_id = {a.get("event_id"): a for a in raw}
    seen: set = set()
    patterns_new = 0
    for eid in eids:
        a = by_id.get(eid)
        if not a:
            continue
        candidate = _pattern_from_alert(a, reason)
        if not candidate:
            continue
        key = _norm_desc(candidate["rule_description"])
        if key in seen:
            continue
        seen.add(key)
        _, was_new = _add_pattern_if_new(candidate)
        if was_new:
            patterns_new += 1
    return jsonify({
        "ok":           True,
        "marked":       len(eids),
        "campaign_id":  campaign_id,
        "patterns_new": patterns_new,
    })


@app.route("/api/false-positive/campaign/<path:campaign_id>", methods=["DELETE"])
def unmark_campaign_false_positive(campaign_id):
    """Restore every alert in a campaign AND drop the auto-suppression
    patterns that were created when the campaign was marked. Mirrors the
    per-event unmark — mark and pattern are created together, so torn down
    together.
    """
    eids = _event_ids_in_campaign(campaign_id)
    with _fp_lock:
        removed = 0
        for eid in eids:
            if _fp_dict.pop(eid, None):
                removed += 1
        if removed:
            save_fps()

    # Collect descriptions from this campaign's alerts and tear down patterns.
    raw = _read_alerts_from_disk()
    by_id = {a.get("event_id"): a for a in raw}
    descs = {
        (by_id.get(eid, {}).get("security") or {}).get("signature")
        for eid in eids if eid in by_id
    }
    patterns_removed = _remove_patterns_for_descriptions(descs)

    return jsonify({
        "ok":               True,
        "removed":          removed,
        "campaign_id":      campaign_id,
        "patterns_removed": patterns_removed,
    })


# ── False-positive PATTERN management ────────────────────────────────────────
# Patterns auto-suppress any future alert whose rule description matches
# (whitespace-trimmed, case-insensitive). One pattern = one rule description.
@app.route("/api/false-positive-patterns")
def list_fp_patterns():
    """Return all FP patterns with a `matched` count over the loaded alerts."""
    raw = load_alerts(include_fp=True)
    counts: dict = {}
    for a in raw:
        p = _matching_pattern(a)
        if p:
            counts[p.get("id")] = counts.get(p.get("id"), 0) + 1
    out = []
    for p in _fp_patterns:
        out.append({**p, "matched": counts.get(p.get("id"), 0)})
    out.sort(key=lambda r: r.get("marked_at", ""), reverse=True)
    return jsonify(out)


@app.route("/api/false-positive-pattern", methods=["POST"])
def mark_fp_pattern():
    """Add a new FP pattern directly. Body: {rule_description, reason}."""
    body = request.get_json(silent=True) or {}
    desc = (str(body.get("rule_description") or "")).strip()
    reason = (body.get("reason") or "").strip()
    if not desc:
        return jsonify({"ok": False, "error": "rule_description required"}), 400
    candidate = {
        "id":               "fpp-" + uuid.uuid4().hex[:10],
        "rule_description": desc,
        "reason":           reason,
        "marked_at":        datetime.now(timezone.utc).isoformat(),
    }
    rec, was_new = _add_pattern_if_new(candidate)
    return jsonify({"ok": True, "record": rec, "deduped": (not was_new)})


@app.route("/api/false-positive-pattern/<pat_id>", methods=["DELETE"])
def unmark_fp_pattern(pat_id):
    with _fp_pat_lock:
        before = len(_fp_patterns)
        _fp_patterns[:] = [p for p in _fp_patterns if p.get("id") != pat_id]
        removed = before - len(_fp_patterns)
        if removed:
            save_fp_patterns()
    return jsonify({"ok": True, "removed": bool(removed)})


# ── Engine health ────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    """Engine + dashboard health snapshot for the Overview health card."""
    alerts = load_alerts(include_fp=True)   # include FPs so we report the real engine output
    total = len(alerts)

    now = datetime.now(timezone.utc)
    one_hour_ago  = now - timedelta(hours=1)
    one_day_ago   = now - timedelta(hours=24)
    alerts_1h     = 0
    alerts_24h    = 0
    newest_iso    = None
    newest_dt     = None
    for a in alerts:
        ts = (a.get("ueba", {}) or {}).get("processed_at") or a.get("event_time") or ""
        try:
            d = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except Exception:
            d = None
        if d is None:
            continue
        if d >= one_hour_ago: alerts_1h  += 1
        if d >= one_day_ago:  alerts_24h += 1
        if newest_dt is None or d > newest_dt:
            newest_dt  = d
            newest_iso = ts

    try:
        file_size = ALERTS_FILE.stat().st_size if ALERTS_FILE.exists() else 0
    except Exception:
        file_size = 0

    fp_count = len(_fp_dict)
    fp_rate  = round((fp_count / total) * 100, 2) if total > 0 else 0.0

    try:
        started = datetime.fromisoformat(_SERVER_STARTED_AT)
        uptime  = int((now - started).total_seconds())
    except Exception:
        uptime = 0

    # "Engine live" = at least one alert within the last 5 minutes.
    engine_live = bool(newest_dt and (now - newest_dt).total_seconds() < 300)

    with _agent_sync_lock:
        agent_sync_status = {
            "enabled":    bool((CFG.get("agent_sync") or {}).get("enabled")),
            "last_at":    _agent_sync_last_at,
            "last_error": _agent_sync_last_error,
        }

    return jsonify({
        "total_alerts":           total,
        "alerts_1h":              alerts_1h,
        "alerts_24h":             alerts_24h,
        "newest_alert_time":      newest_iso,
        "alerts_file_size_bytes": file_size,
        "fp_count":               fp_count,
        "fp_rate_pct":            fp_rate,
        "server_started_at":      _SERVER_STARTED_AT,
        "uptime_secs":            uptime,
        "engine_live":            engine_live,
        "agent_sync":             agent_sync_status,
    })


# ── Live event stream (Server-Sent Events) ───────────────────────────────────
# /api/stream tails ueba_alerts.jsonl from "now" and pushes every new alert as
# an SSE 'alert' event. The dashboard's initial state is already loaded via
# /api/feed at boot — the stream only delivers what arrives after connection.

@app.route("/api/stream")
def stream_alerts():
    @stream_with_context
    def event_stream():
        # Send a hello frame so the EventSource transitions from 'connecting'
        # to 'open' immediately, even before the first alert arrives.
        yield "event: hello\ndata: {}\n\n"

        # Start at the current end of the file — don't replay history.
        try:
            last_pos = ALERTS_FILE.stat().st_size if ALERTS_FILE.exists() else 0
        except Exception:
            last_pos = 0
        last_heartbeat = time.time()

        while True:
            try:
                if ALERTS_FILE.exists():
                    try:
                        size = ALERTS_FILE.stat().st_size
                    except Exception:
                        size = last_pos
                    if size < last_pos:
                        # File rotated / truncated — start from the new end.
                        last_pos = 0
                    if size > last_pos:
                        try:
                            with open(ALERTS_FILE, "r", encoding="utf-8", errors="replace") as f:
                                f.seek(last_pos)
                                chunk = f.read()
                                last_pos = f.tell()
                        except Exception:
                            chunk = ""
                        for line in chunk.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                alert = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            eid = alert.get("event_id")
                            if eid and eid in _fp_dict:
                                continue   # already a known FP, don't push it
                            if _matching_pattern(alert) is not None:
                                continue   # auto-suppressed by a stored pattern
                            payload = _alert_to_feed_item(alert)
                            yield f"event: alert\ndata: {json.dumps(payload, default=str)}\n\n"

                now = time.time()
                if now - last_heartbeat > 15:
                    # SSE comment lines keep the connection alive through
                    # proxies/load balancers without delivering an event.
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
                time.sleep(1.0)
            except GeneratorExit:
                return
            except Exception as e:
                # Surface and continue; the client will keep us open.
                try:
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                except Exception:
                    return
                time.sleep(2.0)

    headers = {
        "Cache-Control":     "no-cache",
        "X-Accel-Buffering": "no",       # disable nginx buffering
        "Connection":        "keep-alive",
    }
    return Response(event_stream(), mimetype="text/event-stream", headers=headers)


# ── AI Security Analyst proxy ────────────────────────────────────────────────
@app.route("/api/ai-analyze", methods=["POST"])
def ai_analyze():
    """Proxy the AI Security Analyst prompt to Anthropic with a server-side key.

    Request body: { "prompt": "..." }
    Response: { "ok": bool, "text": "...", "source": "anthropic"|"fallback", "error": "..." }
    """
    ai_cfg = CFG.get("ai_analyst", {}) or {}
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "missing prompt", "source": "fallback"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if not ai_cfg.get("enabled", True):
        return jsonify({"ok": False, "error": "ai_analyst disabled", "source": "fallback"})
    if not api_key:
        return jsonify({"ok": False, "error": "ANTHROPIC_API_KEY not set", "source": "fallback"})
    if requests is None:
        return jsonify({"ok": False, "error": "requests not installed", "source": "fallback"})

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model":      ai_cfg.get("model", "claude-sonnet-4-6"),
                "max_tokens": int(ai_cfg.get("max_tokens", 1000)),
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=int(ai_cfg.get("timeout_secs", 20)),
        )
        if r.status_code != 200:
            return jsonify({
                "ok": False,
                "error": f"upstream {r.status_code}: {r.text[:200]}",
                "source": "fallback",
            })
        data = r.json()
        text = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
        if not text:
            return jsonify({"ok": False, "error": "empty response", "source": "fallback"})
        return jsonify({"ok": True, "text": text, "source": "anthropic"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "source": "fallback"})


# ── Static + index ───────────────────────────────────────────────────────────
def _serve_index():
    """Serve dashboard/dist/index.html if it exists, otherwise the legacy file."""
    dist_index = DASHBOARD_DIST / "index.html"
    if dist_index.exists():
        return send_from_directory(str(DASHBOARD_DIST), "index.html")
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


@app.route("/")
def index():
    return _serve_index()


@app.route("/assets/<path:filename>")
def dist_assets(filename):
    """Vite's hashed JS/CSS bundles live under dashboard/dist/assets/."""
    return send_from_directory(str(DASHBOARD_DIST / "assets"), filename)


@app.route("/legacy")
def legacy_index():
    """Always serve the pre-Vite single-file dashboard for fallback debugging."""
    return send_from_directory(str(DASHBOARD_DIR), "index.legacy.html")


# SPA fallback — the client-side router activates the right tab based on the
# URL. The order is:
#   1. If a file with that exact path exists in dashboard/dist/ (e.g. logo
#      images copied from public/, favicon.ico, etc.), serve it directly.
#   2. Otherwise, if the first path segment is a known tab name, return
#      index.html so the client router can activate the right tab.
#   3. Otherwise, 404.
SPA_TAB_PATHS = {"overview", "feed", "users", "campaigns", "endpoints", "false-positives"}

@app.route("/<path:subpath>")
def spa_fallback(subpath):
    # 1. Real file in dist/ (Vite's public/ outputs land here at build time).
    candidate = DASHBOARD_DIST / subpath
    try:
        if candidate.is_file() and candidate.resolve().is_relative_to(DASHBOARD_DIST.resolve()):
            return send_from_directory(str(DASHBOARD_DIST), subpath)
    except (OSError, ValueError):
        pass
    # 2. Known SPA tab — return index.html for client-side routing.
    head = subpath.split("/", 1)[0]
    if head in SPA_TAB_PATHS:
        return _serve_index()
    # 3. Otherwise: not found.
    from flask import abort
    abort(404)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CyberSentinel UEBA Dashboard API server")
    p.add_argument("--config", default="ueba_config.yaml",
                   help="Path to ueba_config.yaml (default: ueba_config.yaml)")
    p.add_argument("--host", default=None, help="Bind host (overrides config)")
    p.add_argument("--port", type=int, default=None, help="Bind port (overrides config)")
    p.add_argument("--alerts", default=None, help="Path to ueba_alerts.jsonl (overrides config)")
    p.add_argument("--agents-registry", default=None,
                   help="Path to agents.json (overrides config)")
    p.add_argument("--fp-file", default=None,
                   help="Path to false_positives.json (overrides config)")
    p.add_argument("--fp-patterns-file", default=None,
                   help="Path to fp_patterns.json (overrides config)")
    p.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    return p.parse_args()


def main() -> None:
    global CFG, ALERTS_FILE, AGENTS_REGISTRY, FP_FILE, FP_PATTERNS_FILE
    args = _parse_args()
    CFG = load_dashboard_config(args.config)

    if args.host:             CFG["host"] = args.host
    if args.port:             CFG["port"] = args.port
    if args.alerts:           CFG["alerts_file"] = args.alerts
    if args.agents_registry:  CFG["agents_registry"] = args.agents_registry
    if args.fp_file:          CFG["false_positives_file"] = args.fp_file
    if args.fp_patterns_file: CFG["fp_patterns_file"] = args.fp_patterns_file

    ALERTS_FILE       = Path(CFG["alerts_file"])
    AGENTS_REGISTRY   = Path(CFG["agents_registry"])
    FP_FILE           = Path(CFG["false_positives_file"])
    FP_PATTERNS_FILE  = Path(CFG.get("fp_patterns_file") or _DEFAULTS["fp_patterns_file"])
    load_fps()
    load_fp_patterns()

    dist_ok = (DASHBOARD_DIST / "index.html").exists()
    print("CyberSentinel UEBA Dashboard")
    print(f"  Config:           {args.config}")
    print(f"  Reading alerts:   {ALERTS_FILE}")
    print(f"  Agents registry:  {AGENTS_REGISTRY}")
    print(f"  False positives:  {FP_FILE}  ({len(_fp_dict)} loaded)")
    print(f"  FP patterns:      {FP_PATTERNS_FILE}  ({len(_fp_patterns)} loaded)")
    print(f"  Dashboard build:  {'dist/ (Vite build)' if dist_ok else 'legacy index.html'}")
    print(f"  AI analyst:       "
          f"{'enabled' if CFG['ai_analyst'].get('enabled') and os.environ.get('ANTHROPIC_API_KEY') else 'fallback only'}")
    print(f"  Listening on:     http://{CFG['host']}:{CFG['port']}")
    _start_agent_sync_thread()
    app.run(host=CFG["host"], port=CFG["port"], debug=args.debug)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ueba_dashboard_server.py
─────────────────────────
CyberSentinel UEBA — Dashboard API Server

Serves ueba_alerts.jsonl data as JSON endpoints for the dashboard.
Run: python3 ueba_dashboard_server.py
Access: http://164.52.194.98:5000
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="dashboard")
CORS(app)

ALERTS_FILE = Path("/root/NEW_DRIVE/aditya_ueba/ueba_alerts.jsonl")
MAX_ALERTS  = 10000   # read last N alerts for performance


# ── Alert cache — avoids reading the full file on every API request ──
_alert_cache      = []
_alert_cache_mtime = 0.0
_CACHE_TTL_SECS    = 10   # re-read if file is >10s old in mtime terms

def load_alerts(n=MAX_ALERTS):
    """Load last N alerts — cached by file mtime, refreshes every 10s max."""
    global _alert_cache, _alert_cache_mtime
    if not ALERTS_FILE.exists():
        return []
    try:
        mtime = ALERTS_FILE.stat().st_mtime
    except Exception:
        return _alert_cache
    # Return cache if file hasn't changed in the last TTL window
    import time
    if _alert_cache and mtime == _alert_cache_mtime and \
            (time.time() - _alert_cache_mtime) < _CACHE_TTL_SECS:
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


def get_user(alert):
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

    # 1. Direct subject name — most reliable
    name = sub.get("name")
    if name and name not in ("", None):
        return name

    # 2. Windows event data — subjectUserName / targetUserName
    for field in ("subjectUserName", "targetUserName", "subjectDomainName"):
        v = wed.get(field)
        if v and v not in ("", "-", None):
            return v

    # 3. Generic raw_data fields (FortiGate, Syslog, etc)
    for field in ("srcuser", "dstuser", "user", "username", "accountName"):
        v = raw_data.get(field)
        if v and v not in ("", "-", None):
            return v

    # 4. Object name (destination user in auth events)
    name = obj.get("name")
    if name and name not in ("", None):
        return name

    # 5. Subject IP — used when no username exists (e.g. netstat rule 533)
    ip = sub.get("ip")
    if ip and ip not in ("", None):
        return ip

    # 6. Autoencoder model_used — only if it's a real personal model
    #    Exclude 'global', 'unknown', '' which are not usernames
    model = ueba.get("raw_scores", {}).get("autoencoder", {}).get("model_used", "")
    if model and model not in ("global", "unknown", "", None):
        return model

    # 7. Host name — last resort before 'unknown'
    #    Better to show 'SocSRV_15' than 'unknown' for host-level events
    hname = host.get("name")
    if hname and hname not in ("", None):
        return hname

    return "unknown"


@app.route("/api/stats")
def stats():
    """Overall stats for the stats overview panel."""
    alerts = load_alerts()
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

    # Alerts in last 1 hour
    recent = []
    for a in alerts:
        try:
            t = datetime.fromisoformat(a.get("ueba", {}).get("processed_at", ""))
            if t >= one_hour_ago:
                recent.append(a)
        except Exception:
            pass

    # Top anomaly reasons
    reason_counts = defaultdict(int)
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


@app.route("/api/feed")
def feed():
    """Live alert feed — all loaded alerts, newest first."""
    alerts = load_alerts()
    feed_alerts = []
    for a in reversed(alerts):
        ueba = a.get("ueba", {}) or {}
        sec  = a.get("security", {}) or {}
        host = a.get("host", {}) or {}
        feed_alerts.append({
            "event_id":      a.get("event_id"),
            "event_time":    a.get("event_time"),
            "processed_at":  ueba.get("processed_at"),
            "user":          get_user(a),
            "verdict":       ueba.get("risk_verdict"),
            "score":         ueba.get("combined_score"),
            "reasons":       ueba.get("anomaly_reasons", []),
            "campaign_id":   ueba.get("campaign_id"),
            "signature":     sec.get("signature", "")[:80],
            "signature_id":  sec.get("signature_id"),
            "severity":      sec.get("severity"),
            "host":          host.get("name"),
            "host_ip":       host.get("ip"),
            "mitre_tactic":  (a.get("context", {}) or {}).get("raw_event", {}).get("rule", {}).get("mitre", {}).get("tactic", []),
            "evidence":      a.get("evidence", {}),   # full evidence block for SOC analysts
        })
    return jsonify(feed_alerts)


@app.route("/api/users")
def users():
    """User risk leaderboard — top 20 riskiest users."""
    alerts = load_alerts()
    user_stats = defaultdict(lambda: {
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

    # Score users: weight by max_score * count + highly_anomalous bonus
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
    alerts = load_alerts()
    campaign_data = defaultdict(lambda: {
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
    # Load static agent registry
    registry = []
    registry_path = Path("/root/NEW_DRIVE/aditya_ueba/agents.json")
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except Exception:
            pass

    alerts = load_alerts()
    agent_stats = defaultdict(lambda: {
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

    # Build result — start from registry so all agents appear
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

    # Add any agents seen in alerts but not in registry
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
    alerts = load_alerts()
    result = []
    for a in reversed(alerts):
        host = (a.get("host", {}) or {}).get("name", "")
        if host != agent_name:
            continue
        ueba = a.get("ueba", {}) or {}
        sec  = a.get("security", {}) or {}
        result.append({
            "event_id":     a.get("event_id"),
            "event_time":   a.get("event_time"),
            "processed_at": ueba.get("processed_at"),
            "user":         get_user(a),
            "verdict":      ueba.get("risk_verdict"),
            "score":        ueba.get("combined_score"),
            "reasons":      ueba.get("anomaly_reasons", []),
            "campaign_id":  ueba.get("campaign_id"),
            "signature":    sec.get("signature", "")[:80],
            "signature_id": sec.get("signature_id"),
            "severity":     sec.get("severity"),
            "host":         host,
            "host_ip":      (a.get("host", {}) or {}).get("ip"),
            "mitre_tactic": (a.get("context", {}) or {}).get("raw_event", {}).get("rule", {}).get("mitre", {}).get("tactic", []),
            "evidence":     a.get("evidence", {}),
        })
    return jsonify(result)


@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


if __name__ == "__main__":
    print("CyberSentinel UEBA Dashboard")
    print(f"Reading alerts from: {ALERTS_FILE}")
    print("Starting server on http://0.0.0.0:3026")
    app.run(host="0.0.0.0", port=3026, debug=False)

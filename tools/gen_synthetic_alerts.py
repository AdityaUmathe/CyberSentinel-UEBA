#!/usr/bin/env python3
"""
gen_synthetic_alerts.py
────────────────────────
Generate a realistic synthetic ``ueba_alerts.jsonl`` file for local dashboard
development. The emitted alerts match the schema written by
``ueba_engine.py``'s ``build_ueba_block`` + ``build_evidence_block``, so every
panel in the dashboard renders without engine-side changes.

Usage:
    python3 tools/gen_synthetic_alerts.py
    python3 tools/gen_synthetic_alerts.py --count 5000 --days 30
    python3 tools/gen_synthetic_alerts.py --output /tmp/ueba_alerts.jsonl --seed 42

The companion ``agents.json`` registry can also be emitted with
``--write-agents-registry`` so the ``/api/agents`` endpoint shows them on the
Endpoints tab even when there are no alerts for them yet.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Vocabulary ────────────────────────────────────────────────────────────────
USERS = [
    "jdoe", "asmith", "mwilliams", "rkumar", "nshah", "okonkwo",
    "RootSeeker\\Sujal", "LAPTOP-08MIT8SI\\Asus", "naveen.g",
    "service.scan", "svc_backup", "admin.it",
]

HOSTS = [
    ("SocSRV_15",     "10.0.4.15",   "Windows Server 2019"),
    ("SocSRV_18",     "10.0.4.18",   "Windows Server 2019"),
    ("WS-FIN-04",     "10.0.10.42",  "Windows 11 Pro"),
    ("WS-DEV-07",     "10.0.20.71",  "Ubuntu 22.04"),
    ("LAPTOP-08MIT8SI","10.0.30.8",  "Windows 11"),
    ("FW-PERIM-01",   "203.0.113.4", "FortiOS 7.4"),
    ("DC-PRIMARY",    "10.0.1.2",    "Windows Server 2022"),
    ("WEB-PUB-02",    "203.0.113.20","Ubuntu 22.04"),
]

# (signature, signature_id, severity, category, mitre_tactic, mitre_technique, mitre_id)
SIGNATURES = [
    ("RDP logon from external IP",                    "92653", 12, "auth",
     ["Initial Access", "Lateral Movement"], ["T1021.001"], ["T1021"]),
    ("Multiple failed SSH logins",                    "5712",  10, "auth",
     ["Credential Access"],                  ["T1110.001"], ["T1110"]),
    ("Suspicious PowerShell encoded command",         "92660", 14, "process",
     ["Execution", "Defense Evasion"],       ["T1059.001"], ["T1059"]),
    ("Outbound traffic to known C2 destination",      "120010",13, "network",
     ["Command and Control"],                ["T1071.001"], ["T1071"]),
    ("File written to Windows Defender exclusion",    "92800", 12, "file",
     ["Defense Evasion"],                    ["T1562.001"], ["T1562"]),
    ("Service installed (potential persistence)",     "92700", 11, "process",
     ["Persistence"],                        ["T1543.003"], ["T1543"]),
    ("LSASS memory access by unsigned process",       "92750", 15, "process",
     ["Credential Access"],                  ["T1003.001"], ["T1003"]),
    ("Large outbound transfer (>500MB)",              "81620", 9,  "network",
     ["Exfiltration"],                       ["T1041"],     ["T1041"]),
    ("Unauthorised cloud admin API call",             "120052",13, "cloud",
     ["Privilege Escalation"],               ["T1548"],     ["T1548"]),
    ("Brute force detection (>20 failures / 5m)",     "5715",  13, "auth",
     ["Credential Access"],                  ["T1110.003"], ["T1110"]),
    ("DNS tunneling pattern detected",                "120030",12, "network",
     ["Command and Control"],                ["T1572"],     ["T1572"]),
    ("Scheduled task created from unusual user",      "92710", 10, "process",
     ["Persistence"],                        ["T1053.005"], ["T1053"]),
]

ANOMALY_REASONS_POOL = [
    "behavioral baseline deviation",
    "isolation forest anomaly",
    "after-hours activity",
    "brute force pattern",
    "lateral movement signal",
    "data exfiltration volume",
    "impossible travel",
    "TOR exit node source",
    "high event frequency",
    "process spawned by unusual parent",
    "cross-border auth",
    "new device for user",
]

LOGON_TYPES = ["Interactive", "Network", "RemoteInteractive (RDP)",
               "NetworkCleartext", "CachedInteractive", "Batch", "Service"]

COUNTRIES = ["IN", "US", "GB", "RU", "CN", "DE", "BR", "SG"]


# ── Verdict / score banding ───────────────────────────────────────────────────
def pick_verdict_and_score(rng: random.Random) -> tuple[str, float, str]:
    """Return (verdict, combined_score, confidence) using the same bands as
    the engine's score_fusion config (0.75..1.0)."""
    r = rng.random()
    if r < 0.20:
        score = rng.uniform(0.93, 1.0)
        return "highly_anomalous", round(score, 4), "high"
    if r < 0.55:
        score = rng.uniform(0.85, 0.93)
        return "anomalous", round(score, 4), "medium"
    score = rng.uniform(0.75, 0.85)
    return "suspicious", round(score, 4), "low"


# ── Builder helpers ───────────────────────────────────────────────────────────
def _evidence(
    rng: random.Random,
    user: str,
    host: tuple[str, str, str],
    sig: tuple,
    score: float,
    verdict: str,
    confidence: str,
    reasons: list[str],
    when: datetime,
    profile_first_seen: datetime,
    alerts_today: int,
    campaign_id: str | None,
) -> dict:
    description, sig_id, severity, _category, tactics, techniques, ids = sig
    host_name, host_ip, _os = host

    source_ip = f"10.0.{rng.randint(0,50)}.{rng.randint(1,254)}"
    dest_ip   = f"{rng.randint(20,203)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
    typical_hour = rng.randint(8, 18)
    current_hour = when.hour
    typical_risk = rng.uniform(20, 45)
    current_risk = typical_risk + (score - 0.5) * 60
    avg_ev_1h    = rng.randint(8, 90)
    cur_ev_1h    = max(avg_ev_1h, int(avg_ev_1h * (1 + (score - 0.5) * 4)))
    ae_thresh    = round(rng.uniform(0.01, 0.05), 4)
    ae_error     = round(ae_thresh * rng.uniform(1.5, 12.0), 4)
    if_raw       = round(-rng.uniform(0.50, 0.72), 4)

    raw_event = {
        "event_category":   _category,
        "event_outcome":    rng.choice(["success", "failure", "unknown"]),
        "host":             host_name,
        "host_ip":          host_ip,
        "source_ip":        source_ip,
        "dest_ip":          dest_ip,
        "process_name":     rng.choice(["powershell.exe", "cmd.exe", "rundll32.exe",
                                        "explorer.exe", "sshd", "python3", "curl"]),
        "logon_type":       rng.choice(LOGON_TYPES),
        "event_id_windows": str(rng.choice([4624, 4625, 4688, 4768, 4769, 7045])),
        "failures_5m":      rng.choice([0, 0, 0, 1, 3, 12, 47, 120]),
        "user_events_5m":   rng.randint(2, 600),
        "unique_dests_1h":  rng.randint(1, 35),
        "is_tor":           rng.random() < 0.04,
        "threat_detected":  rng.random() < 0.07,
    }
    if rng.random() < 0.10:
        raw_event["privileges"] = rng.sample(
            ["SeDebugPrivilege", "SeBackupPrivilege", "SeRestorePrivilege",
             "SeTakeOwnershipPrivilege", "SeImpersonatePrivilege"], k=rng.randint(1, 3))

    deviation = round(ae_error / max(ae_thresh, 0.0001), 1)
    baseline = {
        "typical_hour":          f"{typical_hour % 12 or 12}:00 {'AM' if typical_hour < 12 else 'PM'}",
        "current_hour":          f"{current_hour % 12 or 12}:00 {'AM' if current_hour < 12 else 'PM'}",
        "hour_deviation":        f"{abs(current_hour - typical_hour)}h from baseline",
        "typical_risk_score":    round(typical_risk, 1),
        "current_risk_score":    round(current_risk, 1),
        "risk_deviation":        f"+{round(current_risk - typical_risk, 1)}",
        "typical_events_1h":     avg_ev_1h,
        "current_events_1h":     cur_ev_1h,
        "events_multiplier":     f"{round(cur_ev_1h / max(avg_ev_1h, 1), 1)}x normal rate",
        "is_business_hours":     9 <= current_hour <= 18,
        "is_weekend":            when.weekday() >= 5,
        "day_of_week":           when.strftime("%A"),
        "seen_countries":        rng.sample(COUNTRIES, k=rng.randint(1, 3)),
        "autoencoder_error":     ae_error,
        "autoencoder_threshold": ae_thresh,
        "error_vs_threshold":    f"{deviation}x above threshold",
        "model_used":            user if rng.random() < 0.6 else "global",
        "if_raw_score":          if_raw,
    }

    history = {
        "first_seen":        profile_first_seen.isoformat(),
        "last_seen":         when.isoformat(),
        "total_events_seen": rng.randint(500, 80_000),
        "alerts_today":      alerts_today,
        "anomaly_reasons":   reasons,
        "combined_score":    score,
        "verdict":           verdict,
        "confidence":        confidence,
    }
    if campaign_id:
        history["campaign_id"] = campaign_id

    signature_block = {
        "rule_id":          sig_id,
        "description":      description,
        "severity_level":   severity,
        "mitre_tactics":    tactics,
        "mitre_techniques": techniques,
        "mitre_ids":        ids,
        "tags":             [],
    }
    return {"signature": signature_block, "raw_event": raw_event,
            "baseline": baseline, "history": history}


def make_alert(rng: random.Random, when: datetime, campaign_id: str | None,
               alerts_today: int) -> dict:
    user = rng.choice(USERS)
    host = rng.choice(HOSTS)
    sig  = rng.choice(SIGNATURES)
    verdict, score, confidence = pick_verdict_and_score(rng)
    reasons = rng.sample(ANOMALY_REASONS_POOL, k=rng.randint(1, 4))

    profile_first_seen = when - timedelta(days=rng.randint(30, 400))
    host_name, host_ip, _os = host
    description, sig_id, severity, _category, tactics, techniques, ids = sig

    event_id = f"evt-{uuid.uuid4().hex[:12]}"
    return {
        "event_id":   event_id,
        "event_time": when.isoformat(),
        "subject": {
            "name": user,
            "ip":   f"10.0.{rng.randint(0,50)}.{rng.randint(1,254)}",
        },
        "object": {
            "name": rng.choice([None, "Administrator", "service.dbus", user]),
            "ip":   f"10.0.{rng.randint(0,50)}.{rng.randint(1,254)}",
        },
        "host": {"name": host_name, "ip": host_ip},
        "security": {
            "signature":    description,
            "signature_id": sig_id,
            "severity":     severity,
        },
        "context": {
            "raw_event": {
                "rule": {
                    "mitre": {
                        "tactic":    tactics,
                        "technique": techniques,
                        "id":        ids,
                    }
                },
                "data": {
                    "win": {
                        "eventdata": {
                            "subjectUserName": user,
                            "logonType":       "10" if "RDP" in description else "3",
                        }
                    }
                }
            }
        },
        "ueba": {
            "processed_at":     when.isoformat(),
            "risk_verdict":     verdict,
            "confidence":       confidence,
            "combined_score":   score,
            "is_alert":         True,
            "anomaly_reasons":  reasons,
            "campaign_id":      campaign_id,
            "similar_past_events": [],
            "raw_scores": {
                "isolation_forest": {
                    "anomaly_score": round(rng.uniform(0.5, 1.0), 4),
                    "raw_score":     round(-rng.uniform(0.50, 0.72), 4),
                    "verdict":       verdict,
                },
                "autoencoder": {
                    "deviation_score":      round(rng.uniform(0.4, 0.95), 4),
                    "reconstruction_error": round(rng.uniform(0.02, 0.4), 4),
                    "threshold":            round(rng.uniform(0.01, 0.05), 4),
                    "is_anomalous":         True,
                    "model_used":           user if rng.random() < 0.6 else "global",
                    "model_type":           "personal" if rng.random() < 0.6 else "global",
                },
            },
        },
        "evidence": _evidence(
            rng, user, host, sig, score, verdict, confidence, reasons,
            when, profile_first_seen, alerts_today, campaign_id,
        ),
    }


def generate(count: int, days: int, seed: int, campaign_rate: float) -> list[dict]:
    """Generate ``count`` alerts spread over the last ``days`` days.

    ~``campaign_rate`` fraction of alerts share a campaign_id, grouped in
    bursts of 3–12 events to mimic DBSCAN clusters.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    timestamps = sorted(
        start + timedelta(seconds=rng.randint(0, days * 86400))
        for _ in range(count)
    )

    # Decide which alerts go into campaigns
    campaign_assignment: list[str | None] = [None] * count
    i = 0
    cid_num = 1
    while i < count:
        if rng.random() < campaign_rate:
            burst = rng.randint(3, 12)
            cid = f"CAMP-{cid_num:04d}"
            cid_num += 1
            for j in range(min(burst, count - i)):
                campaign_assignment[i + j] = cid
            i += burst
        else:
            i += 1

    # Track alerts per day for the alerts_today evidence field
    alerts_today_by_date: dict = {}
    out = []
    for ts, cid in zip(timestamps, campaign_assignment):
        date_key = ts.date().isoformat()
        alerts_today_by_date[date_key] = alerts_today_by_date.get(date_key, 0) + 1
        out.append(make_alert(rng, ts, cid, alerts_today_by_date[date_key]))
    return out


def write_agents_registry(path: Path) -> None:
    registry = [
        {"name": h[0], "ip": h[1], "os": h[2], "id": f"{i:03d}"}
        for i, h in enumerate(HOSTS, start=1)
    ]
    path.write_text(json.dumps(registry, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic ueba_alerts.jsonl")
    p.add_argument("--count",   type=int,   default=2000, help="Number of alerts to generate")
    p.add_argument("--days",    type=int,   default=30,   help="Spread alerts over this many days")
    p.add_argument("--seed",    type=int,   default=42,   help="RNG seed for reproducibility")
    p.add_argument("--campaign-rate", type=float, default=0.25,
                   help="Probability each alert position starts a campaign burst")
    p.add_argument("--output",  default="ueba_alerts.jsonl",
                   help="Output path (will be overwritten)")
    p.add_argument("--write-agents-registry", default=None,
                   help="Also write the agents.json registry to this path")
    args = p.parse_args()

    alerts = generate(args.count, args.days, args.seed, args.campaign_rate)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, default=str) + "\n")
    print(f"wrote {len(alerts):,} alerts → {out}")

    if args.write_agents_registry:
        reg_path = Path(args.write_agents_registry)
        write_agents_registry(reg_path)
        print(f"wrote agents registry ({len(HOSTS)} hosts) → {reg_path}")


if __name__ == "__main__":
    main()

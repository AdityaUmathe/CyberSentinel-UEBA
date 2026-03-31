#!/usr/bin/env python3
"""
ueba_preprocessor.py
────────────────────
CyberSentinel UEBA — Feature Extraction & Preprocessing

Reads enriched.jsonl (output of L2 enrich_json.py), extracts numerical
feature vectors from each log, and writes them to an HDF5 file for
efficient training. Also maintains per-user behavioral profiles in SQLite.

This is the FOUNDATION of the entire UEBA pipeline. Every other component
depends on the feature vectors produced here.

Key design decisions:
  - Processes in chunks (never loads full file into RAM)
  - Uses plain Python dicts (no Pandas) for single-log inference at runtime
  - HDF5 output supports random access slicing for training
  - Handles missing/null fields gracefully (fills with 0.0)
  - Categorical fields are integer-encoded using config mappings

Usage:
  # Offline — preprocess training data
  python3 ueba_preprocessor.py \
      --input  combined_training.jsonl \
      --output data/features.h5 \
      --config ueba_config.yaml \
      --mode   batch

  # Check what features look like for a single log
  python3 ueba_preprocessor.py --demo --config ueba_config.yaml
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ueba.preprocessor")


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Nested Field Getter ───────────────────────────────────────────────────────

def get_nested(d: dict, key_path: str, default=None) -> Any:
    """
    Safely get a nested dict value using dot notation.
    Example: get_nested(log, 'enrich.temporal.hour', 0)
    Returns default if any key in the path is missing or None.
    """
    keys = key_path.split(".")
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k)
        if val is None:
            return default
    return val


def safe_float(val, default: float = 0.0) -> float:
    """Convert any value to float safely."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_bool(val, default: float = 0.0) -> float:
    """Convert boolean/truthy value to 0.0 or 1.0."""
    if val is None:
        return default
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return 1.0 if val else 0.0
    if isinstance(val, str):
        return 1.0 if val.lower() in ("true", "yes", "1") else 0.0
    return default


# ── Noise Filter ──────────────────────────────────────────────────────────────

# Fortigate rule IDs that represent normal policy enforcement — not behavioral anomalies.
# These fire thousands of times per day on any corporate network and have no
# investigative value for UEBA. Suppress them from alert generation entirely.
_NOISE_SIGNATURE_IDS = {
    "92653",   # RDP logon on SocSRV_15 — normal teammate access
    "67022",   # Non-network local logon — machine account scheduled task noise
    "67028",   # Special privileges assigned to new logon — RDP false positive on SocSRV_15
    "120052",   # Sysmon event1: Process spawned with elevated privileges (SYSTEM — normal Windows behavior)
    "130001",   # Sysmon event3: Termius/mDNS multicast discovery — false positive
    "60107",   # Failed privileged operation (SocSRV_15 chrome.exe noise)
    "81621",   # Fortigate: Multiple high traffic events to same port
    "81620",   # Fortigate: Multiple high traffic events from same destination
    "81619",   # Fortigate: Multiple high traffic events from same source
    "113202",   # MikroTik: L2TP tunnel authenticated
    "81633",   # Fortigate: App passed by firewall
    "81634",   # Fortigate: App blocked by firewall
    "81644",   # Fortigate: Blocked URL (denied category)
}

# Fortigate rule IDs that are low-value but should still be scored
# (kept in training, suppressed from alerts unless score >= 0.95)
_LOW_VALUE_SIGNATURE_IDS = {
    "81612",   # Fortigate: Firewall configuration changes (low severity)
    "81631",   # Fortigate: Firewall pass (low level)
}

_SUPPRESS_SOURCE_IPS = {
    "164.52.194.98",   # engine server — reverse tunnel + rsync traffic
}

# Known internal agents/hostnames that should never appear as alert subjects.
_SUPPRESS_AGENT_NAMES = {
    "DESKTOP-QO9VL7M$",   # Windows machine account
    "NT AUTHORITY\\NETWOR",   # Windows NETWORK SERVICE account
    "NT AUTHORITY\\SYSTEM",   # Windows SYSTEM account — Defender/scheduler noise
    "cybersentinel-manager",
    "cybersentinel-manage",
    "Centos7Trial",
    "164.52.194.98",
}

def is_noise(log: dict) -> bool:
    """
    Returns True if this event should be completely skipped for alert generation.
    These are high-volume, low-value firewall policy events that are normal
    behaviour on any corporate network and should not trigger UEBA alerts.

    Note: These events are still used during TRAINING so the model learns
    what normal network traffic looks like. They are only filtered at the
    alert output stage.
    """
    sig_id = str(get_nested(log, "security.signature_id") or "")
    if sig_id in _NOISE_SIGNATURE_IDS:
        return True

    # Also suppress if no anomaly flags are set AND event is pure Fortigate
    # network allow/pass with anomaly_count == 0
    source = str(get_nested(log, "context.source") or "")
    if "fortigate" in source.lower():
        action = str(get_nested(log, "event_action") or "").lower()
        anomaly_count = get_nested(log, "enrich.anomalies.anomaly_count") or 0
        if action in ("allow", "connect") and int(anomaly_count) == 0:
            return True

    # Suppress Windows machine accounts (name ending in $) — never real users
    user_name = str(get_nested(log, "subject.name") or "")
    if user_name.endswith("$"):
        return True

    # Suppress mDNS multicast traffic (224.x.x.x) — Termius, printers, etc.
    subject_ip = str(get_nested(log, "subject.ip") or "")
    if subject_ip.startswith("224."):
        return True
    # Suppress port 5353 (mDNS) — never security relevant
    subject_port = str(get_nested(log, "subject.port") or "")
    object_port  = str(get_nested(log, "object.port") or "")
    if subject_port == "5353" and object_port == "5353":
        return True

    # Suppress engine server's own IP — tunnel/rsync traffic
    src_ip = str(get_nested(log, "enrich.geo.src_ip") or
                 get_nested(log, "context.raw_event.data.srcip") or
                 get_nested(log, "source.ip") or "")
    if src_ip in _SUPPRESS_SOURCE_IPS:
        return True

    dst_ip = str(get_nested(log, "context.raw_event.data.dstip") or
                 get_nested(log, "destination.ip") or "")
    if dst_ip in _SUPPRESS_SOURCE_IPS:
        return True

    # Suppress events with no resolvable user identity
    user = str(get_nested(log, "subject.name") or "")
    if not user or user.lower() in ("none", "null", ""):
        anomaly_count = int(get_nested(log, "enrich.anomalies.anomaly_count") or 0)
        if anomaly_count == 0:
            return True

    candidate_names = {
        str(get_nested(log, "subject.name") or ""),
        str(get_nested(log, "subject.ip") or ""),
        str(get_nested(log, "object.name") or ""),
        str(get_nested(log, "host.name") or ""),
        str(get_nested(log, "host.ip") or ""),
        str(get_nested(log, "context.raw_event.data.dstuser") or ""),
        str(get_nested(log, "context.raw_event.data.srcuser") or ""),
    }
    if candidate_names & _SUPPRESS_AGENT_NAMES:
        return True

    # Suppress Windows machine accounts (name ending in $) — never real users
    user_name = str(get_nested(log, "subject.name") or "")
    if user_name.endswith("$"):
        return True

    # Suppress mDNS multicast traffic (224.x.x.x) — Termius, printers, etc.
    subject_ip = str(get_nested(log, "subject.ip") or "")
    if subject_ip.startswith("224."):
        return True
    # Suppress port 5353 (mDNS) — never security relevant
    subject_port = str(get_nested(log, "subject.port") or "")
    object_port  = str(get_nested(log, "object.port") or "")
    if subject_port == "5353" and object_port == "5353":
        return True

    # Suppress engine server's own IP — tunnel/rsync traffic
    src_ip = str(get_nested(log, "enrich.geo.src_ip") or
                 get_nested(log, "context.raw_event.data.srcip") or
                 get_nested(log, "source.ip") or "")
    if src_ip in _SUPPRESS_SOURCE_IPS:
        return True

    dst_ip = str(get_nested(log, "context.raw_event.data.dstip") or
                 get_nested(log, "destination.ip") or "")
    if dst_ip in _SUPPRESS_SOURCE_IPS:
        return True

    # Suppress events with no resolvable user identity
    user = str(get_nested(log, "subject.name") or "")
    if not user or user.lower() in ("none", "null", ""):
        anomaly_count = int(get_nested(log, "enrich.anomalies.anomaly_count") or 0)
        if anomaly_count == 0:
            return True

    return False


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Converts a single enriched log (Python dict) into a fixed-size
    numpy feature vector of shape (feature_dim,).

    This class is used both:
      - During offline batch preprocessing (training data)
      - During real-time streaming (one log at a time)

    The feature ordering is FIXED and documented below.
    If you add new features, append them at the end and update feature_dim
    in ueba_config.yaml.

    Feature index map (35 features total):
    ─────────────────────────────────────
    [0]  temporal.hour              0-23
    [1]  temporal.day_of_week       0-6
    [2]  temporal.is_weekend        0/1
    [3]  temporal.is_business_hours 0/1
    [4]  temporal.is_night_shift    0/1
    [5]  flags.is_auth_event        0/1
    [6]  flags.is_network_event     0/1
    [7]  flags.is_file_event        0/1
    [8]  flags.is_process_event     0/1
    [9]  anomalies.is_brute_force          0/1
    [10] anomalies.is_port_scan            0/1
    [11] anomalies.is_tor                  0/1
    [12] anomalies.is_malicious_ip         0/1
    [13] anomalies.is_impossible_travel    0/1
    [14] anomalies.is_lateral_movement     0/1
    [15] anomalies.is_data_exfiltration    0/1
    [16] anomalies.is_after_hours          0/1
    [17] anomalies.is_high_frequency       0/1
    [18] counters.src_ip_5m         integer
    [19] counters.src_ip_1h         integer
    [20] counters.user_1h           integer
    [21] counters.host_1h           integer
    [22] behavioral.unique_destinations_1h integer
    [23] behavioral.unique_ports_1h        integer
    [24] behavioral.recent_failures_5m     integer
    [25] geo.distance_km            float (0 if unknown)
    [26] geo.cross_border           0/1
    [27] geo.cross_continent        0/1
    [28] risk_score                 0-100
    [29] rule.level                 0-15
    [30] network_intel.is_cloud     0/1
    [31] event_category             0-10 (encoded)
    [32] event_outcome              0-2  (encoded)
    [33] traffic_direction          0-3  (encoded)
    [34] user_deviation_baseline    float (from profile store, 0 if new user)
    """

    FEATURE_DIM = 35

    # Category encodings (must match ueba_config.yaml)
    EVENT_CATEGORY_MAP = {
        "auth": 0, "network": 1, "process": 2, "file": 3,
        "dns": 4, "web": 5, "malware": 6, "cloud": 7,
        "policy": 8, "system": 9, "unknown": 10,
    }
    EVENT_OUTCOME_MAP = {
        "success": 0, "failure": 1, "unknown": 2,
    }
    TRAFFIC_DIRECTION_MAP = {
        "inbound": 0, "outbound": 1, "internal": 2, "unknown": 3,
    }

    def __init__(self, config: dict, profile_store=None):
        self.config = config
        self.profile_store = profile_store  # optional, used for feature [34]
        self.null_fill = config["preprocessing"].get("null_fill_value", 0.0)

    def extract(self, log: dict) -> np.ndarray:
        """
        Extract feature vector from one enriched log dict.
        Returns numpy array of shape (FEATURE_DIM,) with dtype float32.
        Never raises — missing fields become null_fill_value (0.0).
        """
        vec = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # ── [0-4] Temporal ────────────────────────────────────────────────────
        # Support both schema versions: hour vs hour_of_day, day_of_week (int) vs day name
        hour = get_nested(log, "enrich.temporal.hour") or get_nested(log, "enrich.temporal.hour_of_day")
        vec[0]  = safe_float(hour, 0.0)
        dow = get_nested(log, "enrich.temporal.day_of_week")
        if isinstance(dow, str):
            dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            dow = dow_map.get(dow, 0)
        vec[1]  = safe_float(dow, 0.0)
        vec[2]  = safe_bool(get_nested(log,  "enrich.temporal.is_weekend"))
        vec[3]  = safe_bool(get_nested(log,  "enrich.temporal.is_business_hours"))
        # is_night_shift (old schema) or is_night (new schema)
        vec[4]  = safe_bool(get_nested(log, "enrich.temporal.is_night_shift") or get_nested(log, "enrich.temporal.is_night"))

        # ── [5-8] Event flags ─────────────────────────────────────────────────
        vec[5]  = safe_bool(get_nested(log, "enrich.flags.is_auth_event"))
        vec[6]  = safe_bool(get_nested(log, "enrich.flags.is_network_event"))
        vec[7]  = safe_bool(get_nested(log, "enrich.flags.is_file_event"))
        vec[8]  = safe_bool(get_nested(log, "enrich.flags.is_process_event"))

        # ── [9-17] Anomaly flags ──────────────────────────────────────────────
        vec[9]  = safe_bool(get_nested(log, "enrich.anomalies.is_brute_force"))
        vec[10] = safe_bool(get_nested(log, "enrich.anomalies.is_port_scan"))
        # is_tor (old schema) or is_tor_traffic (new schema)
        vec[11] = safe_bool(get_nested(log, "enrich.anomalies.is_tor") or get_nested(log, "enrich.anomalies.is_tor_traffic"))
        # is_malicious_ip (old) — check threat_detected in new schema
        vec[12] = safe_bool(get_nested(log, "enrich.anomalies.is_malicious_ip") or get_nested(log, "enrich.network_intel.threat_detected"))
        vec[13] = safe_bool(get_nested(log, "enrich.anomalies.is_impossible_travel"))
        vec[14] = safe_bool(get_nested(log, "enrich.anomalies.is_lateral_movement"))
        # is_data_exfiltration (old) or is_data_exfil (new)
        vec[15] = safe_bool(get_nested(log, "enrich.anomalies.is_data_exfiltration") or get_nested(log, "enrich.anomalies.is_data_exfil"))
        vec[16] = safe_bool(get_nested(log, "enrich.anomalies.is_after_hours"))
        vec[17] = safe_bool(get_nested(log, "enrich.anomalies.is_high_frequency") or get_nested(log, "enrich.anomalies.is_high_frequency"))

        # ── [18-21] Rolling counters ──────────────────────────────────────────
        # New schema only has 5m counters; estimate 1h as 5m * 12 if 1h missing
        src_ip_5m  = safe_float(get_nested(log, "enrich.counters.src_ip_5m"))
        src_ip_1h  = get_nested(log, "enrich.counters.src_ip_1h")
        vec[18] = src_ip_5m
        vec[19] = safe_float(src_ip_1h) if src_ip_1h is not None else src_ip_5m * 12
        user_1h    = get_nested(log, "enrich.counters.user_1h")
        vec[20] = safe_float(user_1h) if user_1h is not None else 0.0
        host_5m    = safe_float(get_nested(log, "enrich.counters.host_5m"))
        host_1h    = get_nested(log, "enrich.counters.host_1h")
        vec[21] = safe_float(host_1h) if host_1h is not None else host_5m * 12

        # ── [22-24] Behavioral ────────────────────────────────────────────────
        vec[22] = safe_float(get_nested(log, "enrich.behavioral.unique_destinations_1h"))
        vec[23] = safe_float(get_nested(log, "enrich.behavioral.unique_ports_1h"))
        vec[24] = safe_float(get_nested(log, "enrich.behavioral.recent_failures_5m"))

        # ── [25-27] Geo ───────────────────────────────────────────────────────
        vec[25] = safe_float(get_nested(log, "enrich.geo.distance_km"))  # 0 if missing
        # cross_border/cross_continent: old schema top-level, new schema in anomalies
        vec[26] = safe_bool(get_nested(log, "enrich.geo.cross_border") or get_nested(log, "enrich.anomalies.cross_border"))
        vec[27] = safe_bool(get_nested(log, "enrich.geo.cross_continent") or get_nested(log, "enrich.anomalies.cross_continent"))

        # ── [28-29] Risk ──────────────────────────────────────────────────────
        vec[28] = safe_float(get_nested(log, "enrich.risk_score"))
        # rule.level: old schema top-level, new schema in context.raw_event.rule.level
        rule_level = (
            get_nested(log, "rule.level")
            or get_nested(log, "security.severity")
            or get_nested(log, "context.raw_event.rule.level")
        )
        vec[29] = safe_float(rule_level)

        # ── [30] Network intel ────────────────────────────────────────────────
        is_cloud = get_nested(log, "enrich.network_intel.cloud_provider")
        vec[30] = 0.0 if (is_cloud is None or is_cloud == "") else 1.0

        # ── [31] Event category (encoded) ─────────────────────────────────────
        cat = str(get_nested(log, "event_category", "unknown")).lower()
        vec[31] = float(self.EVENT_CATEGORY_MAP.get(cat, 10))

        # ── [32] Event outcome (encoded) ──────────────────────────────────────
        outcome = str(get_nested(log, "event_outcome", "unknown")).lower()
        vec[32] = float(self.EVENT_OUTCOME_MAP.get(outcome, 2))

        # ── [33] Traffic direction (encoded) ──────────────────────────────────
        # Old schema: enrich.classification.traffic_direction
        # New schema: enrich.normalization.direction (inbound/outbound) or network.direction
        direction = str(
            get_nested(log, "enrich.classification.traffic_direction")
            or get_nested(log, "enrich.normalization.direction")
            or get_nested(log, "network.direction")
            or "unknown"
        ).lower()
        # Normalize new schema values to old schema encoding
        direction = direction.replace("outgoing", "outbound").replace("incoming", "inbound")
        vec[33] = float(self.TRAFFIC_DIRECTION_MAP.get(direction, 3))

        # ── [34] User baseline deviation ──────────────────────────────────────
        # How different is this event from this user's historical average?
        # 0.0 if user is new / profile store not available
        if self.profile_store:
            user = (
                get_nested(log, "subject.name")
                or get_nested(log, "data.dstuser")
                or get_nested(log, "agent.name")
                or "unknown"
            )
            vec[34] = self.profile_store.get_baseline_deviation(user, vec)
        else:
            vec[34] = 0.0

        return vec

    def extract_metadata(self, log: dict) -> dict:
        """
        Extract non-numeric metadata from a log for storing alongside
        the feature vector (used by FAISS metadata store and output writer).
        Supports both real Wazuh schema (subject/object) and legacy schema (data.srcuser).
        """
        # Username: try all known locations across schema versions
        # IMPORTANT: object.name on network events contains hostnames (e.g. www.google.com)
        # Only use object.name as username fallback for auth events
        is_auth = (
            get_nested(log, "event_category") == "auth"
            or safe_bool(get_nested(log, "enrich.flags.is_auth_event"))
        )
        user = (
            get_nested(log, "subject.name")
            or get_nested(log, "data.srcuser")
            or get_nested(log, "data.dstuser")
            or get_nested(log, "context.raw_event.data.srcuser")
            or get_nested(log, "context.raw_event.data.dstuser")
            or (get_nested(log, "object.name") if is_auth else None)
            or "unknown"
        )

        # Source IP: try all known locations
        src_ip = (
            get_nested(log, "subject.ip")
            or get_nested(log, "data.srcip")
            or get_nested(log, "context.raw_event.data.srcip")
            or ""
        )

        # Destination IP
        dst_ip = (
            get_nested(log, "object.ip")
            or get_nested(log, "data.dstip")
            or get_nested(log, "context.raw_event.data.dstip")
            or ""
        )

        # Destination port
        dst_port = (
            get_nested(log, "object.port")
            or get_nested(log, "data.dstport")
            or get_nested(log, "context.raw_event.data.dstport")
            or ""
        )

        # Agent info: try both schema versions
        agent_name = (
            get_nested(log, "host.name")
            or get_nested(log, "agent.name")
            or get_nested(log, "context.raw_event.agent.name")
            or ""
        )
        agent_ip = (
            get_nested(log, "host.ip")
            or get_nested(log, "agent.ip")
            or get_nested(log, "context.raw_event.agent.ip")
            or ""
        )

        # Rule info
        rule_id   = get_nested(log, "security.signature_id") or get_nested(log, "rule.id", "")
        rule_desc = get_nested(log, "security.signature")    or get_nested(log, "rule.description", "")
        rule_level = (
            get_nested(log, "security.severity")
            or get_nested(log, "rule.level")
            or get_nested(log, "context.raw_event.rule.level")
            or 0
        )

        # Country
        country_src = (
            get_nested(log, "enrich.geo.src.country")
            or get_nested(log, "enrich.geo.src_country")
            or ""
        )

        # Protocol and direction
        protocol  = get_nested(log, "network.protocol") or get_nested(log, "enrich.normalization.protocol") or ""
        direction = (
            get_nested(log, "network.direction")
            or get_nested(log, "enrich.classification.traffic_direction")
            or get_nested(log, "context.raw_event.data.action")
            or ""
        )

        return {
            "event_id":       get_nested(log, "event_id", ""),
            "event_time":     get_nested(log, "event_time", ""),
            "user":           user,
            "src_ip":         src_ip,
            "dst_ip":         dst_ip,
            "dst_port":       str(dst_port) if dst_port else "",
            "agent_name":     agent_name,
            "agent_ip":       agent_ip,
            "agent_role":     get_nested(log, "host.type") or get_nested(log, "context.raw_event.agent.role", ""),
            "event_category": get_nested(log, "event_category", "unknown"),
            "event_action":   get_nested(log, "event_action", ""),
            "event_outcome":  get_nested(log, "event_outcome", "unknown"),
            "rule_id":        rule_id,
            "rule_desc":      rule_desc,
            "rule_level":     int(rule_level) if rule_level else 0,
            "risk_score":     get_nested(log, "enrich.risk_score", 0),
            "country_src":    country_src,
            "protocol":       protocol.upper() if protocol else "",
            "direction":      direction,
            "is_internal":    get_nested(log, "enrich.classification.is_internal_traffic") or False,
            "anomaly_count":  get_nested(log, "enrich.anomalies.anomaly_count", 0),
            "schema_version": get_nested(log, "schema_version", ""),
            "fingerprint":    get_nested(log, "enrich.fingerprint", ""),
            # Noise flag — engine should suppress alert output for these events.
            # They are still scored and used for FAISS/profile updates.
            "is_noise":       is_noise(log),
        }


# ── User Profile Store (SQLite) ───────────────────────────────────────────────

class UserProfileStore:
    """
    Maintains per-user behavioral baselines in SQLite.
    Used to compute feature [34] — how much does this event deviate
    from this user's historical norm?

    Schema:
        users table:
            username        TEXT PRIMARY KEY
            avg_hour        REAL    (average login hour)
            avg_risk        REAL    (average risk score)
            avg_events_1h   REAL    (average events per hour)
            seen_countries  TEXT    (JSON list)
            seen_agents     TEXT    (JSON list)
            total_events    INTEGER
            first_seen      TEXT
            last_seen       TEXT
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        log.info("Profile store opened: %s", db_path)

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                avg_hour        REAL    DEFAULT 12.0,
                avg_risk        REAL    DEFAULT 0.0,
                avg_events_1h   REAL    DEFAULT 0.0,
                seen_countries  TEXT    DEFAULT '[]',
                seen_agents     TEXT    DEFAULT '[]',
                total_events    INTEGER DEFAULT 0,
                first_seen      TEXT    DEFAULT '',
                last_seen       TEXT    DEFAULT ''
            )
        """)
        self.conn.commit()

    def update(self, log: dict, feature_vec: np.ndarray):
        """Update a user's profile with data from a new log event."""
        user = (
            get_nested(log, "subject.name")
            or get_nested(log, "data.dstuser")
            or "unknown"
        )

        existing = self.conn.execute(
            "SELECT * FROM users WHERE username = ?", (user,)
        ).fetchone()

        now = datetime.now(timezone.utc).isoformat()
        hour        = float(feature_vec[0])
        risk        = float(feature_vec[28])
        events_1h   = float(feature_vec[20])
        country     = get_nested(log, "enrich.geo.src.country", "")
        agent       = get_nested(log, "agent.name", "")

        if existing is None:
            # New user — insert baseline
            self.conn.execute("""
                INSERT INTO users
                    (username, avg_hour, avg_risk, avg_events_1h,
                     seen_countries, seen_agents, total_events, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                user, hour, risk, events_1h,
                json.dumps([country] if country else []),
                json.dumps([agent] if agent else []),
                now, now,
            ))
        else:
            # Existing user — exponential moving average update
            # alpha=0.01 means new events have small influence on baseline
            # (keeps baseline stable, not skewed by a single attack burst)
            alpha = 0.01
            n = existing[6]  # total_events column
            avg_hour      = (1 - alpha) * existing[1]      + alpha * hour
            avg_risk      = (1 - alpha) * existing[2]      + alpha * risk
            avg_events_1h = (1 - alpha) * existing[3]      + alpha * events_1h

            countries = json.loads(existing[4])
            if country and country not in countries:
                countries.append(country)
                if len(countries) > 50:       # cap to 50 unique countries
                    countries = countries[-50:]

            agents = json.loads(existing[5])
            if agent and agent not in agents:
                agents.append(agent)
                if len(agents) > 20:
                    agents = agents[-20:]

            self.conn.execute("""
                UPDATE users SET
                    avg_hour      = ?,
                    avg_risk      = ?,
                    avg_events_1h = ?,
                    seen_countries = ?,
                    seen_agents    = ?,
                    total_events   = ?,
                    last_seen      = ?
                WHERE username = ?
            """, (
                avg_hour, avg_risk, avg_events_1h,
                json.dumps(countries), json.dumps(agents),
                n + 1, now, user,
            ))

        self.conn.commit()

    def get_baseline_deviation(self, user: str, feature_vec: np.ndarray) -> float:
        """
        Return how much this event deviates from the user's baseline.
        Returns a 0.0-1.0 float. 0 = perfectly normal, 1 = extremely unusual.
        Returns 0.0 for unknown users.
        """
        row = self.conn.execute(
            "SELECT avg_hour, avg_risk, avg_events_1h FROM users WHERE username = ?",
            (user,)
        ).fetchone()

        if row is None:
            return 0.0   # new user, no baseline yet

        avg_hour, avg_risk, avg_events_1h = row
        current_hour  = float(feature_vec[0])
        current_risk  = float(feature_vec[28])
        current_ev_1h = float(feature_vec[20])

        # Normalized absolute deviations for each dimension
        hour_dev  = abs(current_hour  - avg_hour)  / 23.0        # max range = 23
        risk_dev  = abs(current_risk  - avg_risk)  / 100.0       # max range = 100
        ev_dev    = abs(current_ev_1h - avg_events_1h) / max(avg_events_1h + 1, 10)

        # Weighted average deviation
        deviation = (0.4 * hour_dev) + (0.4 * risk_dev) + (0.2 * ev_dev)
        return float(min(deviation, 1.0))

    def get_user_profile(self, user: str) -> dict | None:
        """Return full profile for a user, or None if not found."""
        row = self.conn.execute(
            "SELECT * FROM users WHERE username = ?", (user,)
        ).fetchone()
        if row is None:
            return None
        return {
            "username":       row[0],
            "avg_hour":       row[1],
            "avg_risk":       row[2],
            "avg_events_1h":  row[3],
            "seen_countries": json.loads(row[4]),
            "seen_agents":    json.loads(row[5]),
            "total_events":   row[6],
            "first_seen":     row[7],
            "last_seen":      row[8],
        }

    def get_all_users(self) -> list[str]:
        """Return list of all usernames in the profile store."""
        rows = self.conn.execute("SELECT username FROM users").fetchall()
        return [r[0] for r in rows]

    def get_user_event_count(self, user: str) -> int:
        """Return how many events we've seen for this user."""
        row = self.conn.execute(
            "SELECT total_events FROM users WHERE username = ?", (user,)
        ).fetchone()
        return row[0] if row else 0

    def close(self):
        self.conn.close()


# ── HDF5 Writer ───────────────────────────────────────────────────────────────

class HDF5Writer:
    """
    Writes feature vectors to an HDF5 file in chunks.
    Supports unlimited appending — file grows as chunks are added.
    Never loads more than one chunk into RAM.

    File structure:
        features.h5
            /features   shape: (N, 35)  dtype: float32
            /metadata   shape: (N,)     dtype: special (JSON strings)
    """

    def __init__(self, output_path: str, feature_dim: int):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.path = output_path
        self.feature_dim = feature_dim
        self.total_written = 0

        # Create file with resizable datasets
        self.f = h5py.File(output_path, "w")
        self.feat_ds = self.f.create_dataset(
            "features",
            shape=(0, feature_dim),
            maxshape=(None, feature_dim),   # unlimited rows
            dtype="float32",
            chunks=(10000, feature_dim),    # read/write 10K rows at a time
            compression="gzip",             # compress to save disk space (~3x)
            compression_opts=4,
        )
        # Metadata stored as JSON strings (one per row)
        self.meta_ds = self.f.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str),
        )
        log.info("HDF5 writer opened: %s", output_path)

    def write_chunk(self, vectors: np.ndarray, metadata: list[dict]):
        """
        Append a chunk of feature vectors and their metadata to the HDF5 file.
        vectors: shape (chunk_size, feature_dim)
        metadata: list of dicts, one per row
        """
        n = len(vectors)
        current = self.total_written

        # Resize datasets to accommodate new rows
        self.feat_ds.resize(current + n, axis=0)
        self.meta_ds.resize(current + n, axis=0)

        # Write chunk
        self.feat_ds[current:current + n] = vectors
        self.meta_ds[current:current + n] = [json.dumps(m) for m in metadata]

        self.f.flush()
        self.total_written += n

    def close(self):
        self.f.close()
        log.info("HDF5 writer closed. Total rows written: %d", self.total_written)


# ── Batch Preprocessor ────────────────────────────────────────────────────────

class BatchPreprocessor:
    """
    Reads combined_training.jsonl in chunks, extracts feature vectors,
    updates user profiles, and writes to HDF5.

    Memory usage: approximately chunk_size * 35 * 4 bytes
    At chunk_size=50000: ~7 MB per chunk (very comfortable).
    """

    def __init__(self, config: dict):
        self.config = config
        self.chunk_size = config["preprocessing"]["chunk_size"]
        self.feature_dim = config["preprocessing"]["feature_dim"]

        # Initialize profile store
        db_path = config["paths"]["profile_db"]
        self.profile_store = UserProfileStore(db_path)

        # Initialize feature extractor
        self.extractor = FeatureExtractor(config, self.profile_store)

    def run(self, input_path: str, output_path: str):
        """
        Main preprocessing loop.
        Reads input JSONL in chunks, writes feature matrix to HDF5.
        """
        writer = HDF5Writer(output_path, self.feature_dim)

        chunk_vectors  = []
        chunk_metadata = []
        total_logs     = 0
        skipped        = 0
        start_time     = time.time()
        chunk_num      = 0

        log.info("Starting batch preprocessing")
        log.info("  Input  : %s", input_path)
        log.info("  Output : %s", output_path)
        log.info("  Chunk  : %d logs", self.chunk_size)

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON
                try:
                    log_entry = json.loads(line)
                except json.JSONDecodeError as e:
                    log.debug("Line %d: JSON parse error: %s", line_num, e)
                    skipped += 1
                    continue

                # Extract feature vector
                try:
                    vec  = self.extractor.extract(log_entry)
                    meta = self.extractor.extract_metadata(log_entry)
                except Exception as e:
                    log.debug("Line %d: Feature extraction error: %s", line_num, e)
                    skipped += 1
                    continue

                chunk_vectors.append(vec)
                chunk_metadata.append(meta)
                total_logs += 1

                # Update user profile (every log)
                try:
                    self.profile_store.update(log_entry, vec)
                except Exception as e:
                    log.debug("Profile update error: %s", e)

                # Flush chunk when full
                if len(chunk_vectors) >= self.chunk_size:
                    chunk_num += 1
                    arr = np.stack(chunk_vectors, axis=0)
                    writer.write_chunk(arr, chunk_metadata)

                    elapsed = time.time() - start_time
                    rate = total_logs / elapsed
                    log.info(
                        "Chunk %d written | logs: %d | rate: %.0f/sec | "
                        "skipped: %d",
                        chunk_num, total_logs, rate, skipped,
                    )

                    chunk_vectors  = []
                    chunk_metadata = []

        # Flush remaining logs (last partial chunk)
        if chunk_vectors:
            chunk_num += 1
            arr = np.stack(chunk_vectors, axis=0)
            writer.write_chunk(arr, chunk_metadata)
            log.info("Final chunk %d written | %d logs", chunk_num, len(arr))

        writer.close()
        self.profile_store.close()

        elapsed = time.time() - start_time
        log.info("=" * 60)
        log.info("PREPROCESSING COMPLETE")
        log.info("  Total logs processed : %d", total_logs)
        log.info("  Skipped (errors)     : %d", skipped)
        log.info("  Chunks written       : %d", chunk_num)
        log.info("  Time elapsed         : %.1f minutes", elapsed / 60)
        log.info("  Output               : %s", output_path)

        output_mb = Path(output_path).stat().st_size / (1024 * 1024)
        log.info("  Output size          : %.1f MB", output_mb)
        log.info("=" * 60)

        return total_logs, skipped


# ── Demo Mode ─────────────────────────────────────────────────────────────────

def run_demo(config: dict):
    """
    Show what a feature vector looks like for a sample enriched log.
    Useful for verifying the extractor works correctly with your data.
    """
    # Sample enriched log (matches your actual Wazuh + L2 output format)
    sample_log = {
        "true": 1771327039.492798,
        "timestamp": "2026-02-17T16:47:19.290+0530",
        "rule": {
            "level": 3,
            "description": "PAM: Login session closed.",
            "id": "5502",
        },
        "agent": {"id": "013", "name": "DB_3", "ip": "34.34.34.3"},
        "event_id": "evt-001",
        "event_time": "2026-02-17T11:17:34Z",
        "event_category": "auth",
        "event_outcome": "success",
        "subject": {"name": "dakshay", "ip": "34.34.34.3"},
        "object":  {"ip": "192.168.1.10"},
        "data": {"dstuser": "dakshay"},
        "enrich": {
            "temporal": {
                "hour": 11,
                "day_of_week": 1,
                "is_weekend": False,
                "is_business_hours": True,
                "is_night_shift": False,
            },
            "flags": {
                "is_auth_event": True,
                "is_network_event": False,
                "is_file_event": False,
                "is_process_event": False,
            },
            "anomalies": {
                "is_brute_force": False,
                "is_port_scan": False,
                "is_tor": False,
                "is_malicious_ip": False,
                "is_impossible_travel": False,
                "is_lateral_movement": False,
                "is_data_exfiltration": False,
                "is_after_hours": False,
                "is_high_frequency": False,
            },
            "counters": {
                "src_ip_5m": 3,
                "src_ip_1h": 47,
                "user_1h": 12,
                "host_1h": 5,
            },
            "behavioral": {
                "unique_destinations_1h": 2,
                "unique_ports_1h": 1,
                "recent_failures_5m": 0,
            },
            "geo": {
                "distance_km": 0.0,
                "cross_border": False,
                "cross_continent": False,
                "src": {"country": "India"},
            },
            "risk_score": 15,
            "network_intel": {"cloud_provider": None},
            "classification": {"traffic_direction": "internal"},
        },
    }

    extractor = FeatureExtractor(config)
    vec  = extractor.extract(sample_log)
    meta = extractor.extract_metadata(sample_log)

    feature_names = [
        "temporal.hour", "temporal.day_of_week", "temporal.is_weekend",
        "temporal.is_business_hours", "temporal.is_night_shift",
        "flags.is_auth_event", "flags.is_network_event",
        "flags.is_file_event", "flags.is_process_event",
        "anomalies.is_brute_force", "anomalies.is_port_scan",
        "anomalies.is_tor", "anomalies.is_malicious_ip",
        "anomalies.is_impossible_travel", "anomalies.is_lateral_movement",
        "anomalies.is_data_exfiltration", "anomalies.is_after_hours",
        "anomalies.is_high_frequency",
        "counters.src_ip_5m", "counters.src_ip_1h",
        "counters.user_1h", "counters.host_1h",
        "behavioral.unique_destinations_1h", "behavioral.unique_ports_1h",
        "behavioral.recent_failures_5m",
        "geo.distance_km", "geo.cross_border", "geo.cross_continent",
        "risk_score", "rule.level", "network_intel.is_cloud",
        "event_category (encoded)", "event_outcome (encoded)",
        "traffic_direction (encoded)", "user_baseline_deviation",
    ]

    print("\n" + "=" * 60)
    print("DEMO — Feature Vector for Sample Log")
    print("=" * 60)
    print(f"{'Index':<6} {'Feature':<40} {'Value'}")
    print("-" * 60)
    for i, (name, val) in enumerate(zip(feature_names, vec)):
        print(f"[{i:>2}]  {name:<40} {val:.4f}")
    print("=" * 60)
    print(f"Vector shape  : {vec.shape}")
    print(f"Vector dtype  : {vec.dtype}")
    print(f"Non-zero vals : {np.count_nonzero(vec)}")
    print("\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UEBA Feature Extraction — JSON enriched logs → HDF5 feature matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  help="Input enriched JSONL file")
    parser.add_argument("--output", help="Output HDF5 file (default from config)")
    parser.add_argument("--config", default="ueba_config.yaml",
                        help="Config file path (default: ueba_config.yaml)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode — show feature vector for a sample log")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.demo:
        run_demo(config)
        return

    if not args.input:
        parser.error("--input is required (unless using --demo)")

    input_path  = args.input
    output_path = args.output or config["paths"]["features_h5"]

    # Create necessary directories
    for d in ["data", "profiles", "logs"]:
        os.makedirs(d, exist_ok=True)

    preprocessor = BatchPreprocessor(config)
    preprocessor.run(input_path, output_path)


if __name__ == "__main__":
    main()

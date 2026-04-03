# CyberSentinel UEBA
### User and Entity Behavior Analytics Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-GPU%20Accelerated-ee4c2c?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-009fff?style=flat-square" />
  <img src="https://img.shields.io/badge/Flask-Dashboard-black?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/Wazuh-SIEM%20Integration-red?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

A real-time behavioral analytics engine built on top of the **CyberSentinel SIEM** platform. CyberSentinel UEBA ingests enriched security events, builds per-user behavioral baselines using autoencoders and isolation forests, detects anomalies, clusters them into attack campaigns, and surfaces findings through a live SOC dashboard.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Stack](#running-the-stack)
- [Dashboard](#dashboard)
- [Retraining](#retraining)
- [Noise Suppression](#noise-suppression)
- [Deployment Notes](#deployment-notes)

---

## Architecture

```
┌─────────────────────────────────┐        ┌──────────────────────────────┐
│        222 — SOC Machine        │        │      98 — GPU Training       │
│  ┌─────────────────────────┐    │  SSH   │  ┌────────────────────────┐  │
│  │  Wazuh + Fortigate Logs │    │ Tunnel │  │   ueba_engine.py       │  │
│  │  → CyberSentinel SIEM   │◄───┼────────┤  │   (Inference)          │  │
│  │  → Kafka Enricher       │    │        │  │                        │  │
│  │  → enriched/*.jsonl.gz  │───►│        │  │   ueba_trainer.py      │  │
│  └─────────────────────────┘    │        │  │   (PyTorch, L40S GPU)  │  │
│  ┌─────────────────────────┐    │        │  └────────────────────────┘  │
│  │  Dashboard (port 3026)  │◄───┼────────┤  ┌────────────────────────┐  │
│  │  ueba_dashboard_server  │    │ rsync  │  │  ueba_alerts.jsonl     │  │
│  └─────────────────────────┘    │        │  └────────────────────────┘  │
└─────────────────────────────────┘        └──────────────────────────────┘
         ▲  autossh reverse tunnel  ▲
```

The engine runs on a powerful GPU server (98) and receives live enriched events from the SOC machine (222) via a reverse SSH tunnel. Alerts are synced back to 222 for the dashboard to serve.

---

## Features

- **Per-user autoencoder models** — individual behavioral baselines for every user seen in training data, falling back to a global model for unknown users
- **Isolation Forest** — ensemble anomaly detection alongside the autoencoders for robust scoring
- **FAISS vector index** — similarity search across 5M+ historical events for campaign clustering and evidence enrichment
- **Campaign detection** — DBSCAN-based clustering groups related anomalous events into attack campaigns
- **Noise suppression** — configurable signature ID blocklists, agent name filters, mDNS/multicast suppression, and machine account filtering
- **Evidence blocks** — every alert includes a rich evidence object with MITRE ATT&CK mapping, baseline comparison, reconstruction error, and historical context
- **Live SOC dashboard** — real-time web dashboard with endpoint analytics, user risk leaderboard, campaign timeline, and expandable evidence panels
- **Incremental retraining** — daily retrain pipeline that processes only new log files and skips already-processed ones

---

## How It Works

### 1. Data Pipeline

Raw Wazuh/Fortigate logs are normalized and enriched by the CyberSentinel Kafka pipeline on 222. The enricher produces rotated `.jsonl.gz` files with fields like:

```json
{
  "subject": { "name": "jdoe", "ip": "10.0.0.5" },
  "security": { "signature_id": "92058", "severity": 12 },
  "enrich": { "anomalies": { "is_brute_force": false }, "risk_score": 50 }
}
```

### 2. Feature Extraction

`ueba_preprocessor.py` extracts a fixed-length feature vector per event covering:
- Temporal features (hour, day, is_weekend, is_after_hours)
- Network features (unique destinations, ports, protocol)
- Behavioral counters (events_5m, failures_5m, brute_force flags)
- Severity and signature metadata

### 3. Anomaly Detection (Dual Model)

Each event is scored by two models:

| Model | Role |
|---|---|
| **Per-user Autoencoder** | Measures reconstruction error vs. the user's personal baseline. High error = behavior deviates from normal. |
| **Isolation Forest** | Global anomaly detector trained on all users. Flags rare/isolated feature combinations. |

The combined score is a weighted blend: `0.6 × AE_deviation + 0.4 × IF_score`.

### 4. Campaign Clustering

Every 5 minutes, the campaign detector runs DBSCAN over the FAISS index of recent anomalous events. Events that cluster together are assigned a campaign ID (e.g. `CAMP-0042`), enabling multi-user, multi-host attack pattern detection.

### 5. Evidence Generation

Every alert includes a structured evidence block:

```json
{
  "evidence": {
    "signature": { "rule_id": "92058", "mitre_ids": ["T1546.011"], "mitre_tactics": ["Persistence"] },
    "raw_event":  { "logon_type": "RemoteInteractive (RDP)", "failures_5m": 0 },
    "baseline":   { "error_vs_threshold": "6.5x above threshold", "model_used": "jdoe" },
    "history":    { "first_seen": "...", "total_events_seen": 47000, "alerts_today": 3 }
  }
}
```

---

## Project Structure

```
aditya_ueba/
├── ueba_engine.py              # Main inference engine — reads enriched.jsonl, writes alerts
├── ueba_preprocessor.py        # Feature extraction, noise suppression, user profile store
├── ueba_trainer.py             # PyTorch autoencoder training loop
├── ueba_prepare_training_data.py # Data prep: decompress → normalize → enrich → HDF5
├── retrain.py                  # Orchestrates full retrain pipeline (data prep + training)
├── enrich_parallel.py          # L2 enrichment with 40 parallel workers
├── ueba_dashboard_server.py    # Flask API server for the dashboard
├── ueba_sync_bridge.sh         # Bash bridge: pull enriched logs from 222, push alerts back
├── ueba_config.yaml            # Engine configuration
├── agents.json                 # Static registry of all Wazuh agents (name, IP, OS)
├── dashboard/
│   └── index.html              # Single-file SOC dashboard (vanilla JS, SVG charts)
├── models/
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   └── autoencoders/
│       ├── global.pt           # Fallback model for unknown users
│       └── <username>.pt       # Per-user models
├── profiles/
│   └── user_profiles.db        # SQLite: per-user baseline stats
├── vector_db/
│   ├── faiss_index.bin         # FAISS IVF index
│   └── faiss_metadata.pkl      # Event metadata for index vectors
└── data/
    └── features.h5             # HDF5 training features (symlink to /data/ueba_training/)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training; inference runs on CPU)
- Wazuh SIEM with enriched log output
- SSH access between SOC machine (222) and GPU server (98)

### Install Dependencies

```bash
cd /root/NEW_DRIVE/aditya_ueba
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu scikit-learn h5py flask flask-cors numpy pandas tqdm autossh
```

### Set Up Reverse SSH Tunnel

On the SOC machine (222), generate a tunnel key and authorize it on 98:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_tunnel -N ""
ssh-copy-id -i ~/.ssh/id_tunnel.pub root@<GPU_SERVER_IP>
# Ensure /root has 755 permissions on 98 (required for PubkeyAuthentication)
ssh root@<GPU_SERVER_IP> "chmod 755 /root"
```

Start the tunnel from 222:

```bash
nohup autossh -M 0 -N \
  -o "ServerAliveInterval=30" \
  -o "ExitOnForwardFailure=yes" \
  -i ~/.ssh/id_tunnel \
  -R 2222:localhost:22 \
  root@<GPU_SERVER_IP> > logs/tunnel.log 2>&1 &
```

Verify from 98:

```bash
ssh -p 2222 -i ~/.ssh/id_ueba soc@localhost "echo tunnel OK"
```

---

## Configuration

`ueba_config.yaml` controls all engine parameters:

```yaml
enriched_input:   "/root/NEW_DRIVE/aditya_ueba/enriched.jsonl"
ueba_output:      "/root/NEW_DRIVE/aditya_ueba/ueba_alerts.jsonl"
alert_threshold:  0.75        # Combined score threshold for alerting
max_index_size:   7000000     # Max FAISS vectors before index is full

noise_suppression:
  signature_ids:
    - "67028"   # Special privileges assigned to new logon (RDP FP)
    - "92058"   # sdbinst.exe — Windows Update compat DB
    - "120052"  # Elevated process — Windows Defender scheduler
    - "92653"   # RDP logon on SocSRV_15
  suppress_source_ips:
    - "164.52.194.98"   # Engine server's own tunnel traffic

models:
  isolation_forest: "models/isolation_forest.pkl"
  scaler:           "models/scaler.pkl"
  autoencoder_dir:  "models/autoencoders/"
  faiss_index:      "vector_db/faiss_index.bin"
  faiss_metadata:   "vector_db/faiss_metadata.pkl"
```

---

## Running the Stack

### On the GPU Server (98)

```bash
cd /root/NEW_DRIVE/aditya_ueba
source venv/bin/activate

# 1. Start sync bridge (pull enriched logs, push alerts to 222)
pkill -f ueba_sync_bridge.sh 2>/dev/null
nohup ./ueba_sync_bridge.sh > logs/sync_bridge.log 2>&1 &

# 2. Start inference engine
pkill -SIGTERM -f ueba_engine.py 2>/dev/null; sleep 2
> ueba_alerts.jsonl
nohup python3 ueba_engine.py --config ueba_config.yaml > logs/ueba_engine.log 2>&1 &

# 3. Start dashboard
pkill -f ueba_dashboard_server.py 2>/dev/null
nohup python3 ueba_dashboard_server.py > logs/dashboard.log 2>&1 &
```

**Dashboard URL:** `http://<GPU_SERVER_IP>:5001`

### Status Check

```bash
tail -f logs/ueba_engine.log        # Engine stats every 60s
tail -f logs/sync_bridge.log        # Sync bridge pull/push status
grep "Stats" logs/ueba_engine.log   # Alert rate summary
wc -l ueba_alerts.jsonl             # Total alerts generated
```

---

## Dashboard

The dashboard is a single-file vanilla JS application served by Flask. No build step required.

| Tab | Description |
|---|---|
| **Overview** | Stat cards (clickable → filtered feed), top risk users, endpoint status panel |
| **Alert Feed** | Full alert table with expandable evidence panels, verdict filters |
| **User Risk** | Risk index leaderboard across all users |
| **Campaigns** | Detected attack campaigns with timeline mini-charts |
| **Endpoints** | Per-endpoint analytics: alert timeline bar chart, verdict donut, top rules, top users |

**Time Window Filter** — filters all tabs to a rolling time window: `24H / 7D / 30D / 90D / ALL`

**Clicking any alert** on the Overview tab navigates directly to the Alert Feed and expands that alert's evidence block.

---

## Retraining

Retraining builds new models from accumulated Wazuh log files. The pipeline:

1. **Data Prep** — decompresses `.json.gz` files, runs L1 normalization + L2 enrichment, outputs `features.h5`
2. **Training** — trains per-user autoencoders (minimum 500 events/user) + global fallback + Isolation Forest

```bash
# Copy new log files to training_zips/
scp -i ~/.ssh/id_tunnel \
  /var/ossec/logs/alerts/2026/Mar/ossec-alerts-{16,17,18}.json.gz \
  root@<GPU_SERVER_IP>:/root/NEW_DRIVE/aditya_ueba/training_zips/

# Run retrain on 98
nohup /data/aditya_ueba/venv/bin/python3 retrain.py \
  --input training_zips/ \
  --output models/ \
  > logs/retrain.log 2>&1 &

tail -f logs/retrain.log
```

The retrain script automatically skips already-processed files (tracked by filename) and backs up the previous models before overwriting.

After retraining, restart the engine:

```bash
source venv/bin/activate
pkill -SIGTERM -f ueba_engine.py; sleep 3
rm -f .state/ueba.state
> ueba_alerts.jsonl
nohup python3 ueba_engine.py --config ueba_config.yaml > logs/ueba_engine.log 2>&1 &
```

---

## Noise Suppression

The preprocessor (`ueba_preprocessor.py`) applies several suppression layers before an event reaches the scoring models:

| Suppression | Description |
|---|---|
| `_NOISE_SIGNATURE_IDS` | Blocks specific Wazuh rule IDs known to be false positives |
| `_SUPPRESS_AGENT_NAMES` | Blocks events from specific agents (e.g. `cybersentinel-manager`) |
| Machine accounts (`$`) | Any `subject.name` ending in `$` is skipped (Windows machine accounts) |
| `NT AUTHORITY\SYSTEM` | SYSTEM-account process events are suppressed |
| mDNS / multicast | `224.x.x.x` destinations and port 5353 traffic are suppressed |
| Source IP suppression | Engine server's own IP is suppressed to avoid tunnel traffic alerts |
| Zero-anomaly None-user | Events with no user and no anomaly flags are dropped |

To add new suppressions, edit the `_NOISE_SIGNATURE_IDS` or `_SUPPRESS_AGENT_NAMES` sets in `ueba_preprocessor.py` and restart the engine.

---

## Deployment Notes

- The GPU server (`164.52.194.98`) must have **port 5001** open for the dashboard
- The SOC machine (222) must have **autossh** running to maintain the reverse tunnel
- The engine resumes from its last file position on restart (stored in `.state/ueba.state`) — delete this file to restart from the end of the current enriched.jsonl
- The FAISS index holds up to `max_index_size` vectors; when full, similarity search still works but new vectors are not added until after a retrain
- Model backups are kept automatically in `models_backup_<timestamp>/` during retraining

---

## Built With

- [PyTorch](https://pytorch.org/) — autoencoder training and inference
- [FAISS](https://github.com/facebookresearch/faiss) — billion-scale similarity search
- [scikit-learn](https://scikit-learn.org/) — Isolation Forest, DBSCAN
- [Wazuh](https://wazuh.com/) — SIEM and log collection
- [Flask](https://flask.palletsprojects.com/) — dashboard API server
- [h5py](https://www.h5py.org/) — HDF5 training data storage

---

## Author

**Aditya Umathe** — CyberSentinel UEBA  
[github.com/AdityaUmathe/cybersentinal_ueba](https://github.com/AdityaUmathe/cybersentinal_ueba)

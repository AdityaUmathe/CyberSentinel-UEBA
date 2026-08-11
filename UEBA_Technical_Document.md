<div align="center">

# CyberSentinel UEBA
### User and Entity Behavior Analytics — Technical Project Document

*Streaming behavioral detection engine for the CyberSentinel SOC platform*

**Version 1.0**  |  **May 2026**

</div>

---

## 1. Project Overview

**User and Entity Behavior Analytics (UEBA)** is a cybersecurity discipline that models the normal activity of users and devices on a network and flags meaningful deviations from that baseline — the class of threats that signature-based tools (firewalls, IDS, traditional SIEM rules) consistently miss.

**CyberSentinel UEBA** is a streaming detection engine layered on top of the SOC's existing log pipeline. It continuously ingests enriched security events, scores every event against learned behavioral baselines, and emits prioritized alerts to analysts.

> The objective is to surface **insider threats, account compromise, and slow-burn attacks** that do not match any static rule, by reasoning over *who* is doing *what*, *when*, and *how often* — rather than analyzing log lines in isolation.

---

## 2. Core Features

| #   | Feature                          | Description                                                                          |
| :-: | :------------------------------- | :----------------------------------------------------------------------------------- |
|  1  | **User Behavior Monitoring**     | Per-user autoencoders learn each account's typical activity envelope.                |
|  2  | **Entity & Device Tracking**     | Host, IP, and process-level rolling counters capture device-side deviations.         |
|  3  | **Multi-Layer Anomaly Detection**| Isolation Forest + autoencoders + RAG similarity fused into a single 0–1 score.      |
|  4  | **Threat-Intelligence Tagging**  | TOR exits, malicious IPs, cloud attribution, and impossible-travel geolocation.      |
|  5  | **High-Throughput Ingestion**    | `orjson` + 24-worker multiprocessing into a 35-feature HDF5 matrix.                  |
|  6  | **Risk Scoring**                 | Weighted fusion of IF and AE scores with rule-derived anomaly boosts.                |
|  7  | **Tiered Alerting**              | Verdicts: `suspicious`, `anomalous`, `highly_anomalous` with MITRE tactic context.   |
|  8  | **Live Dashboard**               | Flask + React UI with SSE feed, risk leaderboard, and embedded AI Analyst.           |
|  9  | **Automated Correlation**        | HDBSCAN groups related alerts into *campaigns* every five minutes.                   |

---

## 3. Working of the UEBA Engine

The engine operates as a continuous seven-stage pipeline:

**1. Log Ingestion** — Tails `enriched.jsonl` produced by the upstream SOC pipeline over an SSH bridge tunnel; events are read incrementally with state checkpointing for crash safety.

**2. Parsing & Normalization** — Events are parsed with `orjson`. Thirty-five features (temporal, event-flag, anomaly, counter, behavioral, geo, risk, network, categorical) are extracted via a config-driven field map and standardized with a persisted `StandardScaler`.

**3. Baseline Creation** *(offline)* — The trainer builds three artifacts from a multi-day historical window:
&nbsp;&nbsp;&nbsp;&nbsp;• A global **Isolation Forest**
&nbsp;&nbsp;&nbsp;&nbsp;• A global **autoencoder** with architecture `35 → 24 → 12 → 24 → 35`
&nbsp;&nbsp;&nbsp;&nbsp;• **Per-user autoencoders** for every account meeting `min_events_per_user`

**4. Detection Logic** — Each live event is scored by:
&nbsp;&nbsp;&nbsp;&nbsp;**(a)** the Isolation Forest score,
&nbsp;&nbsp;&nbsp;&nbsp;**(b)** the user's personal autoencoder (or global fallback) reconstruction error, normalized against its p95 training threshold,
&nbsp;&nbsp;&nbsp;&nbsp;**(c)** rule-based anomaly flags from the upstream enricher.

**5. Correlation** — A periodic HDBSCAN pass over recent alert vectors groups related events into **campaigns**; a FAISS `IndexFlatIP` index supports RAG-style retrieval of historically similar incidents.

**6. Risk Calculation** — Scores are fused as:

> `combined_score = 0.5·IF + 0.5·AE`, clipped to `[0, 1]`
>
> **Verdict bands:** `0.75–0.85` suspicious · `0.85–0.93` anomalous · `0.93–1.00` highly anomalous

**7. Alert Triggering** — Events with `combined_score ≥ 0.75` are written to the alert store with anomaly reasons, raw evidence, baseline deltas, and campaign ID. The dashboard streams them live over Server-Sent Events.

---

## 4. System Architecture

| Layer               | Components                                          | Responsibility                                                       |
| :------------------ | :-------------------------------------------------- | :------------------------------------------------------------------- |
| **Frontend**        | React + Vite SPA                                    | SSE live feed, REST snapshots, false-positive workflow.              |
| **Backend API**     | Flask · `ueba_dashboard_server.py` *(port 3026)*    | Serves `/api/feed`, `/api/stats`, `/api/users`, `/api/stream`, etc.  |
| **Detection Plane** | `ueba_engine.py` · `ueba_trainer.py` · `ueba_models/` | GPU-resident inference, offline training, feature extraction.        |
| **Storage**         | SQLite · HDF5 · JSONL · FAISS · pickled artifacts   | Profiles, feature matrix, raw/alert events, vector index, scaler.    |
| **Integration**     | Upstream SOC pipeline *(Wazuh + custom enrichers)*  | Consumes enriched events via SSH tunnel; emits alerts to dashboard.  |

---

## 5. Technologies Used

| Domain                 | Stack                                                                          |
| :--------------------- | :----------------------------------------------------------------------------- |
| **Language & Runtime** | Python 3.12                                                                    |
| **Machine Learning**   | PyTorch *(CUDA on NVIDIA L40S)*, scikit-learn, HDBSCAN                         |
| **Vector Search**      | FAISS-GPU                                                                      |
| **Data & Storage**     | h5py (HDF5), NumPy, SQLite, `orjson`, joblib                                   |
| **Backend**            | Flask, flask-cors, Server-Sent Events                                          |
| **Frontend**           | React, Vite                                                                    |
| **AI Integration**     | Anthropic Claude API *(AI-assisted alert triage)*                              |
| **Operations**         | YAML configuration, cron-driven shell wrappers for rotation & weekly retrain   |

---

## 6. Real-World Use Cases

| Scenario                       | Detection Path                                                                  |
| :----------------------------- | :------------------------------------------------------------------------------ |
| **Insider Threats**            | Privileged access outside `typical_hours` baseline trips the user autoencoder.  |
| **Account Compromise**         | Impossible-travel + TOR-exit credential success → `highly_anomalous` verdict.   |
| **Abnormal Login Behavior**    | Brute-force at 03:00 against a 09:00–18:00 baseline spikes the AE error.        |
| **Data Exfiltration**          | Surge in `unique_destinations_1h` outbound raises IF score; tagged accordingly. |
| **Suspicious Endpoint Activity**| PowerShell drops + lateral-movement signatures clustered into one HDBSCAN campaign. |
| **Privilege Misuse**           | Service accounts running interactive commands deviate from `typical_agents`.    |

---

<div align="center">

*CyberSentinel UEBA — Behavioral Threat Detection for the Modern SOC*

</div>

#!/usr/bin/env python3
"""
stream_synthetic_alerts.py
──────────────────────────
Continuously append synthetic alerts to ueba_alerts.jsonl at a fixed rate so
the dashboard's SSE /api/stream endpoint has something to push.

This is a development substitute for the real engine writing to the same file
on the GPU server. Each appended alert reuses the same schema (and the same
``make_alert`` helper) as ``gen_synthetic_alerts.py``, so every dashboard
panel renders correctly.

Usage:
    python3 tools/stream_synthetic_alerts.py
    python3 tools/stream_synthetic_alerts.py --rate 1.0          # 1 alert / 1s
    python3 tools/stream_synthetic_alerts.py --rate 0.5 --output ueba_alerts.jsonl
    python3 tools/stream_synthetic_alerts.py --campaign-burst 8  # burst into one campaign
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Import schema helpers from the batch generator that lives in the same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_synthetic_alerts import make_alert  # noqa: E402


def _now_alert(rng: random.Random, campaign_id: str | None) -> dict:
    """Build one alert stamped with the current UTC time."""
    a = make_alert(rng, datetime.now(timezone.utc), campaign_id, alerts_today=1)
    # make_alert uses uuid4 for event_id already; nothing else to fix up.
    return a


def main() -> None:
    p = argparse.ArgumentParser(description="Append synthetic alerts to a JSONL file at a fixed rate.")
    p.add_argument("--output", default="ueba_alerts.jsonl",
                   help="Path to append to (default: ./ueba_alerts.jsonl)")
    p.add_argument("--rate", type=float, default=1 / 3,
                   help="Alerts per second (default 0.33 = one alert every 3s)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; omit for time-based seed")
    p.add_argument("--campaign-burst", type=int, default=6,
                   help="Average size of campaign bursts (alerts share a CAMP-XXXX id). "
                        "Set to 0 to never assign a campaign.")
    p.add_argument("--campaign-prob", type=float, default=0.30,
                   help="Probability a non-campaign alert starts a new campaign burst")
    args = p.parse_args()

    if args.rate <= 0:
        p.error("--rate must be > 0")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / args.rate
    rng = random.Random(args.seed if args.seed is not None else int(time.time()))

    print(f"streaming → {out}   rate={args.rate:.2f}/s   interval={interval:.2f}s   "
          f"(Ctrl-C to stop)")

    cid_num   = 1
    burst_rem = 0
    cid       = None

    try:
        with out.open("a", encoding="utf-8") as f:
            while True:
                # Manage campaign-burst grouping
                if burst_rem > 0:
                    burst_rem -= 1
                    use_cid = cid
                elif args.campaign_burst > 0 and rng.random() < args.campaign_prob:
                    cid = f"CAMP-{cid_num:04d}"
                    cid_num += 1
                    burst_rem = max(1, int(rng.gauss(args.campaign_burst, args.campaign_burst / 3))) - 1
                    use_cid = cid
                else:
                    use_cid = None

                alert = _now_alert(rng, use_cid)
                f.write(json.dumps(alert, default=str) + "\n")
                f.flush()
                eid     = alert["event_id"]
                verdict = alert["ueba"]["risk_verdict"]
                user    = alert["subject"]["name"]
                host    = alert["host"]["name"]
                sig     = alert["security"]["signature"]
                print(f"  {datetime.now().strftime('%H:%M:%S')}  {eid}  {verdict:<16s}  "
                      f"{user:<22s}  {host:<16s}  {sig[:50]}"
                      + (f"  [{use_cid}]" if use_cid else ""),
                      flush=True)

                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()

# UEBA operations scripts

The UEBA pipeline depends on a handful of host-level scripts and unit/config
files that live **outside the application code** on two servers. They were
previously deploy-only (hand-edited on the boxes, hard-coded paths, not in any
repo) — so a server rebuild would lose them. This directory vendors sanitized,
documented copies so they survive and can be re-deployed.

> These are **reference copies**, not auto-deployed. Concrete public IPs and the
> fail2ban allow-list are replaced with `<PLACEHOLDERS>`; fill them in at deploy
> time from the private ops runbook. No secrets/keys are committed.

## Topology

The feed flows from the SOC host to the engine host over a reverse SSH tunnel:

```
222 (SOC, NAT'd)                         98 (GPU / engine, public)
─────────────────                        ─────────────────────────
enricher → enriched_*.jsonl[.gz]
        │
        │  ueba-tunnel.service
        │  (autossh -R 2222:localhost:22) ───────► binds 98 localhost:2222 → 222:22
                                                   │
                                   ueba_sync_bridge.sh  (soc@localhost:2222)
                                   pulls chunks + tails active file
                                                   │
                                                   ▼
                                          enriched.jsonl  ──► ueba_engine.py
                                                                   │
                                                            ueba_alerts.jsonl
                                                                   │
                                                          ueba_dashboard_server.py (:3026)
```

222 is NAT'd and not reachable inbound, so the tunnel is **established by 222**;
98 only consumes `localhost:2222`.

## Files

### `98-engine/` — the GPU/engine host
| File | Deploy to | Purpose |
|------|-----------|---------|
| `ueba_rotate.sh` | `/root/NEW_DRIVE/aditya_ueba/` (cron 00:05) | Archive alerts, flush `enriched.jsonl` (size/age/backlog-guarded), prune archives, restart engine. |
| `ueba_sync_bridge.sh` | `/root/NEW_DRIVE/aditya_ueba/` (long-running) | Pull enriched chunks + tail the active file from 222 over the tunnel; push alerts back. |
| `sshd/10-ueba-maxstartups.conf` | `/etc/ssh/sshd_config.d/` | Raise `MaxStartups` so an SSH brute-force flood can't starve the tunnel. `sshd -t && systemctl reload ssh`. |
| `fail2ban/jail.local` | `/etc/fail2ban/jail.local` | Ban SSH brute-forcers at the local nftables firewall. Fill in `ignoreip`. `systemctl restart fail2ban`. |
| `crontab.example` | `crontab -e` | The two UEBA cron lines (daily rotate, weekly retrain). |
| `systemd/ueba-engine.service` | `/etc/systemd/system/` | Run the engine under systemd so it auto-starts on boot + auto-restarts on crash. `systemctl daemon-reload && systemctl enable --now ueba-engine`. |
| `systemd/ueba-dashboard.service` | `/etc/systemd/system/` | Same for the dashboard server (`:3026`). `systemctl enable --now ueba-dashboard`. |
| `systemd/ueba-bridge.service` | `/etc/systemd/system/` | Same for the feed bridge (`ueba_sync_bridge.sh`) — without this a reboot silently freezes the feed. `systemctl enable --now ueba-bridge`. |

### `222-soc/` — the SOC host
| File | Deploy to | Purpose |
|------|-----------|---------|
| `ueba-tunnel.service` | `/etc/systemd/system/` | Self-healing autossh reverse tunnel. `systemctl enable --now`. Fill in `<ENGINE_98_PUBLIC_IP>`. |

## Recurring failure mode & runbook

"Zero logs on the dashboard" almost always means the **feed tunnel is down**,
not a dashboard bug. Fast triage (no SSH needed):

```
curl 'http://<ENGINE_98>:3026/api/stats?hours=0'   # All-time
curl 'http://<ENGINE_98>:3026/api/stats?hours=24'  # last 24h
```

If All-time > 0 but 24h == 0, the feed is frozen. On 98:

```
ss -ltnp | grep 2222                  # tunnel listener present?
ss -tnp  | grep :2222                 # 222's current egress IP (for ignoreip!)
tail -f /root/NEW_DRIVE/aditya_ueba/logs/sync_bridge.log
```

The dashboard now also shows a **stale-feed banner** automatically when the
newest alert ages past 15m (amber) / 60m (red), so a frozen feed is visible
without manual checking.

**If the dashboard is fully unreachable (not just empty) after a 98 reboot:**
the engine + dashboard run under systemd (`ueba-engine`, `ueba-dashboard`) and
auto-start on boot. Check `systemctl status ueba-engine ueba-dashboard`; if they
were started manually before these units existed, install them from
`98-engine/systemd/`. Confirm boot freshness with `uptime` — a recent boot that
matches the feed-freeze time means a reboot took out manually-launched processes.

### Gotchas
- **autossh needs `-i /root/.ssh/id_tunnel`** — root's default keys are absent on
  222, so without it autossh falls back to password auth and flaps every ~10s.
- **222's egress IP is dynamic (NAT)** — if the tunnel can't reconnect after an
  IP change, check it isn't being blocked and update `ignoreip` in `jail.local`
  with the new `ss -tnp | grep :2222` source.
- **`enriched.jsonl` is read by byte offset** (`.state/ueba.state`) with no
  truncation detection — only an in-place `: > file` + state-clear is safe to
  flush it. That's exactly what `ueba_rotate.sh` does; never `mv`/rotate it.

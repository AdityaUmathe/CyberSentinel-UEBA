// Time-window filter (24H / 3D / 7D / 30D / 60D / 90D / ALL).
// Rebuilds feedData / usersData / agentsData from the immutable `all*` arrays
// then re-renders every dependent panel + chart.

import { state } from "../state.js";
import { renderFeed } from "./feed.js";
import { renderUsers, renderOverviewUsers } from "./users.js";
import { renderAgents, renderOverviewAgents } from "./agents.js";
import { updateGauge } from "../charts/gauge.js";
import { drawTimeline } from "../charts/timeline.js";
import { renderScoreDist } from "../charts/scoredist.js";
import { renderHeatmap } from "../charts/heatmap.js";
import { renderTopSigs } from "../charts/topsigs.js";

export function applyTimelineFilter() {
  const hours = state.timelineHours;
  const label =
    {
      0: "ALL TIME",
      24: "LAST 24H",
      72: "LAST 3 DAYS",
      168: "LAST 7 DAYS",
      720: "LAST 30 DAYS",
      1440: "LAST 60 DAYS",
      2160: "LAST 90 DAYS",
    }[hours] || "";
  const showingEl = document.getElementById("tl-showing");
  if (showingEl) showingEl.textContent = label;

  if (hours === 0) {
    state.feedData   = state.allFeedData;
    state.usersData  = state.allUsersData;
    state.agentsData = state.allAgentsData;
  } else {
    const cutoff = new Date(Date.now() - hours * 3600000);
    state.feedData = state.allFeedData.filter((a) => {
      try {
        return new Date(a.processed_at || a.event_time) >= cutoff;
      } catch {
        return true;
      }
    });
    const uMap = {};
    state.feedData.forEach((a) => {
      const u = a.user || "unknown";
      if (!uMap[u])
        uMap[u] = { user: u, alert_count: 0, max_score: 0, verdicts: {}, hosts: [], top_reason: "", last_seen: "", risk_index: 0 };
      uMap[u].alert_count++;
      uMap[u].max_score = Math.max(uMap[u].max_score, a.score || 0);
      if (a.host && !uMap[u].hosts.includes(a.host)) uMap[u].hosts.push(a.host);
      if (!uMap[u].last_seen || a.processed_at > uMap[u].last_seen) uMap[u].last_seen = a.processed_at;
      const v = a.verdict || "";
      uMap[u].verdicts[v] = (uMap[u].verdicts[v] || 0) + 1;
      if (a.reasons && a.reasons[0]) uMap[u].top_reason = a.reasons[0];
    });
    state.usersData = Object.values(uMap)
      .map((u) => {
        const ha = u.verdicts["highly_anomalous"] || 0;
        u.risk_index = parseFloat(
          (u.max_score * 0.5 + (u.alert_count / 100) * 0.3 + (ha / Math.max(u.alert_count, 1)) * 0.2).toFixed(3)
        );
        u.top_verdict = Object.entries(u.verdicts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
        u.highly_anomalous = ha;
        return u;
      })
      .filter((u) => u.user !== "unknown")
      .sort((a, b) => b.risk_index - a.risk_index);

    const aMap = {};
    state.feedData.forEach((a) => {
      const h = a.host || "";
      if (!h) return;
      if (!aMap[h])
        aMap[h] = { agent: h, ip: a.host_ip || "", alert_count: 0, max_score: 0, highly_anomalous: 0, anomalous: 0, suspicious: 0, last_seen: "" };
      aMap[h].alert_count++;
      aMap[h].max_score = Math.max(aMap[h].max_score, a.score || 0);
      if (a.verdict === "highly_anomalous") aMap[h].highly_anomalous++;
      else if (a.verdict === "anomalous")    aMap[h].anomalous++;
      else if (a.verdict === "suspicious")   aMap[h].suspicious++;
      if (!aMap[h].last_seen || a.processed_at > aMap[h].last_seen) aMap[h].last_seen = a.processed_at;
    });
    state.agentsData = Object.values(aMap).sort(
      (a, b) => b.highly_anomalous - a.highly_anomalous || b.max_score - a.max_score
    );
  }

  renderFeed();
  renderUsers();
  renderOverviewUsers();
  renderAgents();
  renderOverviewAgents();

  if (state.lastStats) {
    updateGauge(state.lastStats, state.feedData);
    drawTimeline(state.feedData);
    renderScoreDist(state.feedData);
    renderHeatmap(state.feedData);
    renderTopSigs(state.feedData);
  }
}

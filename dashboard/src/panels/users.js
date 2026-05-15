// User risk leaderboard (main tab + Overview top-N panel).

import { state } from "../state.js";
import { fmtDate, scoreColor } from "../helpers.js";

export function renderUsers() {
  document.getElementById("users-badge").textContent = state.usersData.length;
  if (!state.usersData.length) {
    document.getElementById("users-list").innerHTML =
      '<div class="empty-state"><p>NO USER DATA</p></div>';
    return;
  }
  document.getElementById("users-list").innerHTML = state.usersData.map((u, i) => {
    const rankCls = i === 0 ? "top1" : i < 3 ? "top3" : i < 5 ? "top5" : "";
    const riColor = u.risk_index > 0.7 ? "var(--red)" : u.risk_index > 0.5 ? "var(--orange)" : "var(--yellow)";
    const safeName = u.user.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    const activeCls = state.selectedUser && state.selectedUser === u.user ? " active" : "";
    return `<div class="user-row clickable${activeCls}" onclick="loadUserDetail('${safeName}')">
      <div class="rank-num ${rankCls}">${i + 1}</div>
      <div class="user-name">${u.user}</div>
      <div class="risk-index-cell"><span class="risk-index-val" style="color:${riColor}">${u.risk_index.toFixed(3)}</span></div>
      <div class="count-val">${u.alert_count}</div>
      <div class="max-score-val" style="color:${scoreColor(u.max_score)}">${u.max_score.toFixed(3)}</div>
      <div class="top-reason-val">${(u.top_reason || "—").replace(/_/g, " ")}</div>
      <div class="last-seen-val">${fmtDate(u.last_seen)}</div>
    </div>`;
  }).join("");
}

export function renderOverviewUsers() {
  document.getElementById("overview-users").innerHTML =
    state.usersData.slice(0, 8).map((u, i) => {
      const riColor = u.risk_index > 0.7 ? "var(--red)" : u.risk_index > 0.5 ? "var(--orange)" : "var(--yellow)";
      const rankCls = i === 0 ? "top1" : i < 3 ? "top3" : "";
      const hostLabel =
        u.hosts && u.hosts.length ? ` <span style="color:var(--text3)">[${u.hosts[0]}]</span>` : "";
      return `<div class="user-row-compact">
        <div class="urc-rank ${rankCls}">${i + 1}</div>
        <div class="urc-name">${u.user}${hostLabel}</div>
        <div class="urc-score" style="color:${riColor}">${u.risk_index.toFixed(3)}</div>
        <div class="urc-count">${u.alert_count} alerts</div>
      </div>`;
    }).join("") || '<div class="empty-state"><p>NO DATA</p></div>';
}

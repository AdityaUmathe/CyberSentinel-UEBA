// Recommended actions panel — priority pill + numbered action list.

export function renderRecommendedActions(stats, score) {
  const body = document.getElementById("rec-actions-body");
  const pill = document.getElementById("rec-priority-pill");
  if (!body) return;

  let priority = "NORMAL", priorityClass = "normal";
  let actions = [];

  if (score >= 75) {
    priority = "CRITICAL"; priorityClass = "high";
    actions = [
      "Immediately isolate affected endpoints from network",
      "Escalate all highly_anomalous alerts to Tier 3 analyst",
      "Enable enhanced logging on SocSRV_15 and Administrator accounts",
      "Review and rotate credentials for flagged accounts",
      "Initiate incident response playbook",
    ];
  } else if (score >= 50) {
    priority = "ELEVATED"; priorityClass = "elevated";
    actions = [
      "Investigate anomalous users identified in leaderboard",
      "Review campaign clusters for coordinated attack patterns",
      "Check SeTcbPrivilege escalation events on VGSOCSRV",
      "Validate ANONYMOUS LOGON NTLM events against expected sources",
      "Schedule security review within 24 hours",
    ];
  } else if (score >= 20) {
    priority = "ELEVATED"; priorityClass = "elevated";
    actions = [
      "Review after-hours activity flagged by UEBA engine",
      "Verify service startup type changes on SocSRV_15",
      "Confirm taskschd.dll loads are from legitimate scheduled tasks",
      "Continue monitoring ANONYMOUS LOGON patterns",
    ];
  } else {
    priority = "NORMAL"; priorityClass = "normal";
    actions = [
      "Maintain current security baseline",
      "Continue routine monitoring",
      "Quarterly security reviews",
      "No immediate action required",
    ];
  }

  if (pill) {
    pill.className = "rec-priority " + priorityClass;
    pill.textContent = "Priority: " + priority;
  }

  body.innerHTML = actions.map((a, i) => `
    <div class="rec-action-row">
      <div class="rec-action-num">${i + 1}</div>
      <div class="rec-action-text">${a}</div>
    </div>`).join("");
}

// Pure formatting helpers — no DOM access, no state.

export function fmtTime(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString("en-IN", {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch {
    return iso.slice(11, 19);
  }
}

export function fmtDate(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return (
      d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" }) +
      " " +
      d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })
    );
  } catch {
    return iso.slice(0, 16);
  }
}

export function scoreColor(s) {
  if (s >= 0.9) return "var(--red)";
  if (s >= 0.8) return "var(--orange)";
  return "var(--yellow)";
}

export function verdictClass(v) {
  return "v-" + (v || "").replace(/ /g, "_");
}

export function verdictLabel(v) {
  return (
    { highly_anomalous: "CRITICAL", anomalous: "ANOMALOUS", suspicious: "SUSPICIOUS" }[v] ||
    (v || "").toUpperCase() ||
    "—"
  );
}

export function reasonTag(r) {
  let cls = "";
  if (r.includes("brute")) cls = "r-brute";
  if (r.includes("lateral")) cls = "r-lateral";
  if (r.includes("exfil")) cls = "r-exfil";
  return `<span class="reason-tag ${cls}">${r
    .replace(/_/g, " ")
    .replace("behavioral baseline deviation", "baseline dev")
    .replace("isolation forest", "IF")}</span>`;
}

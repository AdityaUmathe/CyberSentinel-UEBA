// CSV / JSON export of the filtered feed.
//
// `exportCsv(rows)` produces a flat, analyst-friendly schema (no nested
// evidence) suitable for opening in Excel/Sheets.
// `exportJson(rows)` exports the full feed-item shape including evidence so
// the file can round-trip back into another tool.

const CSV_COLUMNS = [
  "processed_at", "event_id", "user", "host", "host_ip",
  "verdict", "score", "severity", "campaign_id",
  "signature_id", "signature", "reasons", "mitre_tactic",
  "fp_reason",
];

function _csvEscape(v) {
  if (v === null || v === undefined) return "";
  const s = Array.isArray(v) ? v.join("; ") : String(v);
  // Quote if it contains comma, quote, newline, or leading/trailing whitespace
  if (/[",\n\r]/.test(s) || s !== s.trim()) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function _row(a) {
  return {
    processed_at: a.processed_at || "",
    event_id:     a.event_id     || "",
    user:         a.user         || "",
    host:         a.host         || "",
    host_ip:      a.host_ip      || "",
    verdict:      a.verdict      || "",
    score:        (a.score || 0).toFixed(4),
    severity:     a.severity ?? "",
    campaign_id:  a.campaign_id  || "",
    signature_id: a.signature_id || "",
    signature:    a.signature    || "",
    reasons:      (a.reasons || []).join("; "),
    mitre_tactic: (a.mitre_tactic || []).join("; "),
    fp_reason:    (a.fp && a.fp.reason) || "",
  };
}

function _download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 0);
}

function _timestamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return (
    d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) +
    "-" +
    p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds())
  );
}

export function exportCsv(rows) {
  const head = CSV_COLUMNS.join(",");
  const body = rows
    .map((a) => {
      const r = _row(a);
      return CSV_COLUMNS.map((c) => _csvEscape(r[c])).join(",");
    })
    .join("\n");
  const csv = head + "\n" + body + "\n";
  _download(new Blob([csv], { type: "text/csv;charset=utf-8" }), `ueba-export-${_timestamp()}.csv`);
}

export function exportJson(rows) {
  const payload = {
    generated_at: new Date().toISOString(),
    count:        rows.length,
    alerts:       rows,
  };
  _download(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    `ueba-export-${_timestamp()}.json`
  );
}

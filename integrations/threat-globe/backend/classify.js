// Firewall-event classification — JS port of the Python _classify_firewall().
// Pure, no dependencies. Decides the traffic CATEGORY, the arc DIRECTION
// ("in" = external→DC, "out" = DC→external) and which IP is the external party.

// ── private/non-routable IP check (IPv4 + a few IPv6 cases) ──
export function isPublicIp(ip) {
  if (!ip || typeof ip !== "string") return false;
  const s = ip.trim();
  if (s.includes(":")) {                                  // IPv6
    const l = s.toLowerCase();
    if (l === "::1" || l === "::") return false;
    if (l.startsWith("fe80") || l.startsWith("fc") || l.startsWith("fd")) return false;
    return true;
  }
  const p = s.split(".").map(Number);
  if (p.length !== 4 || p.some((n) => Number.isNaN(n) || n < 0 || n > 255)) return false;
  const [a, b] = p;
  if (a === 10) return false;
  if (a === 127) return false;
  if (a === 169 && b === 254) return false;
  if (a === 172 && b >= 16 && b <= 31) return false;
  if (a === 192 && b === 168) return false;
  if (a === 0 || a >= 224) return false;                  // 0.x, multicast, reserved
  return true;
}

// Identify a firewall event in the default UEBA-enriched schema. Override via
// options.isFirewallEvent for your own schema.
export function isFirewallEventDefault(raw) {
  const h = raw.host || {};
  if (h.type === "firewall") return true;
  if (String((h.os || {}).name || "").toLowerCase() === "fortios") return true;
  return String((raw.context || {}).source || "").toLowerCase().includes("fortigate");
}

// Normalise a raw event to the small shape classify() needs. Override via
// options.mapEvent to adapt YOUR SIEM's firewall-log schema.
export function mapEventDefault(raw) {
  return {
    direction: String((raw.network || {}).direction || "").toLowerCase(),
    outcome:   String(raw.event_outcome || "").toLowerCase(),
    sig:       String((raw.security || {}).signature || "Firewall event"),
    subIp:     String((raw.subject || {}).ip || ""),
    objIp:     String((raw.object || {}).ip || ""),
    time:      raw.event_time || raw.ingest_time || null,
  };
}

// Classify a normalised event → { category, dir, extIp }.
export function classify(ev) {
  const sig = (ev.sig || "").toLowerCase();
  const failed = /fail|block|denied|deny|attack|drop|invalid/.test(sig) || /fail/.test(ev.outcome || "");
  const isVpn = sig.includes("vpn");

  if (ev.direction === "outgoing") {
    const ext = isPublicIp(ev.objIp) ? ev.objIp : ev.subIp;
    return { category: "outgoing", dir: "out", extIp: ext };
  }
  // incoming / internal / none / unknown → external party is the source side
  const ext = isPublicIp(ev.subIp) ? ev.subIp : ev.objIp;
  if (isVpn && !failed) return { category: "external_conn", dir: "in", extIp: ext };
  if (failed)           return { category: "incoming_threat", dir: "in", extIp: ext };
  return { category: "normal_incoming", dir: "in", extIp: ext };
}

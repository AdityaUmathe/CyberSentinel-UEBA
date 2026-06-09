// Node geofeed backend for <ThreatGlobe />.
//
// Builds the geofeed JSON the widget consumes: it takes recent FIREWALL events
// (from YOUR SIEM), geolocates the external IP via a local MaxMind GeoLite2 DB
// (fully offline), classifies incoming/outgoing, and returns source↔DC arcs.
//
//   import { createGeofeedHandler } from "threat-globe/backend/geofeed.js";
//   app.get("/api/geofeed", createGeofeedHandler({
//     cityDbPath: "/data/GeoLite2-City.mmdb",
//     asnDbPath:  "/data/GeoLite2-ASN.mmdb",          // optional
//     dcIp: "103.76.143.84",                          // your edge/WAN IP (geolocated as the hub)
//     fetchEvents: async () => mySiem.recentFirewallEvents(2000),   // <-- plug in your source
//     // mapEvent: (raw) => ({ direction, outcome, sig, subIp, objIp, time }),  // if your schema differs
//   }));
//
// Requires the `maxmind` package:  npm i maxmind
// Returns: { geo_ok, dc, arcs[], stats{} }  (see ../README.md for the contract).

import maxmind from "maxmind";
import { classify, isPublicIp, isFirewallEventDefault, mapEventDefault } from "./classify.js";

export function createGeofeedHandler(options = {}) {
  const opts = {
    cityDbPath: "",
    asnDbPath: "",
    maxArcs: 400,
    dcIp: "",
    dcLat: 21.1463, dcLng: 79.0849, dcLabel: "Datacenter",
    isFirewallEvent: isFirewallEventDefault,
    mapEvent: mapEventDefault,
    fetchEvents: null,         // REQUIRED: async () => [rawEvent, ...]
    ...options,
  };

  let cityReader = null, asnReader = null, readersTried = false;
  const geoCache = new Map();          // ip -> geo | null
  let dcLoc = null;

  async function openReaders() {
    if (readersTried) return;
    readersTried = true;
    try { if (opts.cityDbPath) cityReader = await maxmind.open(opts.cityDbPath); }
    catch (e) { console.warn("[threat-globe] City DB open failed:", e.message); }
    try { if (opts.asnDbPath) asnReader = await maxmind.open(opts.asnDbPath); }
    catch (e) { console.warn("[threat-globe] ASN DB open failed:", e.message); }
  }

  function geo(ip) {
    if (geoCache.has(ip)) return geoCache.get(ip);
    let out = null;
    if (cityReader) {
      try {
        const c = cityReader.get(ip);
        const lat = c && c.location && c.location.latitude;
        const lng = c && c.location && c.location.longitude;
        if (lat != null && lng != null) {
          out = {
            lat: Math.round(lat * 1e4) / 1e4, lng: Math.round(lng * 1e4) / 1e4,
            country: (c.country && c.country.iso_code) || "??",
            country_name: (c.country && c.country.names && c.country.names.en) || "Unknown",
            city: (c.city && c.city.names && c.city.names.en) || "",
          };
          if (asnReader) {
            try {
              const a = asnReader.get(ip);
              if (a) { out.asn = a.autonomous_system_number; out.org = a.autonomous_system_organization || ""; }
            } catch (e) { /* asn optional */ }
          }
        }
      } catch (e) { out = null; }
    }
    if (geoCache.size > 50000) geoCache.clear();
    geoCache.set(ip, out);
    return out;
  }

  function resolveDC() {
    if (dcLoc) return dcLoc;
    const g = opts.dcIp && isPublicIp(opts.dcIp) ? geo(opts.dcIp) : null;
    dcLoc = g
      ? { lat: g.lat, lng: g.lng, country: g.country, label: opts.dcLabel }
      : { lat: opts.dcLat, lng: opts.dcLng, country: "", label: opts.dcLabel };
    return dcLoc;
  }

  async function build() {
    await openReaders();
    const dc = resolveDC();
    const geoOk = !!cityReader;
    const arcs = [];
    const byCountry = new Map(), bySig = new Map();
    const byCategory = { incoming_threat: 0, normal_incoming: 0, outgoing: 0, external_conn: 0 };
    const seenIps = new Set();
    let scanned = 0;

    let events = [];
    if (geoOk && typeof opts.fetchEvents === "function") {
      try { events = (await opts.fetchEvents()) || []; }
      catch (e) { console.warn("[threat-globe] fetchEvents failed:", e.message); }
    }

    for (const raw of events) {
      if (!opts.isFirewallEvent(raw)) continue;
      scanned++;
      const ev = opts.mapEvent(raw);
      const { category, dir, extIp } = classify(ev);
      if (!extIp || !isPublicIp(extIp)) continue;
      const g = geo(extIp);
      if (!g) continue;

      arcs.push({
        srcLat: g.lat, srcLng: g.lng, direction: dir, category,
        country: g.country, country_name: g.country_name, city: g.city,
        ip: extIp, org: g.org || "", asn: g.asn || null,
        sig: ev.sig, outcome: ev.outcome, time: ev.time,
      });
      seenIps.add(extIp);
      byCategory[category] = (byCategory[category] || 0) + 1;
      const bc = byCountry.get(g.country) || { country: g.country, country_name: g.country_name, count: 0 };
      bc.count++; byCountry.set(g.country, bc);
      bySig.set(ev.sig, (bySig.get(ev.sig) || 0) + 1);
    }

    const keptArcs = arcs.length > opts.maxArcs ? arcs.slice(-opts.maxArcs) : arcs;
    const topCountries = [...byCountry.values()].sort((a, b) => b.count - a.count).slice(0, 12);
    const topSignatures = [...bySig.entries()].map(([sig, count]) => ({ sig, count }))
      .sort((a, b) => b.count - a.count).slice(0, 8);

    return {
      geo_ok: geoOk, dc, arcs: keptArcs,
      stats: {
        events: scanned, mapped: keptArcs.length, unique_ips: seenIps.size,
        countries: byCountry.size, categories: byCategory,
        top_countries: topCountries, top_signatures: topSignatures,
      },
    };
  }

  // Express-style handler: app.get("/api/geofeed", handler)
  return async function geofeedHandler(req, res) {
    try {
      const payload = await build();
      res.json(payload);
    } catch (e) {
      console.error("[threat-globe] geofeed error:", e);
      res.status(500).json({ geo_ok: false, error: String(e && e.message || e), arcs: [], stats: {} });
    }
  };
}

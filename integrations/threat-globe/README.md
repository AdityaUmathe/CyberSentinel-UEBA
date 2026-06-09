# Threat Globe — portable live firewall threat-map

A self-contained **3D globe** that plots firewall traffic in real time —
direction-aware arcs (incoming/outgoing), attacker points, impact rings, a HUD
(top countries, signatures, live ticker) and clickable category filters. Lifted
out of the CyberSentinel UEBA dashboard so it can drop into another app.

Two halves, fully decoupled by a small JSON contract:

| Half | Tech | What it is |
|------|------|------------|
| `frontend/` | **React (web)** + framework-agnostic engine | `<ThreatGlobe/>` component you `import`. Renders the globe + HUD + filters. Owns its own scoped CSS. |
| `backend/` | **Node.js** | `createGeofeedHandler()` — GeoLite2 geoIP + incoming/outgoing classification, serves `/api/geofeed`. Pluggable log source. |
| `assets/` | static files | `earth-night.jpg`, `night-sky.png`, `countries-110m.geojson` — served offline, no CDN/API key. |

The frontend only needs **any** endpoint that returns the contract below — it
does not care whether that's this Node backend, the original Python one, or a
mock. So you can adopt the two halves independently.

> **React web, not React Native.** This renders WebGL via Three.js, so it needs a
> DOM `<canvas>`. In a React Native app, host it inside a `react-native-webview`
> pointing at a tiny page that mounts `<ThreatGlobe/>`.

---

## 1. The data contract — `GET /api/geofeed`

```jsonc
{
  "geo_ok": true,                       // false → widget shows a "GeoIP unavailable" overlay
  "dc": { "lat": 21.99, "lng": 79.00, "label": "Datacenter" },
  "arcs": [
    {
      "srcLat": 42.69, "srcLng": 23.33, // the EXTERNAL endpoint (already geolocated)
      "direction": "in",                // "in" = external→DC, "out" = DC→external
      "category": "incoming_threat",    // incoming_threat | normal_incoming | outgoing | external_conn
      "country": "BG", "country_name": "Bulgaria", "city": "Sofia",
      "ip": "94.26.69.122", "asn": 209854, "org": "Cyberzone S.A.",
      "sig": "SQL injection attempt.", "outcome": "failure",
      "time": "2026-06-09T01:36:00+05:30"
    }
  ],
  "stats": {
    "events": 1867, "mapped": 400, "unique_ips": 126, "countries": 12,
    "categories": { "incoming_threat": 1184, "normal_incoming": 277, "outgoing": 170, "external_conn": 2 },
    "top_countries":   [ { "country": "ZA", "country_name": "South Africa", "count": 507 } ],
    "top_signatures":  [ { "sig": "Fortigate: Login failed.", "count": 295 } ]
  }
}
```

**Categories** (drive colour + flow direction):

| category | colour | meaning | arc |
|----------|--------|---------|-----|
| `incoming_threat` | 🔴 red | blocked/failed inbound (login fail, SQLi, drop) | external → DC |
| `normal_incoming` | 🔵 blue | allowed/successful inbound | external → DC |
| `outgoing` | 🟢 green | server reaching out | DC → external |
| `external_conn` | 🟡 yellow | VPN / session connections | external → DC |

---

## 2. Frontend (React)

```bash
# copy the frontend/ folder into your app (or publish it as a package) and:
npm i globe.gl            # peer-ish dep; react/react-dom you already have
```

Serve the three files in `assets/` at some base path (e.g. `/threat-globe/…`)
via your static server / CDN.

```jsx
import ThreatGlobe from "./threat-globe/frontend/ThreatGlobe.jsx";

export default function SecurityMapPage() {
  return (
    <ThreatGlobe
      feedUrl="/api/geofeed"        // your endpoint returning the contract above
      assetBase="/threat-globe"     // where the 3 asset files are served
      pollMs={6000}                 // poll cadence (0 = manual; see push mode)
      height="640px"
    />
  );
}
```

**Push mode** (drive it from your own WebSocket/store instead of polling):

```jsx
<ThreatGlobe data={geofeedPayload} />   // re-renders the globe whenever `data` changes
```

Props: `feedUrl`, `assetBase`, `pollMs`, `data`, `height`, `title`, `className`.
Re-theme by overriding the `--tg-*` CSS variables on `.tg-root`. The CSS is
imported by the component; nothing leaks outside `.tg-root`.

The component lazy-loads globe.gl/Three.js (~1.5 MB) only when it mounts, so it
never weighs down the rest of your bundle.

### Without React
`frontend/globeEngine.js` is framework-agnostic:

```js
import { createGlobeEngine } from "./threat-globe/frontend/globeEngine.js";
const engine = await createGlobeEngine(document.getElementById("globe"), {
  assetBase: "/threat-globe",
  onUpdate: ({ stats, fresh }) => { /* update your own HUD */ },
});
engine.applyFeed(await (await fetch("/api/geofeed")).json());
// engine.setActiveCategories(new Set([...])); engine.resize(); engine.destroy();
```

---

## 3. Backend (Node)

```bash
npm i maxmind
# download GeoLite2-City.mmdb (+ optional GeoLite2-ASN.mmdb) from MaxMind (free)
```

```js
import express from "express";
import { createGeofeedHandler } from "./threat-globe/backend/geofeed.js";

const app = express();
app.get("/api/geofeed", createGeofeedHandler({
  cityDbPath: "/data/GeoLite2-City.mmdb",
  asnDbPath:  "/data/GeoLite2-ASN.mmdb",          // optional (adds ASN/org)
  dcIp:       "203.0.113.10",                      // your edge/WAN IP → geolocated as the hub
  // dcLat / dcLng / dcLabel — fallback hub location if dcIp won't resolve

  // *** plug in YOUR firewall events (most recent few thousand) ***
  fetchEvents: async () => mySiem.getRecentFirewallEvents(2000),
}));
// also: app.use("/threat-globe", express.static(".../threat-globe/assets"));
```

`fetchEvents` returns an array of raw firewall events. By default the handler
reads the UEBA-enriched schema (`network.direction`, `event_outcome`,
`security.signature`, `subject.ip`, `object.ip`). **If your schema differs,
adapt it with one function — no other change needed:**

```js
createGeofeedHandler({
  /* … */,
  isFirewallEvent: (raw) => raw.log_type === "firewall",
  mapEvent: (raw) => ({
    direction: raw.flow_dir,          // "incoming" | "outgoing"
    outcome:   raw.action,            // "failure"/"deny"/"success"/…
    sig:       raw.rule_name,
    subIp:     raw.src_ip,
    objIp:     raw.dst_ip,
    time:      raw.timestamp,
  }),
});
```

The classification rules live in `backend/classify.js` (pure JS) if you want to
tweak them or port them to another language.

---

## 4. Files

```
threat-globe/
  frontend/
    ThreatGlobe.jsx      React component (public API)
    globeEngine.js       framework-agnostic globe.gl engine (WebGL + tooltip)
    categories.js        category colours/labels + format helpers (shared)
    threat-globe.css     scoped styles (.tg-*) — self-contained, themeable
    index.js             barrel export
    package.json
  backend/
    geofeed.js           createGeofeedHandler() — geoIP + feed builder
    classify.js          incoming/outgoing + category classification (pure JS)
    package.json
  assets/
    earth-night.jpg  night-sky.png  countries-110m.geojson
  README.md
```

No license headers added — vendor under your own terms. GeoLite2 data is © MaxMind
(its own license); globe.gl/three.js are MIT.

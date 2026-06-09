// <ThreatGlobe /> — a self-contained live firewall threat-map for React (web).
//
//   import ThreatGlobe from "threat-globe/frontend/ThreatGlobe.jsx";
//   <ThreatGlobe feedUrl="/api/geofeed" assetBase="/threat-globe" />
//
// It renders a 3D globe of geolocated firewall traffic (incoming/outgoing arcs,
// attacker points, impact rings), a HUD (stats, top countries, top signatures,
// live ticker) and clickable direction/category filters. It owns its own styles
// (scoped under .tg-root) so it won't collide with your app's CSS.
//
// Props:
//   feedUrl   string   GET endpoint returning the geofeed JSON (see README contract). Default "/api/geofeed".
//   assetBase string   base URL where earth-night.jpg / night-sky.png / countries-110m.geojson are served. Default "/threat-globe".
//   pollMs    number   poll interval in ms (default 6000). Ignored when `data` is provided.
//   data      object   optional: push a geofeed payload directly (push mode; disables polling).
//   height    string   CSS height of the widget (default "640px").
//   title     string   header label (default "Live Threat Map").
//   className string   extra class on the root.

import { useEffect, useMemo, useRef, useState } from "react";
import { createGlobeEngine } from "./globeEngine.js";
import { CATEGORY, CATEGORY_ORDER, catColor, flagEmoji, shortSig, esc } from "./categories.js";
import "./threat-globe.css";

const TICKER_MAX = 40;

export default function ThreatGlobe({
  feedUrl = "/api/geofeed",
  assetBase = "/threat-globe",
  pollMs = 6000,
  data = null,
  height = "640px",
  title = "Live Threat Map",
  className = "",
}) {
  const hostRef = useRef(null);
  const engineRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [geoOk, setGeoOk] = useState(true);
  const [stats, setStats] = useState({});
  const [ticker, setTicker] = useState([]);
  const [active, setActive] = useState(() => new Set(CATEGORY_ORDER));

  // Create / destroy the globe engine with the host element's lifetime.
  useEffect(() => {
    let engine = null, cancelled = false;
    (async () => {
      engine = await createGlobeEngine(hostRef.current, {
        assetBase,
        onUpdate: ({ geoOk, stats, fresh }) => {
          setGeoOk(geoOk);
          setStats(stats || {});
          if (fresh && fresh.length) {
            setTicker((t) => [...fresh.slice(-8).reverse(), ...t].slice(0, TICKER_MAX));
          }
        },
      });
      if (cancelled) { engine.destroy(); return; }
      engineRef.current = engine;
      setReady(true);
    })();

    const onResize = () => engineRef.current && engineRef.current.resize();
    const onVis = () => engineRef.current && engineRef.current.setRunning(!document.hidden);
    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVis);
      if (engineRef.current) { engineRef.current.destroy(); engineRef.current = null; }
    };
  }, [assetBase]);

  // Polling mode (when no `data` prop is supplied).
  useEffect(() => {
    if (!ready || data) return;
    let stop = false, timer = null;
    const tick = async () => {
      if (document.hidden) return;
      try {
        const r = await fetch(feedUrl);
        if (!r.ok) throw new Error("HTTP " + r.status);
        const json = await r.json();
        if (!stop && engineRef.current) engineRef.current.applyFeed(json);
      } catch (e) { /* transient — keep polling */ }
    };
    tick();
    if (pollMs > 0) timer = setInterval(tick, pollMs);
    return () => { stop = true; if (timer) clearInterval(timer); };
  }, [ready, feedUrl, pollMs, data]);

  // Push mode (when `data` prop changes).
  useEffect(() => {
    if (ready && data && engineRef.current) engineRef.current.applyFeed(data);
  }, [ready, data]);

  // Apply category filters to the engine whenever they change.
  useEffect(() => {
    if (engineRef.current) engineRef.current.setActiveCategories(active);
  }, [active]);

  const toggleCat = (cat) => setActive((prev) => {
    const next = new Set(prev);
    next.has(cat) ? next.delete(cat) : next.add(cat);
    return next;
  });

  const cats = stats.categories || {};
  const topCountries = stats.top_countries || [];
  const topSigs = stats.top_signatures || [];
  const maxC = topCountries.length ? topCountries[0].count : 1;
  const visTicker = useMemo(() => ticker.filter((a) => active.has(a.category)), [ticker, active]);

  return (
    <div className={`tg-root ${className}`} style={{ height }}>
      <div className="tg-wrap">
        <div className="tg-globe-col">
          <div ref={hostRef} className="tg-globe" />

          <div className="tg-filters">
            {CATEGORY_ORDER.map((cat) => (
              <button
                key={cat}
                className={`tg-filt ${active.has(cat) ? "" : "off"}`}
                onClick={() => toggleCat(cat)}
                title={CATEGORY[cat].label}
              >
                <i style={{ background: CATEGORY[cat].color, color: CATEGORY[cat].color }} />
                {CATEGORY[cat].label} <b>{(cats[cat] || 0).toLocaleString()}</b>
              </button>
            ))}
            <span className="tg-filt-dc"><i />Datacenter</span>
          </div>

          {!geoOk && (
            <div className="tg-offline">
              <div className="tg-offline-box">
                <div className="tg-offline-icon">🌐</div>
                <div className="tg-offline-title">GeoIP feed unavailable</div>
                <div className="tg-offline-sub">The map needs a reachable geofeed with a local GeoLite2 DB.</div>
              </div>
            </div>
          )}
        </div>

        <aside className="tg-side">
          <div className="tg-panel">
            <div className="tg-side-title">⬡ {title}</div>
            <div className="tg-stats">
              <div className="tg-stat"><b>{(stats.events || 0).toLocaleString()}</b><span>EVENTS / WINDOW</span></div>
              <div className="tg-stat"><b>{(stats.mapped || 0).toLocaleString()}</b><span>GEO-MAPPED</span></div>
              <div className="tg-stat"><b>{(stats.unique_ips || 0).toLocaleString()}</b><span>UNIQUE IPs</span></div>
              <div className="tg-stat"><b>{(stats.countries || 0).toLocaleString()}</b><span>COUNTRIES</span></div>
            </div>
          </div>

          <div className="tg-panel tg-grow">
            <div className="tg-side-title">↗ Top Source Countries</div>
            <div className="tg-countries">
              {topCountries.length ? topCountries.map((r) => (
                <div className="tg-crow" key={r.country}>
                  <span className="tg-flag">{flagEmoji(r.country)}</span>
                  <span className="tg-cn" title={r.country_name}>{r.country_name || r.country}</span>
                  <span className="tg-bar"><i style={{ width: `${Math.max(4, (r.count / maxC) * 100)}%` }} /></span>
                  <span className="tg-ct">{r.count.toLocaleString()}</span>
                </div>
              )) : <div className="tg-empty">No external sources in window</div>}
            </div>
          </div>

          <div className="tg-panel">
            <div className="tg-side-title">⚑ Top Signatures</div>
            <div className="tg-sigs">
              {topSigs.length ? topSigs.map((r, i) => (
                <div className="tg-sig" key={i}><span>{r.sig}</span><b>{r.count.toLocaleString()}</b></div>
              )) : <div className="tg-empty">—</div>}
            </div>
          </div>

          <div className="tg-panel tg-grow">
            <div className="tg-side-title">⟜ Live Attack Stream</div>
            <div className="tg-ticker">
              {visTicker.map((a, i) => (
                <div className="tg-tick" key={a.ip + a.time + i}>
                  <span className="tg-dot" style={{ background: catColor(a.category), color: catColor(a.category) }} />
                  <span className="tg-tip">{a.direction === "out" ? "↑" : "↓"} {a.country_name || a.country}{a.city ? " · " + a.city : ""}</span>
                  <span className="tg-tsig">{shortSig(a.sig)}</span>
                  <span className="tg-tip2">{a.ip}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

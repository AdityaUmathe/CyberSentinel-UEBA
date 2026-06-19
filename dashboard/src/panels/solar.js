// Solar position — the "subsolar point": the lat/lng on Earth where the Sun is
// directly overhead right now. Used to draw the live day/night terminator (and a
// moving sun) on both the 3D globe and the 2D map.
//
// Standard NOAA solar-position formulas (same maths as the `solar-calculator`
// package), implemented inline so there's no extra dependency and it stays
// fully offline. Accuracy is well within a fraction of a degree — far better
// than this cosmetic feature needs.

const RAD = Math.PI / 180;

function julian(date) { return (+date) / 86400000 + 2440587.5; }
function century(date) { return (julian(date) - 2451545) / 36525; }

function meanLongitude(t) {
  let l = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360;
  return l < 0 ? l + 360 : l;
}
function meanAnomaly(t) { return 357.52911 + t * (35999.05029 - 0.0001537 * t); }
function eccentricity(t) { return 0.016708634 - t * (0.000042037 + 0.0000001267 * t); }
function equationOfCenter(t) {
  const m = meanAnomaly(t) * RAD;
  return Math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
       + Math.sin(2 * m) * (0.019993 - 0.000101 * t)
       + Math.sin(3 * m) * 0.000289;
}
function trueLongitude(t) { return meanLongitude(t) + equationOfCenter(t); }
function apparentLongitude(t) {
  return trueLongitude(t) - 0.00569 - 0.00478 * Math.sin((125.04 - 1934.136 * t) * RAD);
}
function meanObliquity(t) {
  const s = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813));
  return 23 + (26 + s / 60) / 60;
}
function obliquityCorrection(t) {
  return meanObliquity(t) + 0.00256 * Math.cos((125.04 - 1934.136 * t) * RAD);
}

// Solar declination (degrees) — the subsolar latitude.
export function declination(t) {
  const e = obliquityCorrection(t) * RAD;
  const lambda = apparentLongitude(t) * RAD;
  return Math.asin(Math.sin(e) * Math.sin(lambda)) / RAD;
}

// Equation of time (minutes) — the discrepancy between solar and clock noon.
export function equationOfTime(t) {
  const eps = obliquityCorrection(t) * RAD;
  const l0 = meanLongitude(t) * RAD;
  const m = meanAnomaly(t) * RAD;
  const e = eccentricity(t);
  const y = Math.tan(eps / 2) ** 2;
  const eq = y * Math.sin(2 * l0) - 2 * e * Math.sin(m)
    + 4 * e * y * Math.sin(m) * Math.cos(2 * l0)
    - 0.5 * y * y * Math.sin(4 * l0) - 1.25 * e * e * Math.sin(2 * m);
  return (eq / RAD) * 4;
}

// The subsolar point for a given instant: { lat, lng } in degrees.
export function subsolarPoint(date) {
  const dt = +date;
  const t = century(date);
  const dayStart = new Date(dt).setUTCHours(0, 0, 0, 0);
  // longitude where it's solar noon: -180 at 00:00 UTC, ~0 at 12:00 UTC
  let lng = ((dayStart - dt) / 864e5) * 360 - 180 - equationOfTime(t) / 4;
  lng = ((lng + 180) % 360 + 360) % 360 - 180;   // wrap to [-180, 180]
  return { lat: declination(t), lng };
}

// AI Security Analyst — proxies through /api/ai-analyze; falls back to local
// summary text when the proxy can't reach Anthropic.

import { state } from "../state.js";

export async function generateAIAnalysis(stats, feedData, score) {
  const body = document.getElementById("ai-analyst-body");
  const ts   = document.getElementById("ai-timestamp");
  if (!body) return;
  if (state.aiGenerated) return;
  state.aiGenerated = true;

  const now = new Date().toLocaleString("en-IN");
  if (ts) ts.textContent = "Behavioural Analysis Report · " + now;

  const total      = stats.total_alerts || 0;
  const highly     = stats.highly_anomalous || 0;
  const anomalous  = stats.anomalous || 0;
  const suspicious = stats.suspicious || 0;
  const campaigns  = stats.campaigns || 0;
  const users      = stats.unique_users || 0;
  const rate1h     = stats.alert_rate_1h || 0;
  const topReasons = (stats.top_reasons || [])
    .slice(0, 3)
    .map((r) => r.reason.replace(/_/g, " "))
    .join(", ");
  const topSig = feedData.length ? (feedData[0].signature || "").slice(0, 60) : "N/A";

  const prompt = `You are a SOC security analyst writing a concise behavioural analysis report. Based on the following UEBA metrics, write a 2-3 sentence professional summary suitable for a SOC dashboard. Be specific, use the numbers provided, and end with a risk assessment verdict.

Data:
- Total alerts: ${total}
- Highly anomalous: ${highly} (${total > 0 ? ((highly / total) * 100).toFixed(1) : 0}%)
- Anomalous: ${anomalous}
- Suspicious: ${suspicious}
- Campaigns detected: ${campaigns}
- Unique users with anomalies: ${users}
- Alert rate last hour: ${rate1h}
- Top anomaly reasons: ${topReasons || "behavioral baseline deviation"}
- Top signature: ${topSig}
- Risk score: ${score.toFixed(1)}/100

Write only the summary paragraph, no heading, no bullet points. Start with the endpoint/system name context.`;

  try {
    // Server-side proxy: Flask reads ANTHROPIC_API_KEY from env.
    // Falls back to local summary below when the proxy can't reach the API.
    const response = await fetch("/api/ai-analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    const text = data && data.ok && data.text ? data.text : "";
    if (text) {
      const first = text.charAt(0);
      const rest  = text.slice(1);
      body.innerHTML = `<div class="ai-report-text"><span class="ai-first-letter">${first}</span>${rest}</div>`;
    } else {
      throw new Error((data && data.error) || "No content");
    }
  } catch (e) {
    const verdict =
      score < 20
        ? "✅ Excellent security posture. No significant concerns detected."
        : score < 50
        ? "⚠️ Moderate risk level. Review flagged users and campaigns."
        : "🚨 Elevated risk. Immediate investigation recommended.";
    const text =
      `UEBA engine analysis reveals ${total.toLocaleString()} total security alerts over the monitored period. ` +
      `${highly} events (${total > 0 ? ((highly / total) * 100).toFixed(1) : 0}%) are classified as highly anomalous with ${campaigns} active campaigns detected across ${users} unique user accounts. ` +
      `The most prominent anomaly patterns include ${topReasons || "behavioral baseline deviation"}. ` +
      verdict;
    const first = text.charAt(0);
    body.innerHTML = `<div class="ai-report-text"><span class="ai-first-letter">${first}</span>${text.slice(1)}</div>`;
  }
}

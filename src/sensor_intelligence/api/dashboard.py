"""Server-rendered monitoring dashboard.

A single self-contained HTML page that fetches the ``/demo`` endpoint and
renders the observed series with its forecast band (via Chart.js from a CDN)
and a table of alerts coloured by severity. It is intentionally dependency-free
on the backend — just an HTML string — so it ships with the API and needs no
separate frontend build.
"""

from __future__ import annotations

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sensor Intelligence Platform</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; margin: 0; background: #0f1117; color: #e6e6e6; }
    header { padding: 1.25rem 2rem; background: #161922; border-bottom: 1px solid #262a36; }
    header h1 { margin: 0; font-size: 1.25rem; }
    header p { margin: 0.25rem 0 0; color: #9aa0ad; font-size: 0.9rem; }
    main { padding: 1.5rem 2rem; display: grid; gap: 1.5rem; max-width: 1100px; }
    .card { background: #161922; border: 1px solid #262a36; border-radius: 10px;
            padding: 1rem 1.25rem; }
    .card h2 { margin: 0 0 0.75rem; font-size: 1rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #262a36; }
    th { color: #9aa0ad; font-weight: 600; }
    .sev { font-weight: 700; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; }
    .critical { background: #5b1a1a; color: #ffb4b4; }
    .warning { background: #5b4a1a; color: #ffe4a1; }
    .info { background: #1a3a5b; color: #a1d4ff; }
    .muted { color: #9aa0ad; }
    button { background: #2d6cdf; color: white; border: 0; border-radius: 7px;
             padding: 0.5rem 0.9rem; cursor: pointer; font-size: 0.85rem; }
    button:hover { background: #3b78e8; }
  </style>
</head>
<body>
  <header>
    <h1>Sensor Intelligence Platform</h1>
    <p>Forecasting, anomaly detection, and drift monitoring for sensor data.</p>
  </header>
  <main>
    <div class="card">
      <h2>Live demo <button id="refresh">Refresh</button></h2>
      <p class="muted" id="summary">Loading…</p>
      <canvas id="chart" height="110"></canvas>
    </div>
    <div class="card">
      <h2>Alerts</h2>
      <table>
        <thead><tr><th>Severity</th><th>Sensor</th><th>Time</th><th>Message</th></tr></thead>
        <tbody id="alerts"></tbody>
      </table>
    </div>
  </main>
  <script>
    let chart = null;

    function severityClass(sev) {
      return sev === 'critical' ? 'critical' : sev === 'warning' ? 'warning' : 'info';
    }

    function renderAlerts(alerts) {
      const tbody = document.getElementById('alerts');
      tbody.innerHTML = '';
      if (!alerts.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="muted">No alerts.</td></tr>';
        return;
      }
      for (const a of alerts) {
        const tr = document.createElement('tr');
        const sev = severityClass(a.severity);
        tr.innerHTML = `<td><span class="sev ${sev}">${a.severity.toUpperCase()}</span></td>` +
          `<td>${a.sensor_id}</td><td>${new Date(a.timestamp).toLocaleString()}</td>` +
          `<td>${a.message}</td>`;
        tbody.appendChild(tr);
      }
    }

    function renderChart(series, forecast) {
      const primary = series[0];
      const histLabels = primary.timestamps;
      const labels = histLabels.concat(forecast.timestamps);
      const pad = new Array(histLabels.length).fill(null);
      const fpad = new Array(forecast.timestamps.length).fill(null);
      const datasets = [
        { label: primary.sensor_id, data: primary.values.concat(fpad),
          borderColor: '#2d6cdf', pointRadius: 0, tension: 0.2 },
        { label: 'forecast', data: pad.concat(forecast.mean),
          borderColor: '#e8a13b', borderDash: [5, 4], pointRadius: 0, tension: 0.2 },
        { label: 'upper', data: pad.concat(forecast.upper),
          borderColor: 'transparent', pointRadius: 0 },
        { label: 'lower', data: pad.concat(forecast.lower), borderColor: 'transparent',
          backgroundColor: 'rgba(232,161,59,0.15)', fill: '-1', pointRadius: 0 },
      ];
      const cfg = { type: 'line', data: { labels, datasets },
        options: { animation: false, scales: { x: { display: false } },
          plugins: { legend: { labels: { color: '#9aa0ad' } } } } };
      if (chart) { chart.destroy(); }
      chart = new Chart(document.getElementById('chart'), cfg);
    }

    async function load() {
      document.getElementById('summary').textContent = 'Loading…';
      const resp = await fetch('demo');
      const data = await resp.json();
      const total = data.series.reduce((n, s) => n + s.values.length, 0);
      document.getElementById('summary').textContent =
        `${total} readings · ${data.series.length} sensors · ${data.alerts.length} alerts`;
      renderChart(data.series, data.forecast);
      renderAlerts(data.alerts);
    }

    document.getElementById('refresh').addEventListener('click', load);
    load();
  </script>
</body>
</html>
"""


def render_dashboard() -> str:
    """Return the dashboard HTML page."""
    return _DASHBOARD_HTML

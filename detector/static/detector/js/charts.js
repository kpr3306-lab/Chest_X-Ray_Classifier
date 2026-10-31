// Dependency-free SVG line charts for the training curves, plus the overview sparklines.
(function () {
  const raw = document.getElementById("history-data");
  if (!raw) return;
  const h = JSON.parse(raw.textContent);
  if (!h || !h.train_loss || !h.train_loss.length) return;

  function draw(elId, series, yLabel) {
    const el = document.getElementById(elId);
    const W = 520, H = 300, m = { l: 48, r: 14, t: 14, b: 58 };
    const n = series[0].values.length;
    const all = series.flatMap(s => s.values);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.1 || 0.05; lo = Math.max(0, lo - pad); hi = hi + pad;
    const x = i => m.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - m.l - m.r);
    const y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${yLabel} per epoch">`;
    for (let k = 0; k <= 4; k++) {
      const v = lo + (k / 4) * (hi - lo), yy = y(v);
      svg += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}"/><text x="${m.l - 6}" y="${yy + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
    }
    svg += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    for (let i = 0; i < n; i++) svg += `<text x="${x(i)}" y="${H - m.b + 16}" text-anchor="middle">${i + 1}</text>`;
    svg += `<text x="${(W + m.l) / 2}" y="${H - m.b + 32}" text-anchor="middle">Epoch</text>`;

    // dashed line where fine-tuning starts
    if (h.phases) {
      const k = h.phases.findIndex((p, i) => i > 0 && p !== h.phases[i - 1]);
      if (k > 0) { const xx = (x(k - 1) + x(k)) / 2; svg += `<line x1="${xx}" x2="${xx}" y1="${m.t}" y2="${H - m.b}" stroke="#5b7bb0" stroke-dasharray="4 4"/>`; }
    }
    series.forEach(s => {
      const pts = s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
      svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2.5" points="${pts}"/>`;
      s.values.forEach((v, i) => { svg += `<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="${s.color}"/>`; });
    });
    let lx = m.l;
    series.forEach(s => {
      svg += `<rect x="${lx}" y="${H - 14}" width="12" height="4" fill="${s.color}"/><text x="${lx + 18}" y="${H - 8}">${s.name}</text>`;
      lx += 110;
    });
    el.innerHTML = svg + "</svg>";
  }

  // Area sparkline: <div data-spark="val_acc" data-color="#hex">
  document.querySelectorAll("[data-spark]").forEach((el, k) => {
    const v = h[el.dataset.spark];
    if (!v || v.length < 2) return;
    const W = 220, H = 80, c = el.dataset.color, lo = Math.min(...v), hi = Math.max(...v);
    const x = i => 4 + i / (v.length - 1) * (W - 12);
    const y = n => 8 + (1 - (n - lo) / ((hi - lo) || 1)) * (H - 16);
    // Catmull-Rom -> cubic bezier for a smooth line
    let d = `M${x(0)},${y(v[0])}`;
    for (let i = 0; i < v.length - 1; i++) {
      const p0 = v[Math.max(i - 1, 0)], p1 = v[i], p2 = v[i + 1], p3 = v[Math.min(i + 2, v.length - 1)];
      d += ` C${x(i) + (x(i + 1) - x(Math.max(i - 1, 0))) / 6},${y(p1) + (y(p2) - y(p0)) / 6} ${x(i + 1) - (x(Math.min(i + 2, v.length - 1)) - x(i)) / 6},${y(p2) - (y(p3) - y(p1)) / 6} ${x(i + 1)},${y(p2)}`;
    }
    const n = v.length - 1;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${el.dataset.spark.replace("_", " ")} over ${v.length} epochs">
      <defs><linearGradient id="sg${k}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${c}" stop-opacity=".45"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></linearGradient></defs>
      <path d="${d} L${x(n)},${H} L${x(0)},${H} Z" fill="url(#sg${k})"/>
      <path d="${d}" fill="none" stroke="${c}" stroke-width="2.5" vector-effect="non-scaling-stroke"/>
      <circle cx="${x(n)}" cy="${y(v[n])}" r="4" fill="#fff" stroke="${c}" stroke-width="3" vector-effect="non-scaling-stroke"/></svg>`;
  });

  if (!document.getElementById("chart-loss")) return;
  draw("chart-loss", [
    { name: "Training", values: h.train_loss, color: "#38d6ff" },
    { name: "Validation", values: h.val_loss, color: "#f5b524" },
  ], "Loss");
  draw("chart-acc", [
    { name: "Training", values: h.train_acc, color: "#38d6ff" },
    { name: "Validation", values: h.val_acc, color: "#f5b524" },
  ], "Accuracy");
})();

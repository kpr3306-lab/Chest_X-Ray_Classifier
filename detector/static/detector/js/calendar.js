// Toolbar calendar: per-day prediction counts (red = any non-normal result, green = all normal);
// picking a day filters the Overview, or History when you are on it, via ?date=YYYY-MM-DD.
(function () {
  const root = document.getElementById("cal");
  if (!root) return;
  const btn = document.getElementById("cal-btn");
  const pop = document.getElementById("cal-pop");
  const daysEl = document.getElementById("cal-days");
  const monthEl = document.getElementById("cal-month");
  const pad = n => String(n).padStart(2, "0");
  const iso = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const today = new Date();
  const picked = new Date(root.dataset.selected + "T00:00");
  const selected = isNaN(picked) ? "" : root.dataset.selected;
  let view = new Date((selected ? picked : today).getFullYear(), (selected ? picked : today).getMonth(), 1);
  let data = null;

  if (selected) {
    document.getElementById("cal-label").textContent =
      picked.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  }

  function go(day) {
    const base = root.dataset.onHistory ? root.dataset.history : root.dataset.home;
    location.href = day ? `${base}?date=${day}` : base;
  }

  function render() {
    monthEl.textContent = view.toLocaleDateString(undefined, { month: "long", year: "numeric" });
    const lead = (view.getDay() + 6) % 7;  // Monday-first grid
    const last = new Date(view.getFullYear(), view.getMonth() + 1, 0).getDate();
    let html = "<span></span>".repeat(lead);
    for (let d = 1; d <= last; d++) {
      const key = iso(new Date(view.getFullYear(), view.getMonth(), d));
      const info = data && data[key];
      const cls = ["cal-day", key === iso(today) && "today", key === selected && "sel"].filter(Boolean).join(" ");
      const title = info
        ? `${info.n} prediction${info.n > 1 ? "s" : ""}, ${info.bad ? info.bad + " non-normal" : "all normal"}`
        : "No predictions";
      html += `<button type="button" class="${cls}" data-date="${key}" title="${title}" aria-label="${key}: ${title}"` +
        `${key === selected ? ' aria-current="date"' : ""}>${d}` +
        `${info ? `<i class="cal-dot ${info.bad ? "bad" : "ok"}">${info.n}</i>` : ""}</button>`;
    }
    daysEl.innerHTML = html;
  }

  function load() {
    if (data) return;
    fetch(root.dataset.api, { credentials: "same-origin" })
      .then(r => r.json())
      .then(j => { data = j.days || {}; render(); })
      .catch(() => { data = {}; });  // calendar still works for filtering, just without counts
  }

  function toggle(open) {
    pop.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    if (open) { render(); load(); }
  }

  btn.addEventListener("click", () => toggle(pop.hidden));
  document.addEventListener("click", e => { if (!pop.hidden && !root.contains(e.target)) toggle(false); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !pop.hidden) { toggle(false); btn.focus(); } });
  pop.querySelectorAll(".cal-nav").forEach(b => b.addEventListener("click", () => {
    view = new Date(view.getFullYear(), view.getMonth() + Number(b.dataset.step), 1);
    render();
  }));
  daysEl.addEventListener("click", e => { const b = e.target.closest("[data-date]"); if (b) go(b.dataset.date); });
  document.getElementById("cal-today").addEventListener("click", () => go(iso(today)));
  const clear = document.getElementById("cal-clear");
  clear.disabled = !root.dataset.selected;
  clear.addEventListener("click", () => go(""));
})();

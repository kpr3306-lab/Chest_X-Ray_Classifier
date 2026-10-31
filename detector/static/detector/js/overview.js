// Overview page: thumbnail -> detail switching, hover tooltips, and the rotatable 3D lung.
(function () {
  // ---------------------------------------------------------------- thumbnail selection
  const thumbs = [...document.querySelectorAll(".lung-thumbs [data-pk]")];
  function select(t) {
    thumbs.forEach(x => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", String(x === t)); });
    document.querySelectorAll(".lung-panel [data-for]").forEach(el => { el.hidden = el.dataset.for !== t.dataset.pk; });
  }
  thumbs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", e => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      const next = step && thumbs[i + step];
      if (next) { e.preventDefault(); next.focus(); select(next); }
    });
  });

  // ---------------------------------------------------------------- tooltips (name / ID / prediction)
  // A fixed-position element, so it is not clipped by the scrollable thumbnail column.
  const tip = document.createElement("div");
  tip.className = "tip"; tip.setAttribute("role", "tooltip"); tip.hidden = true;
  document.body.append(tip);
  function show(el) {
    const title = document.createElement("strong"), sub = document.createElement("span");
    title.textContent = el.dataset.tipTitle; sub.textContent = el.dataset.tipSub;
    tip.replaceChildren(title, sub);
    tip.hidden = false;
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let x = r.left + r.width / 2 - w / 2, y = r.top - h - 10;
    if (y < 8) y = r.bottom + 10;
    tip.style.left = Math.max(8, Math.min(x, innerWidth - w - 8)) + "px";
    tip.style.top = y + "px";
  }
  const hide = () => { tip.hidden = true; };
  document.querySelectorAll("[data-tip-title]").forEach(el => {
    el.addEventListener("mouseenter", () => show(el));
    el.addEventListener("focus", () => show(el));
    el.addEventListener("mouseleave", hide);
    el.addEventListener("blur", hide);
  });
  addEventListener("scroll", hide, true);

  // ---------------------------------------------------------------- 3D lung
  const canvas = document.getElementById("lung-3d");
  const img2d = document.getElementById("lung-2d");
  if (!canvas || !canvas.getContext || !window.Path2D) return;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  fetch(img2d.currentSrc || img2d.src)
    .then(r => r.text())
    .then(txt => {
      const doc = new DOMParser().parseFromString(txt, "image/svg+xml");
      const ds = [...doc.querySelectorAll("clipPath path")].map(p => p.getAttribute("d"));
      if (ds.length !== 2) throw new Error("lung outlines not found");
      start(ds.map(d => new Path2D(d)));
    })
    .catch(() => {});  // keep the 2D illustration

  function start(lungs) {
    const probe = document.createElement("canvas").getContext("2d");
    const CX = 300, CY = 300, DEPTH = 92;
    // Each lung: [path, centre x, half width]; SVG space, viewer's left lung first.
    const L = [[lungs[0], 178, 100], [lungs[1], 424, 102]];
    let seed = 11;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

    function lungAt(x, y) { return L.find(l => probe.isPointInPath(l[0], x, y)); }
    // Half-thickness of the lung volume at (x, y): rounded at the sides, apex and base.
    function zMax(l, x, y) {
      const u = (x - l[1]) / l[2], v = (y - 310) / 225;
      return DEPTH * Math.sqrt(Math.max(0, 1 - u * u * 0.85)) * Math.pow(Math.max(0, 1 - v * v), 0.35);
    }

    // Point cloud: mostly on the surface shell (dense silhouettes read as an X-ray), some interior haze.
    const pts = [];
    // Outline walls: points on the lung boundary spread through the depth keep the silhouette crisp.
    const edge = [];
    for (let y = 85; y < 520; y += 4) {
      for (let x = 60; x < 540; x += 4) {
        const l = lungAt(x, y);
        if (l && (!lungAt(x + 4, y) || !lungAt(x - 4, y) || !lungAt(x, y + 4) || !lungAt(x, y - 4))) edge.push([x, y, l]);
      }
    }
    for (let k = 0; k < 2200 && edge.length; k++) {
      const [x, y, l] = edge[Math.floor(rnd() * edge.length)], zm = zMax(l, x, y);
      pts.push([x - CX, y - CY, (rnd() * 2 - 1) * zm, 1.25]);
    }
    while (pts.length < 7400) {
      const x = 60 + rnd() * 480, y = 85 + rnd() * 435, l = lungAt(x, y);
      if (!l) continue;
      const zm = zMax(l, x, y);
      const shell = pts.length < 6400;
      const z = shell ? (rnd() < 0.5 ? -1 : 1) * zm * (0.92 + rnd() * 0.08) : (rnd() * 2 - 1) * zm * 0.8;
      pts.push([x - CX, y - CY, z, shell ? 1 : 0.5]);
    }

    // Airway segments [x1,y1,z1,x2,y2,z2,width] in model space.
    const segs = [];
    const ring = [];
    for (let y = 34; y <= 206; y += 9) {             // trachea cartilage rings
      for (let k = 0; k < 18; k++) {
        const a1 = k / 18 * Math.PI * 2, a2 = (k + 1) / 18 * Math.PI * 2;
        ring.push([Math.cos(a1) * 11, y - CY, Math.sin(a1) * 11, Math.cos(a2) * 11, y - CY, Math.sin(a2) * 11, 1.4]);
      }
    }
    segs.push([0, 24 - CY, 0, 0, 215 - CY, 0, 9]);
    segs.push([0, 215 - CY, 0, 262 - CX, 276 - CY, 0, 7]);
    segs.push([0, 215 - CY, 0, 344 - CX, 284 - CY, 0, 7]);

    function grow(p, dir, len, w, depth) {
      if (!depth || w < 0.3) return;
      for (let tries = 0; tries < 4; tries++) {
        const q = [p[0] + dir[0] * len, p[1] + dir[1] * len, p[2] + dir[2] * len];
        const l = lungAt(q[0] + CX, q[1] + CY);
        if (l && Math.abs(q[2]) < zMax(l, q[0] + CX, q[1] + CY) * 0.85) {
          segs.push([...p, ...q, w]);
          for (let k = 0; k < 2; k++) {
            const j = [dir[0] + (rnd() - 0.5) * 1.1, dir[1] + (rnd() - 0.5) * 1.1, dir[2] + (rnd() - 0.5) * 1.1];
            const n = Math.hypot(...j);
            grow(q, j.map(c => c / n), len * (0.72 + rnd() * 0.12), w * 0.7, depth - 1);
          }
          return;
        }
        len *= 0.7;
      }
    }
    const lobar = [[[-38, -24], [-95, -150, 10]], [[-38, -24], [-135, 45, 25]], [[-38, -24], [-110, 140, -20]],
                   [[44, -16], [95, -150, -10]], [[44, -16], [145, 30, 20]], [[44, -16], [120, 140, -25]]];
    lobar.forEach(([[x, y], t]) => {
      const n = Math.hypot(t[0] - x, t[1] - y, t[2]);
      grow([x, y, 0], [(t[0] - x) / n, (t[1] - y) / n, t[2] / n], 46, 6, 8);
    });

    // Small pre-rendered glow sprite for the particles.
    const sprite = document.createElement("canvas");
    sprite.width = sprite.height = 16;
    const sg = sprite.getContext("2d"), grad = sg.createRadialGradient(8, 8, 0, 8, 8, 8);
    grad.addColorStop(0, "rgba(150,225,255,1)"); grad.addColorStop(0.35, "rgba(60,160,255,.45)"); grad.addColorStop(1, "rgba(30,110,255,0)");
    sg.fillStyle = grad; sg.fillRect(0, 0, 16, 16);

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1, scale = 1;
    function resize() {
      dpr = Math.min(devicePixelRatio || 1, 2);
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      scale = Math.min(W / 520, H / 540) * 0.98;
    }

    let yaw = -0.35, pitch = 0.12, vyaw = reduced ? 0 : 0.004, dragging = false, last = null, visible = true, mode3d = false, running = false;
    const kick = () => { if (!running) { running = true; requestAnimationFrame(frame); } };
    function project(x, y, z, cy, sy, cp, sp) {
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const f = 900 / (900 + z2);
      return [W / 2 + x1 * f * scale, H / 2 + (y2 + 30) * f * scale, z2];
    }

    function frame() {
      if (!mode3d || !visible) { running = false; return; }
      if (!dragging) { yaw += vyaw; if (!reduced) vyaw += (0.004 - vyaw) * 0.02; }
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.globalCompositeOperation = "lighter";

      const size = 5 * scale + 1.5;
      for (const [x, y, z, b] of pts) {
        const [px, py, pz] = project(x, y, z, cy, sy, cp, sp);
        ctx.globalAlpha = b * (0.16 + 0.14 * (1 - (pz + DEPTH) / (2 * DEPTH)));  // nearer = brighter
        ctx.drawImage(sprite, px - size / 2, py - size / 2, size, size);
      }

      // Airways: a soft wide pass, then a thin bright core.
      for (const [col, wMul, alpha] of [["rgba(56,170,255,1)", 2.6, 0.12], ["rgba(190,240,255,1)", 0.8, 0.55]]) {
        ctx.strokeStyle = col; ctx.lineCap = "round";
        for (const list of [segs, ring]) {
          for (const s of list) {
            const a = project(s[0], s[1], s[2], cy, sy, cp, sp), c = project(s[3], s[4], s[5], cy, sy, cp, sp);
            ctx.globalAlpha = alpha * (s[6] < 1 ? 0.6 : 1);
            ctx.lineWidth = Math.max(0.4, s[6] * wMul * scale * 0.55);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]); ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
      requestAnimationFrame(frame);
    }

    canvas.addEventListener("pointerdown", e => { dragging = true; last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.classList.add("grabbing"); });
    canvas.addEventListener("pointermove", e => {
      if (!dragging) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      vyaw = dx * 0.008; yaw += vyaw;
      pitch = Math.max(-0.9, Math.min(0.9, pitch + dy * 0.008));
    });
    const end = () => { dragging = false; canvas.classList.remove("grabbing"); if (reduced) vyaw = 0; };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);

    function setMode(is3d) {
      mode3d = is3d;
      canvas.hidden = !is3d; img2d.hidden = is3d;
      document.getElementById("drag-hint").hidden = !is3d;
      document.querySelectorAll("#view-toggle button").forEach(b => b.classList.toggle("on", (b.dataset.view === "3d") === is3d));
      if (is3d) { resize(); kick(); }
    }
    document.getElementById("view-toggle").hidden = false;
    document.querySelectorAll("#view-toggle button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.view === "3d")));
    new ResizeObserver(() => { if (mode3d) resize(); }).observe(canvas.parentElement);
    new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible && mode3d) kick();
    }).observe(canvas);
    setMode(true);
  }
})();

(function () {
  const form = document.getElementById("upload-form");
  const input = document.getElementById("xray-input");
  const zone = document.getElementById("dropzone");
  const empty = document.getElementById("dz-empty");
  const preview = document.getElementById("dz-preview");
  const img = document.getElementById("preview-img");
  const nameEl = document.getElementById("preview-name");
  const errEl = document.getElementById("client-error");
  const submit = document.getElementById("submit-btn");
  const clear = document.getElementById("clear-btn");
  const working = document.getElementById("working");
  const maxBytes = parseFloat(form.dataset.maxMb) * 1024 * 1024;
  const okTypes = ["image/jpeg", "image/png"];
  const okExt = /\.(jpe?g|png)$/i;
  let objectUrl = null;

  function showError(msg) { errEl.textContent = msg; errEl.hidden = !msg; }

  function reset() {
    input.value = "";
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    img.removeAttribute("src");
    preview.hidden = true; empty.hidden = false;
    submit.disabled = true; clear.hidden = true;
  }

  function handle(file) {
    showError("");
    if (!file) { reset(); return; }
    if (!okExt.test(file.name) || (file.type && !okTypes.includes(file.type))) {
      reset(); showError("Only JPG, JPEG and PNG files are supported."); return;
    }
    if (file.size > maxBytes) {
      reset(); showError("File is too large. The maximum size is " + form.dataset.maxMb + " MB."); return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    img.src = objectUrl;
    nameEl.textContent = file.name + " (" + (file.size / 1024).toFixed(0) + " KB)";
    empty.hidden = true; preview.hidden = false;
    submit.disabled = false; clear.hidden = false;
  }

  input.addEventListener("change", () => handle(input.files[0]));

  ["dragenter", "dragover"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", e => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) {
      try { input.files = files; } catch (_) { /* older browsers */ }
      handle(files[0]);
    }
  });

  clear.addEventListener("click", () => { showError(""); reset(); });

  form.addEventListener("submit", e => {
    if (!input.files.length) { e.preventDefault(); showError("Please choose an image first."); return; }
    submit.disabled = true; working.hidden = false;
  });
})();

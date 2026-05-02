(function () {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const input = document.getElementById("video-input");
  const submitBtn = document.getElementById("submit-btn");
  const fname = document.getElementById("dz-filename");
  const form = document.getElementById("upload-form");
  const status = document.getElementById("status");
  const errorBox = document.getElementById("error-box");
  const resultBox = document.getElementById("result-box");

  const elLabel = document.getElementById("result-label");
  const elFill = document.getElementById("conf-fill");
  const elPct = document.getElementById("conf-pct");
  const elPFake = document.getElementById("p-fake");
  const elPReal = document.getElementById("p-real");
  const elThr = document.getElementById("thr");
  const elFaces = document.getElementById("frames-faces");
  const elTInf = document.getElementById("t-inf");
  const elTTot = document.getElementById("t-tot");

  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden"); }

  function setFile(file) {
    if (!file) {
      fname.textContent = "No file selected";
      submitBtn.disabled = true;
      return;
    }
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    fname.textContent = file.name + " (" + sizeMb + " MB)";
    submitBtn.disabled = false;
  }

  dropzone.addEventListener("click", function () { input.click(); });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  ["dragenter", "dragover"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (e) {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) { input.files = e.dataTransfer.files; setFile(f); }
  });
  input.addEventListener("change", function () {
    setFile(input.files && input.files[0]);
  });

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const f = input.files && input.files[0];
    if (!f) return;

    hide(errorBox); hide(resultBox); show(status);
    submitBtn.disabled = true;

    const fd = new FormData();
    fd.append("video", f);

    try {
      const resp = await fetch("/predict", { method: "POST", body: fd });
      let data;
      try { data = await resp.json(); }
      catch (_jsonErr) {
        throw new Error("Server returned an unparseable response (status " + resp.status + ").");
      }

      hide(status);

      if (!resp.ok || !data.ok) {
        const msg = (data && data.error && data.error.message) ||
                    "Unknown error (HTTP " + resp.status + ").";
        errorBox.textContent = msg;
        show(errorBox);
        submitBtn.disabled = false;
        return;
      }

      renderResult(data);
      submitBtn.disabled = false;
    } catch (err) {
      hide(status);
      errorBox.textContent = "Network or server error: " + (err.message || err);
      show(errorBox);
      submitBtn.disabled = false;
    }
  });

  function renderResult(d) {
    elLabel.textContent = d.label;
    elLabel.classList.remove("real", "fake");
    elLabel.classList.add(d.label === "FAKE" ? "fake" : "real");

    const pct = d.confidence_pct;
    elFill.style.width = pct + "%";
    elPct.textContent = pct.toFixed(2) + "%";

    elPFake.textContent = d.probability_fake_pct.toFixed(2) + "%";
    elPReal.textContent = d.probability_real_pct.toFixed(2) + "%";
    elThr.textContent = d.threshold.toFixed(3);
    elFaces.textContent =
      (d.video_info && d.video_info.frames_with_faces != null
        ? d.video_info.frames_with_faces : "?")
      + " / " + ((d.video_info && d.video_info.frames_sampled) || "?");
    elTInf.textContent = d.inference_seconds.toFixed(2) + " s";
    elTTot.textContent = d.total_seconds.toFixed(2) + " s";

    show(resultBox);
  }
})();

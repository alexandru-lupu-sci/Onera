/* Onera SleepScorer Frontend Application Controller */
(function () {
  "use strict";

  var STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"];
  var STAGE_PASTEL = {
    W: ["rgba(242, 185, 116, 0.36)", "rgba(217, 130, 54, 0.55)"],
    N1: ["rgba(125, 191, 237, 0.34)", "rgba(91, 143, 217, 0.50)"],
    N2: ["rgba(91, 143, 217, 0.34)", "rgba(91, 143, 217, 0.58)"],
    N3: ["rgba(74, 115, 184, 0.38)", "rgba(74, 115, 184, 0.62)"],
    REM: ["rgba(148, 122, 206, 0.36)", "rgba(139, 114, 190, 0.60)"]
  };
  var STAGE_Y_MAP = { N3: 0.0, S4: 0.0, N2: 1.0, N1: 2.0, REM: 3.0, W: 4.0 };

  function $(id) { return document.getElementById(id); }

  function escHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatHms(totalSec) {
    var sec = Math.max(0, Math.round(Number(totalSec) || 0));
    var h = Math.floor(sec / 3600);
    var m = Math.floor((sec % 3600) / 60);
    var s = sec % 60;
    return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }

  function isCalibrationMarker(desc) {
    return /eyes?|teeth|blink|calib|biocal/i.test(String(desc || ""));
  }

  function apiJson(url, opts) {
    return fetch(url, opts).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) {
          throw new Error((data && data.detail) || ("Request failed (" + res.status + ")"));
        }
        return data;
      });
    });
  }

  /* ---- Theme Management (Light -> Night -> Dark OLED) ---- */
  var THEME_ORDER = ["light", "night", "dark"];

  function detectInitialTheme() {
    try {
      var saved = localStorage.getItem("onera-theme");
      if (THEME_ORDER.indexOf(saved) >= 0) return saved;
    } catch (e) {}
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "night" : "light";
  }

  var currentTheme = detectInitialTheme();
  var hypnoView = null;
  var waveView = null;

  function applyTheme(theme) {
    currentTheme = THEME_ORDER.indexOf(theme) >= 0 ? theme : "light";
    document.documentElement.dataset.theme = currentTheme;
    document.documentElement.classList.toggle("dark", currentTheme === "dark");
    document.documentElement.classList.toggle("night", currentTheme === "night");
    try { localStorage.setItem("onera-theme", currentTheme); } catch (e) {}
    var tBtn = $("theme_toggle_btn");
    if (tBtn) {
      if (currentTheme === "light") {
        tBtn.textContent = "🌑 Night";
        tBtn.title = "Theme: Light · Click for Night";
      } else if (currentTheme === "night") {
        tBtn.textContent = "🌙 Dark";
        tBtn.title = "Theme: Night · Click for Dark (OLED)";
      } else {
        tBtn.textContent = "☀️ Light";
        tBtn.title = "Theme: Dark (OLED) · Click for Light";
      }
    }
    if (hypnoView) hypnoView.setTheme(currentTheme);
    if (waveView) waveView.setTheme(currentTheme);
  }

  function cycleTheme() {
    var nextIdx = (THEME_ORDER.indexOf(currentTheme) + 1) % THEME_ORDER.length;
    applyTheme(THEME_ORDER[nextIdx]);
  }

  /* ---- Searchable Combobox ---- */
  function createComboBox(containerEl, onSelect) {
    var options = [];
    var value = "";
    var open = false;

    var trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "onera-combobox-trigger";
    var triggerText = document.createElement("span");
    triggerText.style.overflow = "hidden";
    triggerText.style.textOverflow = "ellipsis";
    triggerText.style.whiteSpace = "nowrap";
    var arrow = document.createElement("span");
    arrow.textContent = "▾";
    trigger.appendChild(triggerText);
    trigger.appendChild(arrow);

    var menu = document.createElement("div");
    menu.className = "onera-combobox-menu hidden";
    var searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "text-input";
    searchInput.placeholder = "Search recordings...";
    var listEl = document.createElement("div");
    listEl.className = "onera-combobox-list";
    menu.appendChild(searchInput);
    menu.appendChild(listEl);

    containerEl.innerHTML = "";
    containerEl.appendChild(trigger);
    containerEl.appendChild(menu);

    function renderList() {
      var q = searchInput.value.trim().toLowerCase();
      listEl.innerHTML = "";
      var matched = options.filter(function (opt) { return !q || opt.toLowerCase().indexOf(q) >= 0; });
      if (!matched.length) {
        var empty = document.createElement("div");
        empty.style.padding = "6px 8px";
        empty.style.fontSize = "12px";
        empty.style.color = "var(--onera-muted)";
        empty.textContent = "No matching recordings";
        listEl.appendChild(empty);
        return;
      }
      matched.forEach(function (opt) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "onera-combobox-item" + (opt === value ? " is-selected" : "");
        btn.textContent = opt;
        btn.onclick = function () {
          value = opt;
          triggerText.textContent = opt;
          closeMenu();
          if (onSelect) onSelect(opt);
        };
        listEl.appendChild(btn);
      });
    }

    function openMenu() {
      if (!options.length) return;
      open = true;
      searchInput.value = "";
      renderList();
      menu.classList.remove("hidden");
      searchInput.focus();
    }

    function closeMenu() {
      open = false;
      menu.classList.add("hidden");
    }

    trigger.onclick = function (e) {
      e.stopPropagation();
      if (open) closeMenu();
      else openMenu();
    };
    searchInput.oninput = renderList;
    searchInput.onclick = function (e) { e.stopPropagation(); };
    document.addEventListener("click", function (e) {
      if (open && !containerEl.contains(e.target)) closeMenu();
    });

    return {
      setOptions: function (optsList, selectedVal) {
        options = (optsList || []).slice();
        value = selectedVal && options.indexOf(selectedVal) >= 0 ? selectedVal : (options[0] || "");
        triggerText.textContent = value || "(None)";
        trigger.disabled = options.length === 0;
        closeMenu();
      },
      getValue: function () { return value; }
    };
  }

  /* ---- Chip Multi-Select ---- */
  function createMultiSelect(containerEl) {
    var allOptions = [];
    var selected = [];
    var menuOpen = false;

    var box = document.createElement("div");
    box.className = "onera-ms-box";

    var menu = document.createElement("div");
    menu.className = "onera-combobox-menu hidden";
    var search = document.createElement("input");
    search.type = "text";
    search.className = "text-input";
    search.placeholder = "Filter channels...";
    var listEl = document.createElement("div");
    listEl.className = "onera-combobox-list";
    menu.appendChild(search);
    menu.appendChild(listEl);

    containerEl.innerHTML = "";
    containerEl.appendChild(box);
    containerEl.appendChild(menu);

    function renderBox() {
      box.innerHTML = "";
      selected.forEach(function (val) {
        var chip = document.createElement("span");
        chip.className = "onera-ms-chip";
        var lbl = document.createElement("span");
        lbl.textContent = val;
        var rm = document.createElement("button");
        rm.type = "button";
        rm.textContent = "×";
        rm.title = "Remove " + val;
        rm.onclick = function (e) {
          e.stopPropagation();
          selected = selected.filter(function (v) { return v !== val; });
          renderBox();
          if (menuOpen) renderMenu();
        };
        chip.appendChild(lbl);
        chip.appendChild(rm);
        box.appendChild(chip);
      });
      var addBtn = document.createElement("button");
      addBtn.type = "button";
      addBtn.className = "onera-ms-add";
      addBtn.textContent = "+ Add";
      addBtn.onclick = function (e) {
        e.stopPropagation();
        if (menuOpen) closeMenu();
        else openMenu();
      };
      box.appendChild(addBtn);
    }

    function renderMenu() {
      var q = search.value.trim().toLowerCase();
      listEl.innerHTML = "";
      var avail = allOptions.filter(function (o) {
        return selected.indexOf(o) < 0 && (!q || o.toLowerCase().indexOf(q) >= 0);
      });
      if (!avail.length) {
        var em = document.createElement("div");
        em.style.padding = "5px 8px";
        em.style.fontSize = "12px";
        em.style.color = "var(--onera-muted)";
        em.textContent = "No more channels";
        listEl.appendChild(em);
        return;
      }
      avail.forEach(function (opt) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "onera-combobox-item";
        btn.textContent = opt;
        btn.onclick = function (e) {
          e.stopPropagation();
          if (selected.indexOf(opt) < 0) selected.push(opt);
          renderBox();
          renderMenu();
        };
        listEl.appendChild(btn);
      });
    }

    function openMenu() {
      menuOpen = true;
      search.value = "";
      renderMenu();
      menu.classList.remove("hidden");
      search.focus();
    }

    function closeMenu() {
      menuOpen = false;
      menu.classList.add("hidden");
    }

    search.oninput = renderMenu;
    search.onclick = function (e) { e.stopPropagation(); };
    document.addEventListener("click", function (e) {
      if (menuOpen && !containerEl.contains(e.target)) closeMenu();
    });

    return {
      setOptions: function (opts, selVals) {
        allOptions = (opts || []).slice();
        selected = (selVals || []).filter(function (v) { return allOptions.indexOf(v) >= 0; });
        renderBox();
        closeMenu();
      },
      getValues: function () { return selected.slice(); }
    };
  }

  /* ---- Application State ---- */
  var state = {
    recMeta: null,
    ignorePairs: [["", ""]],
    session: null,
    activeEp: 0,
    selectedEps: [],
    showAll: false,
    autoAdvance: true,
    isFS: false,
    showKeys: false,
    progressES: null
  };

  var recCombo = null;
  var refMs = null;
  var visualMs = null;
  var customEegMs = null;
  var manualMs = null;

  /* ---- Recording Config & Ignore Pairs ---- */
  function fillSelect(selEl, opts, selectedVal, noneLabel) {
    selEl.innerHTML = "";
    if (noneLabel !== null && noneLabel !== undefined) {
      var opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = noneLabel;
      selEl.appendChild(opt0);
    }
    (opts || []).forEach(function (v) {
      var o = document.createElement("option");
      o.value = v;
      o.textContent = v;
      selEl.appendChild(o);
    });
    selEl.value = (selectedVal && (opts || []).indexOf(selectedVal) >= 0) ? selectedVal : (noneLabel !== undefined && noneLabel !== null ? "" : (opts[0] || ""));
  }

  function renderIgnorePairs() {
    var wrap = $("ignore_pairs_wrap");
    wrap.innerHTML = "";
    var markers = (state.recMeta && state.recMeta.marker_labels) || [];
    state.ignorePairs.forEach(function (pair, idx) {
      var row = document.createElement("div");
      row.className = "ignore-pair-row";

      var f1 = document.createElement("div");
      f1.className = "field";
      var l1 = document.createElement("label");
      l1.className = "field-label";
      l1.textContent = "Ignore Pair #" + (idx + 1) + " Start:";
      var s1 = document.createElement("select");
      s1.className = "select-input";
      fillSelect(s1, markers, pair[0], "(None)");
      s1.onchange = function () { state.ignorePairs[idx][0] = s1.value; };
      f1.appendChild(l1);
      f1.appendChild(s1);

      var f2 = document.createElement("div");
      f2.className = "field";
      var l2 = document.createElement("label");
      l2.className = "field-label";
      l2.textContent = "Ignore Pair #" + (idx + 1) + " Stop:";
      var s2 = document.createElement("select");
      s2.className = "select-input";
      fillSelect(s2, markers, pair[1], "(None)");
      s2.onchange = function () { state.ignorePairs[idx][1] = s2.value; };
      f2.appendChild(l2);
      f2.appendChild(s2);

      row.appendChild(f1);
      row.appendChild(f2);
      wrap.appendChild(row);
    });
    $("remove_ignore_pair_btn").disabled = state.ignorePairs.length <= 1;
  }

  function renderRecordingMeta(rec) {
    state.recMeta = rec;
    $("fs_input").value = rec.sfreq_detected || 500;
    $("session_err_callout").classList.add("hidden");
    $("recording_meta_card").classList.remove("hidden");
    $("config_section").classList.remove("hidden");

    function metaPill(lbl, val) {
      return "<span class='meta-pill'><span class='meta-pill-lbl'>" + escHtml(lbl) + "</span><strong>" + escHtml(val) + "</strong></span>";
    }
    $("recording_meta_pills").innerHTML =
      metaPill("Recording:", rec.recording_id || rec.filename) +
      metaPill("Duration:", rec.duration_hms) +
      metaPill("Epochs:", rec.n_epochs_est + " (30s)") +
      metaPill("Channels:", String(rec.n_channels));

    var markers = rec.marker_labels || [];
    if (markers.length) {
      $("recording_marker_pills").innerHTML =
        "<span style='font-size:12px;color:var(--onera-muted);font-weight:600;'>Markers:</span>" +
        markers.map(function (m) {
          var low = m.toLowerCase().replace(/[_\s-]/g, "");
          var cls = (low.indexOf("lightsoff") >= 0 || low.indexOf("lightoff") >= 0 || low.indexOf("lightson") >= 0)
            ? "is-lights"
            : (isCalibrationMarker(m) ? "is-cal" : "is-event");
          return "<span class='marker-pill " + cls + "'>" + escHtml(m) + "</span>";
        }).join("");
    } else {
      $("recording_marker_pills").innerHTML =
        "<span style='font-size:12px;color:var(--onera-muted);font-weight:600;'>Markers:</span>" +
        "<span style='font-size:12px;color:var(--onera-muted);'>None</span>";
    }

    var defs = rec.defaults || {};
    fillSelect($("lightsoff_select"), markers, defs.lightsoff || "", "(None — Start from Sample 1)");
    state.ignorePairs = [[defs.ignore_start_1 || "", defs.ignore_stop_1 || ""]];
    renderIgnorePairs();

    var allChs = rec.ch_names || [];
    var eegCands = defs.eeg_candidates || allChs;
    var nonRef = defs.non_ref_chs || allChs;

    refMs.setOptions(allChs, defs.refs || []);
    fillSelect($("primary_eeg_select"), eegCands, defs.primary_eeg || "", null);
    fillSelect($("eog_select"), allChs, defs.eog || "", "(None)");
    fillSelect($("emg_select"), allChs, defs.emg || "", "(None)");
    visualMs.setOptions(nonRef, defs.visual_chs || []);
    customEegMs.setOptions(eegCands, defs.custom_eegs || []);
    manualMs.setOptions(nonRef, defs.manual_chs || []);
  }

  function selectRecording(recKey) {
    if (!recKey) return;
    apiJson("/api/recording/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recording_key: recKey })
    })
      .then(renderRecordingMeta)
      .catch(function (err) {
        $("recording_meta_card").classList.add("hidden");
        $("session_err_callout").textContent = "⚠️ Session Error: " + err.message;
        $("session_err_callout").classList.remove("hidden");
      });
  }

  function scanDirectories() {
    var dataDir = $("data_dir_input").value.trim();
    var outputDir = $("output_dir_input").value.trim();
    var callout = $("scan_callout");
    callout.className = "onera-callout is-info";
    callout.textContent = "Scanning directory...";
    apiJson("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_dir: dataDir, output_dir: outputDir })
    })
      .then(function (res) {
        var recs = res.recordings || [];
        if (res.warning) {
          callout.className = "onera-callout is-warn";
          callout.textContent = "⚠️ " + res.warning;
          $("recording_selector_row").classList.add("hidden");
          $("recording_meta_card").classList.add("hidden");
          $("config_section").classList.add("hidden");
          return;
        }
        callout.className = "onera-callout is-info";
        callout.innerHTML = "Found <strong>" + res.count + "</strong> recording(s).";
        $("recording_selector_row").classList.remove("hidden");
        recCombo.setOptions(recs, recs[0] || "");
        if (recs.length) selectRecording(recs[0]);
      })
      .catch(function (err) {
        callout.className = "onera-callout is-danger";
        callout.textContent = "❌ " + err.message;
      });
  }

  /* ---- Consensus Label Inference ---- */
  function stripScoringExt(name) {
    var s = String(name || "").trim();
    if (/\.onera\.json$/i.test(s)) return s.slice(0, -11);
    return s.replace(/\.[^.]+$/, "");
  }

  function inferScorerLabels(name1, name2) {
    var s1 = stripScoringExt(name1);
    var s2 = stripScoringExt(name2);
    if (!s1 || !s2 || s1 === s2) return ["Scorer 1", "Scorer 2"];

    var maxP = Math.min(s1.length, s2.length);
    var pLen = 0;
    while (pLen < maxP && s1[pLen] === s2[pLen]) pLen++;
    if (pLen > 0 && pLen < maxP && !/[_\-.\s]/.test(s1[pLen - 1])) {
      var sepPre = Math.max(s1.lastIndexOf("_", pLen - 1), s1.lastIndexOf("-", pLen - 1), s1.lastIndexOf(".", pLen - 1), s1.lastIndexOf(" ", pLen - 1));
      pLen = sepPre >= 0 ? (sepPre + 1) : 0;
    }

    var r1 = s1.slice(pLen);
    var r2 = s2.slice(pLen);
    var maxS = Math.min(r1.length, r2.length);
    var sLen = 0;
    while (sLen < maxS && r1[r1.length - 1 - sLen] === r2[r2.length - 1 - sLen]) sLen++;
    if (sLen > 0 && sLen < maxS && !/[_\-.\s]/.test(r1[r1.length - sLen])) {
      var tail1 = r1.slice(r1.length - sLen);
      var sepSuf = tail1.search(/[_\-.\s]/);
      sLen = sepSuf >= 0 ? (tail1.length - sepSuf) : 0;
    }

    var lbl1 = (sLen > 0 ? r1.slice(0, -sLen) : r1).replace(/^[_\-.\s]+|[_\-.\s]+$/g, "").slice(0, 24);
    var lbl2 = (sLen > 0 ? r2.slice(0, -sLen) : r2).replace(/^[_\-.\s]+|[_\-.\s]+$/g, "").slice(0, 24);
    return [lbl1 || "Scorer 1", lbl2 || "Scorer 2"];
  }

  function syncConsensusLabelsFromFiles() {
    var f1 = $("cons_s1_file_input").files && $("cons_s1_file_input").files[0];
    var f2 = $("cons_s2_file_input").files && $("cons_s2_file_input").files[0];
    var inferred = (f1 && f2) ? inferScorerLabels(f1.name, f2.name) : ["Scorer 1", "Scorer 2"];
    var in1 = $("cons_s1_label_input");
    var in2 = $("cons_s2_label_input");
    if (in1 && in1.dataset.userEdited !== "1") in1.value = (f1 && f2 && inferred[0] !== "Scorer 1") ? inferred[0] : "";
    if (in2 && in2.dataset.userEdited !== "1") in2.value = (f1 && f2 && inferred[1] !== "Scorer 2") ? inferred[1] : "";
  }

  /* ---- Scoring Start & SSE Progress ---- */
  function startScoring(mode) {
    if (!state.recMeta) return;
    if (state.progressES) {
      try { state.progressES.close(); } catch (e) {}
      state.progressES = null;
    }

    if (mode === "consensus") {
      var c1 = $("cons_s1_file_input").files && $("cons_s1_file_input").files[0];
      var c2 = $("cons_s2_file_input").files && $("cons_s2_file_input").files[0];
      if (!c1 || !c2) {
        $("progress_section").classList.remove("hidden");
        $("progress_card").classList.add("hidden");
        $("scoring_error_callout").textContent = "❌ Please select both Scorer 1 and Scorer 2 scoring files before starting Consensus Mode.";
        $("scoring_error_callout").classList.remove("hidden");
        return;
      }
    }

    var withYasa = $("cons_with_yasa_chk").checked;
    var cfg = {
      mode: mode,
      fs: Number($("fs_input").value) || state.recMeta.sfreq_detected || 500,
      lightsoff: $("lightsoff_select").value,
      ignore_pairs: state.ignorePairs.slice(),
      refs: refMs.getValues(),
      primary_eeg: $("primary_eeg_select").value,
      eog: $("eog_select").value,
      emg: $("emg_select").value,
      visual_chs: visualMs.getValues(),
      multi_electrode: $("multi_electrode_chk").checked,
      electrode_preset: $("preset_select").value,
      custom_eegs: customEegMs.getValues(),
      confidence_thresh: Number($("confidence_slider").value) || 60,
      exclude_forced_wake: $("exclude_forced_wake_chk").checked,
      manual_chs: manualMs.getValues(),
      resume_with_yasa: $("resume_yasa_chk").checked,
      scorer1_label: ($("cons_s1_label_input").value || "").trim() || "Scorer 1",
      scorer2_label: ($("cons_s2_label_input").value || "").trim() || "Scorer 2",
      flag_ma_diffs: $("cons_flag_ma_chk").checked,
      consensus_with_yasa: withYasa,
      auto_resolve_yasa: withYasa && $("cons_auto_resolve_chk").checked
    };

    var fd = new FormData();
    fd.append("config", JSON.stringify(cfg));
    if (mode === "resume") {
      var files = $("resume_file_input").files;
      if (files && files[0]) fd.append("resume_file", files[0]);
    } else if (mode === "consensus") {
      var s1Files = $("cons_s1_file_input").files;
      var s2Files = $("cons_s2_file_input").files;
      if (s1Files && s1Files[0]) fd.append("scorer1_file", s1Files[0]);
      if (s2Files && s2Files[0]) fd.append("scorer2_file", s2Files[0]);
    }

    $("progress_section").classList.remove("hidden");
    $("progress_card").classList.remove("hidden");
    $("scoring_error_callout").classList.add("hidden");
    $("progress_title").textContent = "Starting...";
    $("progress_pct").textContent = "0%";
    $("progress_fill").style.width = "0%";
    $("progress_subtitle").textContent = "Submitting request...";

    fetch("/api/score/start", { method: "POST", body: fd })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (d) {
          if (!res.ok) throw new Error(d.detail || ("Error " + res.status));
          return d;
        });
      })
      .then(function (data) {
        var jobId = data.job_id;
        var es = new EventSource("/api/score/progress/" + encodeURIComponent(jobId));
        state.progressES = es;

        function handleJobUpdate(job) {
          if (!job) return;
          if (job.title) $("progress_title").textContent = job.title;
          if (typeof job.pct === "number") {
            $("progress_pct").textContent = job.pct + "%";
            $("progress_fill").style.width = job.pct + "%";
          }
          if (job.subtitle) $("progress_subtitle").textContent = job.subtitle;
          if (job.status === "done") {
            es.close();
            state.progressES = null;
            apiJson("/api/session").then(function (sess) {
              $("progress_card").classList.add("hidden");
              $("progress_section").classList.add("hidden");
              applySession(sess);
            });
          } else if (job.status === "error") {
            es.close();
            state.progressES = null;
            $("progress_card").classList.add("hidden");
            $("scoring_error_callout").textContent = "❌ Scoring Error: " + (job.error || "Unknown error");
            $("scoring_error_callout").classList.remove("hidden");
          }
        }

        es.onmessage = function (ev) {
          try { handleJobUpdate(JSON.parse(ev.data)); } catch (e) {}
        };
        es.onerror = function () {
          es.close();
          state.progressES = null;
          apiJson("/api/score/status/" + encodeURIComponent(jobId))
            .then(handleJobUpdate)
            .catch(function (err) {
              $("progress_card").classList.add("hidden");
              $("scoring_error_callout").textContent = "❌ Scoring Error: " + err.message;
              $("scoring_error_callout").classList.remove("hidden");
            });
        };
      })
      .catch(function (err) {
        $("progress_card").classList.add("hidden");
        $("scoring_error_callout").textContent = "❌ Scoring Error: " + err.message;
        $("scoring_error_callout").classList.remove("hidden");
      });
  }

  /* ---- Workspace Rendering & Derived Hypnogram State ---- */
  function computeHypnoDerived() {
    var sess = state.session;
    var nEp = sess.n_ep;
    var isManual = sess.is_manual_mode;
    var ep = sess.epochs;
    var finals = ep.final_stage;
    var flagged = ep.is_flagged;
    var edited = ep.is_manual_edit;
    var hovers = ep.hover_text;
    var ovSrcs = ep.override_source;
    var baseStats = ep.base_status;

    var firstScoredY = 4.0;
    for (var fi = 0; fi < nEp; fi++) {
      if (finals[fi] in STAGE_Y_MAP) { firstScoredY = STAGE_Y_MAP[finals[fi]]; break; }
    }
    var lastScoredY = null;
    var stageY = new Array(nEp);
    var artIdx = [];
    var isArtArr = new Uint8Array(nEp);

    for (var i = 0; i < nEp; i++) {
      var st = finals[i];
      var yv;
      if (st in STAGE_Y_MAP) {
        yv = STAGE_Y_MAP[st];
        lastScoredY = yv;
      } else if (st === "ART") {
        yv = lastScoredY !== null ? lastScoredY : firstScoredY;
        artIdx.push(i);
        isArtArr[i] = 1;
      } else if (st === "UNS") {
        yv = isManual ? -0.65 : 4.0;
      } else {
        yv = 4.0;
      }
      stageY[i] = yv;
    }

    var isCons = Boolean(sess.is_consensus_mode && sess.consensus);
    var s2StageY = null;
    var s2DiffMask = null;
    var resolvedSet = null;
    if (isCons) {
      var s2Arr = sess.consensus.scorer2_stage || [];
      var resolvedArr = (sess.consensus.counts && sess.consensus.counts.resolved_indices) || [];
      resolvedSet = new Set(resolvedArr);
      s2StageY = new Array(nEp);
      s2DiffMask = new Uint8Array(nEp);
      for (var si2 = 0; si2 < nEp; si2++) {
        var s2St = s2Arr[si2];
        if (s2St && s2St !== finals[si2] && (s2St in STAGE_Y_MAP)) {
          s2StageY[si2] = STAGE_Y_MAP[s2St];
          s2DiffMask[si2] = 1;
        } else {
          s2StageY[si2] = -1;
        }
      }
    }

    var selSet = new Set(state.selectedEps);
    var visIdx = [];
    var visKind = [];
    var visR = [];
    var visHover = [];
    var lblIdx = [];
    var lblEd = [];

    for (var j = 0; j < nEp; j++) {
      var visible = isManual || state.showAll || flagged[j] || edited[j] || isArtArr[j];
      if (!visible) continue;
      visIdx.push(j);
      visHover.push(hovers[j]);
      var isSel = selSet.has(j);
      var isEd = Boolean(edited[j]);
      var isFl = Boolean(flagged[j]);
      var isA = Boolean(isArtArr[j]);
      var isRes = isCons ? Boolean(isEd || (resolvedSet && resolvedSet.has(j))) : isEd;

      if (isManual) {
        if (isA) {
          visKind.push("art"); visR.push(isSel ? 5.0 : 4.2);
        } else if (isSel) {
          visKind.push("sel"); visR.push(5.4);
        } else if (isEd) {
          visKind.push("edit"); visR.push(4.0);
        } else if (baseStats[j] === "Imported") {
          visKind.push("imp"); visR.push(3.8);
        } else if (ovSrcs[j]) {
          visKind.push("wake"); visR.push(3.8);
        } else {
          visKind.push("uns"); visR.push(3.0);
        }
      } else {
        if (isA) {
          visKind.push("art"); visR.push(isSel ? 5.0 : 4.2);
        } else {
          if (isCons ? (isEd || (isFl && isRes)) : isEd) visKind.push("edit");
          else if (isSel) visKind.push("sel");
          else if (isFl) visKind.push("flag");
          else visKind.push("norm");
          visR.push(isSel ? 5.4 : ((isFl || isEd) ? 4.0 : 3.2));
        }
        if (isEd || isFl || isA) {
          lblIdx.push(j);
          lblEd.push(isA ? 2 : ((isCons ? isRes : isEd) ? 1 : 0));
        }
      }
    }

    return {
      stageY: stageY,
      s2StageY: s2StageY,
      s2DiffMask: s2DiffMask,
      artIdx: artIdx,
      visIdx: visIdx,
      visKind: visKind,
      visR: visR,
      visHover: visHover,
      lblIdx: lblIdx,
      lblEd: lblEd,
      hasS4: Boolean(sess.has_s4)
    };
  }

  function pillHtml(label, value, bg, border, fg, bold, extraCls) {
    var fw = bold ? "700" : "600";
    var cls = ("onera-sum-pill " + (extraCls || "")).trim();
    return (
      "<span class='" + cls + "' style='background:" + (bg || "var(--onera-surface)") +
      ";color:" + (fg || "var(--onera-fg)") + ";border-color:" + (border || "var(--onera-border)") + ";'>" +
      "<span class='onera-sum-pill-lbl'>" + escHtml(label) + "</span>" +
      "<strong style='font-weight:" + fw + ";'>" + escHtml(value) + "</strong></span>"
    );
  }

  function buildSummaryRowsHtml() {
    var sess = state.session;
    var ep = sess.epochs;
    var idx = state.activeEp;
    var nEp = sess.n_ep;
    var isManual = sess.is_manual_mode;
    var epNum = idx + 1;

    var inspectPill = pillHtml(
      "Inspecting",
      "Ep #" + epNum + " (" + ep.time_start[idx] + "–" + ep.time_end[idx] + ")",
      "var(--onera-amber-bg)", "var(--onera-amber-border)", "var(--onera-amber-fg)", true
    );
    var multiSelPill = "";
    if (state.selectedEps.length > 1) {
      var sSorted = state.selectedEps.slice().sort(function (a, b) { return a - b; });
      multiSelPill = pillHtml(
        "Selected:",
        sSorted.length + " Epochs (Ep #" + (sSorted[0] + 1) + "–#" + (sSorted[sSorted.length - 1] + 1) + ") — Click a stage button above to apply to all",
        "var(--onera-lavender-bg)", "var(--onera-lavender-border)", "var(--onera-lavender-fg)", true
      );
    }

    var timings = sess.init_timings;
    var timingPill = "";
    if (timings) {
      var tStr = isManual
        ? ("Load " + timings.load_s.toFixed(1) + "s · Ref " + timings.resample_s.toFixed(1) + "s · Filter " + timings.filter_s.toFixed(1) + "s (" + timings.total_s.toFixed(1) + "s)")
        : ("Load " + timings.load_s.toFixed(1) + "s · Resample " + timings.resample_s.toFixed(1) + "s · YASA " + timings.yasa_s.toFixed(1) + "s · Filter " + timings.filter_s.toFixed(1) + "s (" + timings.total_s.toFixed(1) + "s)");
      timingPill = pillHtml("Init:", tStr, "transparent", "var(--onera-border)", "var(--onera-muted)", false, "onera-fs-hide-pill");
    }

    var rf = sess.resumed_from;
    var resumePills = "";
    if (rf) {
      var resLbl = (rf.sidecar_used && rf.format !== ".onera.json") ? (rf.filename + " (+ session state)") : rf.filename;
      resumePills += pillHtml("Resumed:", resLbl, "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", false);
      if (rf.warning) {
        resumePills += pillHtml("Import Note:", rf.warning, "var(--onera-amber-bg)", "var(--onera-amber-border)", "var(--onera-amber-fg)", false);
      }
    }

    var maSuffix = ep.has_ma[idx] ? " +MA" : "";
    var row1 = "", row2 = "";

    if (isManual) {
      var nUns = 0;
      for (var i = 0; i < nEp; i++) { if (ep.final_stage[i] === "UNS") nUns++; }
      var nScored = nEp - nUns;
      var pctSc = nEp > 0 ? (100.0 * nScored / nEp) : 0.0;
      var visChs = waveView ? waveView.getVisibleChannelNames() : sess.display_chs;
      row1 =
        pillHtml("Recording:", sess.recording_label, null, null, null, false, "onera-fs-hide-pill") +
        pillHtml("Mode:", "Manual", "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", true) +
        pillHtml("Scored:", nScored + " / " + nEp + " (" + pctSc.toFixed(1) + "%)", "var(--onera-sage-bg)", "var(--onera-sage-border)", "var(--onera-sage-fg)", false) +
        pillHtml("Remaining Unscored:", String(nUns), nUns > 0 ? "var(--onera-amber-bg)" : "var(--onera-sage-bg)", nUns > 0 ? "var(--onera-amber-border)" : "var(--onera-sage-border)", nUns > 0 ? "var(--onera-amber-fg)" : "var(--onera-sage-fg)", false) +
        pillHtml("Channels:", visChs.join(", "), null, null, null, false, "onera-fs-hide-pill") +
        resumePills + timingPill;

      var stagePill = "";
      if (ep.is_manual_edit[idx]) {
        var stVal = (ep.auto_stage[idx] !== "UNS" && ep.auto_stage[idx] !== ep.final_stage[idx])
          ? ("Override: " + ep.auto_stage[idx] + " → Manual: " + ep.final_stage[idx] + maSuffix)
          : ("Manual: " + ep.final_stage[idx] + maSuffix);
        stagePill = pillHtml("Stage:", stVal, "var(--onera-sage-bg)", "var(--onera-sage-border)", "var(--onera-sage-fg)", false);
      } else if (ep.final_stage[idx] !== "UNS" && ep.base_status[idx] === "Imported") {
        stagePill = pillHtml("Stage:", ep.final_stage[idx] + maSuffix + " (Imported)", "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", false);
      } else if (ep.override_source[idx]) {
        stagePill = pillHtml("Stage:", ep.final_stage[idx] + maSuffix + " (Wake Override)", "var(--onera-amber-bg)", "var(--onera-amber-border)", "var(--onera-amber-fg)", false);
      } else {
        stagePill = pillHtml("Stage:", "Unscored (UNS)" + maSuffix);
      }

      var epEvs = (sess.epoch_events && sess.epoch_events[String(idx)]) || [];
      var markersPill = epEvs.length
        ? pillHtml("Markers in Epoch:", epEvs.map(function (ev) { return ev[1] + " (" + Number(ev[0]).toFixed(1) + "s)"; }).join(", "), "var(--onera-lavender-bg)", "var(--onera-lavender-border)", "var(--onera-lavender-fg)", false)
        : "";
      row2 = inspectPill + multiSelPill + stagePill + markersPill;
    } else if (sess.is_consensus_mode && sess.consensus) {
      var cons = sess.consensus;
      var irr = cons.irr || {};
      var counts = cons.counts || {};
      var s1Lbl = cons.scorer1_label || "Scorer 1";
      var s2Lbl = cons.scorer2_label || "Scorer 2";
      var agrTxt = (irr.agree_pct !== null && irr.agree_pct !== undefined) ? (Number(irr.agree_pct).toFixed(1) + "%") : "n/a";
      var kapTxt = (irr.kappa !== null && irr.kappa !== undefined) ? Number(irr.kappa).toFixed(3) : "n/a";
      var nRes = Number(counts.n_resolved || 0);
      var nUnres = Number(counts.n_unresolved || 0);
      var nTotC = Number(counts.total_flagged || 0);
      var nTie = Number(cons.n_yasa_resolved || 0);
      var tiePill = cons.auto_resolve_yasa
        ? pillHtml("YASA Tie-Breaks:", String(nTie), "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", false)
        : "";

      row1 =
        pillHtml("Recording:", sess.recording_label, null, null, null, false, "onera-fs-hide-pill") +
        pillHtml("Mode:", sess.has_yasa ? "Consensus + YASA" : "Consensus", "var(--onera-lavender-bg)", "var(--onera-lavender-border)", "var(--onera-lavender-fg)", true) +
        pillHtml("Epochs:", nEp + " (" + formatHms(nEp * 30) + ")") +
        pillHtml("Agreement:", agrTxt, "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", true) +
        pillHtml("Cohen's κ:", kapTxt, "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", true) +
        pillHtml("Resolved:", nRes + " / " + nTotC, "var(--onera-sage-bg)", "var(--onera-sage-border)", "var(--onera-sage-fg)", false) +
        pillHtml("Unresolved:", String(nUnres), nUnres > 0 ? "var(--onera-coral-bg)" : "var(--onera-sage-bg)", nUnres > 0 ? "var(--onera-coral-border)" : "var(--onera-sage-border)", nUnres > 0 ? "var(--onera-coral-fg)" : "var(--onera-sage-fg)", true) +
        tiePill + timingPill;

      var s1Disp = cons.scorer1_stage[idx] + (cons.scorer1_ma[idx] ? "+MA" : "");
      var s2Disp = cons.scorer2_stage[idx] + (cons.scorer2_ma[idx] ? "+MA" : "");
      var stDiff = Boolean(cons.stage_diff[idx]);
      var resolvedHere = (counts.resolved_indices || []).indexOf(idx) >= 0;
      var stTextC = ep.final_stage[idx] + maSuffix + (ep.is_manual_edit[idx] ? " (Resolved)" : (stDiff && !cons.yasa_resolved[idx] ? " (" + s1Lbl + " default)" : " (" + ep.status[idx] + ")"));
      var stStyle = (stDiff && !cons.yasa_resolved[idx] && !ep.is_manual_edit[idx])
        ? ["var(--onera-coral-bg)", "var(--onera-coral-border)", "var(--onera-coral-fg)"]
        : (ep.is_manual_edit[idx] || resolvedHere
            ? ["var(--onera-sage-bg)", "var(--onera-sage-border)", "var(--onera-sage-fg)"]
            : ["var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)"]);

      var yasaPillC = "";
      if (sess.has_yasa && ep.raw_yasa_stage && ep.confidence_pct[idx] !== null) {
        yasaPillC = pillHtml("YASA:", ep.raw_yasa_stage[idx] + " (" + Number(ep.confidence_pct[idx]).toFixed(1) + "%)");
      }

      row2 =
        inspectPill + multiSelPill +
        pillHtml("Final Stage:", stTextC, stStyle[0], stStyle[1], stStyle[2], true) +
        pillHtml(s1Lbl + " vs " + s2Lbl + ":", s1Disp + " / " + s2Disp) +
        yasaPillC +
        pillHtml("Note:", ep.flag_reason[idx]);
    } else {
      var nFlagged = (sess.flagged_indices || []).length;
      var nEdited = 0, sumConf = 0, sumAgr = 0;
      for (var k = 0; k < nEp; k++) {
        if (ep.is_manual_edit[k]) nEdited++;
        sumConf += Number(ep.confidence_pct[k] || 0);
        sumAgr += Number(ep.agreement_pct[k] || 0);
      }
      var meanConf = nEp > 0 ? sumConf / nEp : 0;
      var meanAgr = nEp > 0 ? sumAgr / nEp : 0;
      row1 =
        pillHtml("Recording:", sess.recording_label, null, null, null, false, "onera-fs-hide-pill") +
        pillHtml("Epochs:", nEp + " (" + formatHms(nEp * 30) + ")") +
        pillHtml("EEG:", (sess.active_eegs || []).join(", "), null, null, null, false, "onera-fs-hide-pill") +
        pillHtml("Mean Conf:", meanConf.toFixed(1) + "%") +
        pillHtml("Agreement:", meanAgr.toFixed(1) + "%") +
        pillHtml("Flagged (<" + Math.round(sess.thresh) + "%):", String(nFlagged), nFlagged > 0 ? "var(--onera-coral-bg)" : "var(--onera-sage-bg)", nFlagged > 0 ? "var(--onera-coral-border)" : "var(--onera-sage-border)", nFlagged > 0 ? "var(--onera-coral-fg)" : "var(--onera-sage-fg)") +
        pillHtml("Manual Edits:", String(nEdited), nEdited > 0 ? "var(--onera-sage-bg)" : null, nEdited > 0 ? "var(--onera-sage-border)" : null, nEdited > 0 ? "var(--onera-sage-fg)" : null) +
        resumePills + timingPill;

      var epConf = Number(ep.confidence_pct[idx] || 0);
      var confLow = epConf < Number(sess.thresh);
      var stagePillA = "";
      if (ep.is_manual_edit[idx]) {
        var stText = ep.auto_stage[idx] !== ep.final_stage[idx]
          ? ("Auto: " + ep.auto_stage[idx] + " → Manual: " + ep.final_stage[idx] + maSuffix)
          : ("Manual: " + ep.final_stage[idx] + maSuffix);
        stagePillA = pillHtml("Stage:", stText, "var(--onera-sage-bg)", "var(--onera-sage-border)", "var(--onera-sage-fg)", false);
      } else if (ep.override_source[idx]) {
        stagePillA = pillHtml("Stage:", ep.final_stage[idx] + maSuffix + " (Wake Override)", "var(--onera-amber-bg)", "var(--onera-amber-border)", "var(--onera-amber-fg)", false);
      } else {
        stagePillA = pillHtml("Stage:", ep.final_stage[idx] + maSuffix + " (Auto)", "var(--onera-sky-bg)", "var(--onera-sky-border)", "var(--onera-sky-fg)", false);
      }

      row2 =
        inspectPill + multiSelPill + stagePillA +
        pillHtml("Confidence:", epConf.toFixed(1) + "%", confLow ? "var(--onera-coral-bg)" : "var(--onera-sage-bg)", confLow ? "var(--onera-coral-border)" : "var(--onera-sage-border)", confLow ? "var(--onera-coral-fg)" : "var(--onera-sage-fg)") +
        pillHtml("Diagnostic:", ep.flag_reason[idx]);
    }
    return { row1: row1, row2: row2 };
  }

  function updateSummaryAndSidebar() {
    if (!state.session) return;
    var rows = buildSummaryRowsHtml();
    $("summary_row_global").innerHTML = rows.row1;
    $("summary_row_epoch").innerHTML = rows.row2;

    if (!state.session.is_manual_mode) {
      var sess = state.session;
      var idx = state.activeEp;
      var ep = sess.epochs;
      var thresh = Number(sess.thresh);
      var isConsMode = Boolean(sess.is_consensus_mode && sess.consensus);
      var hasYasa = Boolean(sess.has_yasa);

      $("prob_ep_num").textContent = "Ep #" + (idx + 1);
      var badge = $("prob_cue_badge");

      if (isConsMode) {
        var cons = sess.consensus;
        var counts = cons.counts || {};
        var irr = cons.irr || {};
        var s1Lbl = cons.scorer1_label || "Scorer 1";
        var s2Lbl = cons.scorer2_label || "Scorer 2";
        var s1St = cons.scorer1_stage[idx];
        var s2St = cons.scorer2_stage[idx];
        var s1Disp = s1St + (cons.scorer1_ma[idx] ? " +MA" : "");
        var s2Disp = s2St + (cons.scorer2_ma[idx] ? " +MA" : "");
        var curFinal = ep.final_stage[idx];
        var curIsResolved = (counts.resolved_indices || []).indexOf(idx) >= 0;
        var stDiff = Boolean(cons.stage_diff[idx]);
        var maDiff = Boolean(cons.ma_diff[idx]);

        if (ep.is_manual_edit[idx] || curIsResolved) {
          badge.textContent = "Resolved";
          badge.style.background = "var(--onera-sage-bg)";
          badge.style.color = "var(--onera-sage-fg)";
          badge.style.borderColor = "var(--onera-sage-border)";
        } else if (cons.yasa_resolved[idx]) {
          badge.textContent = "YASA Tie";
          badge.style.background = "var(--onera-sky-bg)";
          badge.style.color = "var(--onera-sky-fg)";
          badge.style.borderColor = "var(--onera-sky-border)";
        } else if (stDiff) {
          badge.textContent = "Conflict";
          badge.style.background = "var(--onera-coral-bg)";
          badge.style.color = "var(--onera-coral-fg)";
          badge.style.borderColor = "var(--onera-coral-border)";
        } else if (cons.flag_ma_diffs && maDiff) {
          badge.textContent = "+MA Diff";
          badge.style.background = "var(--onera-amber-bg)";
          badge.style.color = "var(--onera-amber-fg)";
          badge.style.borderColor = "var(--onera-amber-border)";
        } else {
          badge.textContent = "Agreed";
          badge.style.background = "var(--onera-sage-bg)";
          badge.style.color = "var(--onera-sage-fg)";
          badge.style.borderColor = "var(--onera-sage-border)";
        }

        $("cons_s1_btn_lbl").textContent = "Accept " + s1Lbl;
        $("cons_s1_btn_stage").textContent = s1Disp;
        $("cons_s2_btn_lbl").textContent = "Accept " + s2Lbl;
        $("cons_s2_btn_stage").textContent = s2Disp;
        var b1 = $("cons_accept_s1_btn");
        var b2 = $("cons_accept_s2_btn");
        var s1Match = curFinal === s1St && (!cons.flag_ma_diffs || Boolean(ep.has_ma[idx]) === Boolean(cons.scorer1_ma[idx]));
        var s2Match = curFinal === s2St && (!cons.flag_ma_diffs || Boolean(ep.has_ma[idx]) === Boolean(cons.scorer2_ma[idx]));
        b1.classList.toggle("is-chosen", s1Match && (ep.is_manual_edit[idx] || !stDiff));
        b2.classList.toggle("is-chosen", s2Match && (ep.is_manual_edit[idx] || !stDiff));

        var agrStr = (irr.agree_pct !== null && irr.agree_pct !== undefined) ? (Number(irr.agree_pct).toFixed(1) + "%") : "n/a";
        var kapStr = (irr.kappa !== null && irr.kappa !== undefined) ? Number(irr.kappa).toFixed(3) : "n/a";
        var tieNote = cons.auto_resolve_yasa ? ("<div>YASA Ties Broken: <strong>" + Number(cons.n_yasa_resolved || 0) + "</strong></div>") : "";
        $("cons_irr_box").innerHTML =
          "<div>Agree: <strong>" + escHtml(agrStr) + "</strong> · κ = <strong>" + escHtml(kapStr) + "</strong></div>" +
          "<div>Resolved: <strong>" + Number(counts.n_resolved || 0) + "</strong> / " + Number(counts.total_flagged || 0) +
          " (<strong>" + Number(counts.n_unresolved || 0) + "</strong> left)</div>" +
          tieNote;
      } else {
        var epConf = Number(ep.confidence_pct[idx] || 0);
        var lowConf = epConf < thresh;
        badge.textContent = lowConf ? ("Low <" + Math.round(thresh) + "%") : ("≥" + Math.round(thresh) + "%");
        badge.style.background = lowConf ? "var(--onera-coral-bg)" : "var(--onera-sage-bg)";
        badge.style.color = lowConf ? "var(--onera-coral-fg)" : "var(--onera-sage-fg)";
        badge.style.borderColor = lowConf ? "var(--onera-coral-border)" : "var(--onera-sage-border)";
      }

      if (hasYasa) {
        var winStage = isConsMode && ep.raw_yasa_stage ? ep.raw_yasa_stage[idx] : ep.auto_stage[idx];
        var thLeft = Math.min(99.0, Math.max(1.0, thresh));
        var probs = (ep.probs && ep.probs[idx]) || [0, 0, 0, 0, 0];
        var votesStr = isConsMode && ep.raw_yasa_stage
          ? ("YASA (" + winStage + "): " + (ep.pass_votes[idx] || "—"))
          : ep.pass_votes[idx];
        $("prob_votes").innerHTML = "Votes: <strong style='color:var(--onera-fg);'>" + escHtml(votesStr) + "</strong>";
        $("prob_thresh_lbl").textContent = "Thresh " + Math.round(thresh) + "%";

        var barsHtml = "";
        for (var si = 0; si < STAGE_ORDER.length; si++) {
          var sname = STAGE_ORDER[si];
          var pv = Number(probs[si] || 0);
          var cols = STAGE_PASTEL[sname];
          var isW = sname === winStage;
          var pvW = Math.min(100.0, Math.max(0.0, pv));
          barsHtml +=
            "<div class='onera-prob-bar" + (isW ? " is-winner" : "") + "'" + (isW ? " style='border-color:" + cols[1] + ";'" : "") +
            " title='" + escHtml(sname) + ": " + pv.toFixed(1) + "% (threshold " + Math.round(thresh) + "%)'>" +
            "<span class='onera-prob-fill' style='width:" + pvW.toFixed(1) + "%;background:" + cols[0] + ";'></span>" +
            "<span class='onera-prob-thresh' style='left:" + thLeft.toFixed(1) + "%;'></span>" +
            "<span class='onera-prob-lbl'>" + escHtml(sname) + "</span>" +
            "<span class='onera-prob-val'>" + pv.toFixed(1) + "%</span></div>";
        }
        $("prob_bars").innerHTML = barsHtml;
      }

      $("prob_summary_fs").innerHTML =
        "<div class='onera-prob-sum-grp'>" + rows.row2 + "</div>" +
        "<div class='onera-prob-sum-grp'>" + rows.row1 + "</div>";
    }
  }

  function updateAddChannelSelect() {
    if (!state.session || !waveView) return;
    var vis = waveView.getVisibleChannelNames();
    var avail = (state.session.all_recording_chs || []).filter(function (c) { return vis.indexOf(c) < 0; });
    var sel = $("onera_add_ch_select");
    sel.innerHTML = "<option value=''>+ Channel</option>";
    avail.forEach(function (c) {
      var o = document.createElement("option");
      o.value = c;
      o.textContent = c;
      sel.appendChild(o);
    });
    sel.style.visibility = avail.length ? "visible" : "hidden";
  }

  function updateExportConflictCallout(nUnresOverride, s1LabelOverride) {
    var el = $("export_conflict_callout");
    if (!el) return;
    var nUnres = 0;
    var s1Lbl = s1LabelOverride || "Scorer 1";
    if (typeof nUnresOverride === "number") {
      nUnres = nUnresOverride;
      if (state.session && state.session.consensus && state.session.consensus.scorer1_label) {
        s1Lbl = state.session.consensus.scorer1_label;
      }
    } else if (state.session && state.session.is_consensus_mode && state.session.consensus) {
      var counts = state.session.consensus.counts || {};
      nUnres = Number(counts.n_unresolved || 0);
      s1Lbl = state.session.consensus.scorer1_label || "Scorer 1";
    }
    if (nUnres > 0) {
      el.textContent = "⚠️ " + nUnres + " disagreement(s) remain unresolved in Consensus Mode. Unresolved stage conflicts automatically default to " + s1Lbl + " upon export.";
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  }

  function updateToolbarState() {
    if (!state.session) return;
    var isManual = state.session.is_manual_mode;
    var jumpList = state.session.jump_indices || [];
    $("btn_prev_flag").textContent = isManual ? "⏮️ Prev Unscored" : "⏮️ Prev Flag";
    $("btn_next_flag").textContent = isManual ? "⏭️ Next Unscored" : "⏭️ Next Flag";
    $("btn_prev_flag").disabled = !isManual && jumpList.length === 0;
    $("btn_next_flag").disabled = !isManual && jumpList.length === 0;
    $("btn_revert").disabled = !state.session.can_undo;
    var hasSel = state.selectedEps.length > 0;
    ["btn_stage_w", "btn_stage_n1", "btn_stage_n2", "btn_stage_n3", "btn_stage_s4", "btn_stage_rem", "btn_stage_art", "btn_toggle_ma"].forEach(function (id) {
      $(id).disabled = !hasSel;
    });
    if ($("cons_accept_s1_btn")) $("cons_accept_s1_btn").disabled = !hasSel;
    if ($("cons_accept_s2_btn")) $("cons_accept_s2_btn").disabled = !hasSel;
  }

  function applyPatchResponse(res) {
    if (!state.session || !res) return;
    var ep = state.session.epochs;
    (res.patches || []).forEach(function (p) {
      var i = p.idx;
      ep.final_stage[i] = p.final_stage;
      ep.is_manual_edit[i] = p.is_manual_edit;
      ep.has_ma[i] = p.has_ma;
      ep.status[i] = p.status;
      ep.hover_text[i] = p.hover_text;
    });
    state.session.can_undo = Boolean(res.can_undo);
    state.session.has_s4 = Boolean(res.has_s4);
    state.session.jump_indices = res.jump_indices || [];
    if (state.session.consensus && res.consensus_counts) {
      state.session.consensus.counts = res.consensus_counts;
    }
    updateExportConflictCallout();
  }

  function applySession(sess) {
    state.session = sess;
    state.activeEp = sess.init_active_ep || 0;
    state.selectedEps = (sess.init_selected_eps || [state.activeEp]).slice();
    state.showAll = false;
    $("chk_show_all").checked = false;

    var isManual = Boolean(sess.is_manual_mode);
    var isCons = Boolean(sess.is_consensus_mode);
    var hasYasa = Boolean(sess.has_yasa);
    var modeAttr = isCons ? "consensus" : (isManual ? "manual" : "auto");
    $("inspector_card").setAttribute("data-onera-mode", modeAttr);
    $("summary_bar").setAttribute("data-onera-mode", modeAttr);
    $("lbl_show_all_wrap").classList.toggle("hidden", isManual);
    $("prob_panel").classList.toggle("hidden", isManual);
    $("consensus_sidebar_block").classList.toggle("hidden", !isCons);
    $("yasa_sidebar_block").classList.toggle("hidden", !hasYasa);

    $("hypno_title").textContent = (isCons ? "Interactive Consensus Hypnogram — " : (isManual ? "Interactive Manual Hypnogram — " : "Interactive Hypnogram — ")) + sess.recording_label;
    var s2Name = (sess.consensus && sess.consensus.scorer2_label) ? sess.consensus.scorer2_label : "Scorer 2";
    var legendItems = isManual ? [
      "<span style='display:inline-block;width:12px;height:2px;background:var(--onera-sky-solid);vertical-align:middle;margin-right:3px;'></span>Stage",
      "<span style='display:inline-block;width:10px;height:2.5px;background:var(--onera-coral-dot);vertical-align:middle;margin-right:2px;'></span><span style='display:inline-block;width:7px;height:7px;background:var(--onera-coral-dot);border:1px solid var(--onera-surface);vertical-align:middle;margin-right:3px;'></span>ART",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-sage-dot);vertical-align:middle;margin-right:3px;'></span>Edited",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-sky-solid);vertical-align:middle;margin-right:3px;'></span>Imported",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-amber-ring);vertical-align:middle;margin-right:3px;'></span>Wake Ov.",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-muted);vertical-align:middle;margin-right:3px;'></span>Unscored",
      "<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--onera-amber-dot);border:2px solid var(--onera-amber-ring);vertical-align:middle;margin-right:3px;'></span>Selected"
    ] : (isCons ? [
      "<span style='display:inline-block;width:12px;height:2px;background:var(--onera-sky-solid);vertical-align:middle;margin-right:3px;'></span>Consensus",
      "<span style='display:inline-block;width:14px;height:0;border-top:2px dashed var(--onera-lavender-solid);vertical-align:middle;margin-right:3px;'></span>" + escHtml(s2Name) + " Diff",
      "<span style='display:inline-block;width:10px;height:2.5px;background:var(--onera-coral-dot);vertical-align:middle;margin-right:2px;'></span><span style='display:inline-block;width:7px;height:7px;background:var(--onera-coral-dot);border:1px solid var(--onera-surface);vertical-align:middle;margin-right:3px;'></span>ART",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-coral-dot);vertical-align:middle;margin-right:3px;'></span>Unresolved",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-sage-dot);vertical-align:middle;margin-right:3px;'></span>Resolved",
      "<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--onera-amber-dot);border:2px solid var(--onera-amber-ring);vertical-align:middle;margin-right:3px;'></span>Selected"
    ] : [
      "<span style='display:inline-block;width:12px;height:2px;background:var(--onera-sky-solid);vertical-align:middle;margin-right:3px;'></span>Stage",
      "<span style='display:inline-block;width:10px;height:2.5px;background:var(--onera-coral-dot);vertical-align:middle;margin-right:2px;'></span><span style='display:inline-block;width:7px;height:7px;background:var(--onera-coral-dot);border:1px solid var(--onera-surface);vertical-align:middle;margin-right:3px;'></span>ART",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-coral-dot);vertical-align:middle;margin-right:3px;'></span>Flagged",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-sage-dot);vertical-align:middle;margin-right:3px;'></span>Edited",
      "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--onera-sky-solid);vertical-align:middle;margin-right:3px;'></span>Normal",
      "<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--onera-amber-dot);border:2px solid var(--onera-amber-ring);vertical-align:middle;margin-right:3px;'></span>Selected"
    ]);
    $("hypno_legend").innerHTML = legendItems.map(function (it) { return "<span style='white-space:nowrap;'>" + it + "</span>"; }).join("");

    $("workspace_section").classList.remove("hidden");
    $("export_section").classList.remove("hidden");

    hypnoView.setSession(sess, computeHypnoDerived(), state.activeEp, state.selectedEps);
    waveView.setSession(sess, state.activeEp);
    updateAddChannelSelect();
    updateSummaryAndSidebar();
    updateToolbarState();

    // Initialize Export Panel
    $("export_dir_input").value = sess.default_output_dir || $("output_dir_input").value || ".";
    syncExportFormatFields();
    updateExportConflictCallout();
    $("export_unscored_callout").classList.add("hidden");
    $("export_status_callout").classList.add("hidden");
    $("export_preview_pre").classList.add("hidden");
    $("export_download_btn").classList.add("hidden");
  }

  /* ---- Navigation & Stage Mutations ---- */
  function jumpSingle(newIdx) {
    if (!state.session) return;
    var mx = Math.max(0, state.session.n_ep - 1);
    var clamped = Math.max(0, Math.min(mx, Math.round(Number(newIdx) || 0)));
    state.activeEp = clamped;
    state.selectedEps = [clamped];
    hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, true);
    waveView.jumpToEpoch(clamped);
    updateSummaryAndSidebar();
    updateToolbarState();
  }

  function advanceAfterMutation(targets) {
    var isManual = state.session.is_manual_mode;
    var isCons = Boolean(state.session.is_consensus_mode && state.session.consensus);
    var mx = Math.max(0, state.session.n_ep - 1);
    var ensureVis = false;
    if (state.autoAdvance && targets.length === 1) {
      var curEp = targets[0];
      if (isManual) {
        var nextEp = Math.min(curEp + 1, mx);
        state.activeEp = nextEp;
        state.selectedEps = [nextEp];
        ensureVis = true;
        waveView.jumpToEpoch(nextEp);
      } else if (isCons) {
        var unres = (state.session.consensus.counts && state.session.consensus.counts.unresolved_indices) || [];
        var nxtC = unres.find(function (i) { return i > curEp; });
        if (nxtC === undefined && unres.length) nxtC = unres[0];
        if (nxtC !== undefined) {
          state.activeEp = nxtC;
          state.selectedEps = [nxtC];
          ensureVis = true;
          waveView.jumpToEpoch(nxtC);
        } else {
          state.selectedEps = [];
        }
      } else {
        var flAll = state.session.flagged_indices || [];
        var editedArr = state.session.epochs.is_manual_edit;
        var uneditedFl = flAll.filter(function (i) { return !editedArr[i]; });
        var nxt = uneditedFl.find(function (i) { return i > curEp; });
        if (nxt === undefined && uneditedFl.length) nxt = uneditedFl[0];
        if (nxt !== undefined) {
          state.activeEp = nxt;
          state.selectedEps = [nxt];
          ensureVis = true;
          waveView.jumpToEpoch(nxt);
        } else {
          state.selectedEps = [];
        }
      }
    } else {
      state.selectedEps = [];
    }
    hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, ensureVis);
    updateSummaryAndSidebar();
    updateToolbarState();
  }

  function applyStage(newStage) {
    if (!state.session || !state.selectedEps.length) return;
    var targets = state.selectedEps.slice().sort(function (a, b) { return a - b; });
    apiJson("/api/edits/stage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: targets, stage: newStage })
    }).then(function (res) {
      applyPatchResponse(res);
      advanceAfterMutation(targets);
    }).catch(function (err) {
      alert("Edit failed: " + err.message);
    });
  }

  function acceptConsensusScorer(scorerNum) {
    if (!state.session || !state.selectedEps.length) return;
    var targets = state.selectedEps.slice().sort(function (a, b) { return a - b; });
    apiJson("/api/edits/consensus_accept", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: targets, scorer: scorerNum })
    }).then(function (res) {
      applyPatchResponse(res);
      advanceAfterMutation(targets);
    }).catch(function (err) {
      alert("Consensus accept failed: " + err.message);
    });
  }

  function toggleMa() {
    if (!state.session || !state.selectedEps.length) return;
    apiJson("/api/edits/ma", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: state.selectedEps })
    }).then(function (res) {
      applyPatchResponse(res);
      hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, false);
      updateSummaryAndSidebar();
      updateToolbarState();
    }).catch(function (err) {
      alert("+MA toggle failed: " + err.message);
    });
  }

  function revertLast() {
    if (!state.session) return;
    apiJson("/api/edits/undo", { method: "POST" }).then(function (res) {
      applyPatchResponse(res);
      var restored = res.restored_epochs || [];
      if (restored.length) {
        state.activeEp = restored[0];
        state.selectedEps = restored.slice();
        waveView.jumpToEpoch(state.activeEp);
        hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, true);
      }
      updateSummaryAndSidebar();
      updateToolbarState();
    });
  }

  function clearAllEdits() {
    if (!state.session) return;
    apiJson("/api/edits/clear", { method: "POST" }).then(function (res) {
      applyPatchResponse(res);
      state.selectedEps = [];
      hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, false);
      updateSummaryAndSidebar();
      updateToolbarState();
    });
  }

  function unselectEpochs() {
    if (!state.session) return;
    state.selectedEps = [];
    hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, false);
    updateSummaryAndSidebar();
    updateToolbarState();
  }

  /* ---- Full-Screen Workspace ---- */
  function getFsEl() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }

  function syncFSUI() {
    document.documentElement.classList.toggle("onera-fs-active", state.isFS);
    var fsBtn = $("onera_fs_btn");
    if (fsBtn) {
      fsBtn.textContent = state.isFS ? "✕ Exit Full Screen" : "⛶ Full Screen";
      fsBtn.classList.toggle("is-active", state.isFS);
    }
    var kBtn = $("onera_keys_btn");
    if (kBtn) kBtn.classList.toggle("is-active", state.showKeys);
    $("workspace_toolbar").classList.toggle("onera-fs-show-keys", state.showKeys);
    requestAnimationFrame(function () {
      if (hypnoView) hypnoView.resizeAndDraw();
      if (waveView) waveView.resizeAndDraw();
    });
  }

  function toggleFS() {
    state.isFS = !state.isFS;
    var docEl = document.documentElement;
    if (state.isFS) {
      syncFSUI();
      var reqFn = docEl && (docEl.requestFullscreen || docEl.webkitRequestFullscreen);
      if (!getFsEl() && reqFn) {
        try {
          var p = reqFn.call(docEl);
          if (p && typeof p.catch === "function") p.catch(function () {});
        } catch (e) {}
      }
    } else {
      var exitFn = document.exitFullscreen || document.webkitExitFullscreen;
      if (getFsEl() && exitFn) {
        try {
          var pe = exitFn.call(document);
          if (pe && typeof pe.catch === "function") pe.catch(function () {});
        } catch (e) {}
      }
      syncFSUI();
    }
  }

  /* ---- Export Helpers ---- */
  function syncExportFormatFields() {
    if (!state.session) return;
    var fmt = $("export_format_select").value;
    var stem = state.session.stem || "scored";
    var bidsBase = stem.replace(/_(eeg|ieeg|meg|psg)$/i, "");
    var nameMap = {
      txt: [stem + ".txt", "📄 Output Filename (.txt):", false],
      bids_tsv: [bidsBase + "_events.tsv", "📄 Output Filename (*_events.tsv):", true],
      mne_csv: [stem + "_annotations.csv", "📄 Output Filename (*_annotations.csv):", true],
      rich_csv: [stem + "_hypnogram.csv", "📄 Output Filename (*_hypnogram.csv):", false],
      vmrk: [stem + ".vmrk", "📄 Output Filename (.vmrk):", true],
      markers: [stem + ".Markers", "📄 Output Filename (.Markers):", true]
    };
    var spec = nameMap[fmt] || nameMap.markers;
    $("export_filename_input").value = spec[0];
    $("export_filename_lbl").textContent = spec[1];
    $("include_events_wrap").classList.toggle("hidden", !spec[2]);
  }

  function getExportPayload() {
    var rad = document.querySelector('input[name="notation_radio"]:checked');
    return {
      fmt_key: $("export_format_select").value,
      output_dir: $("export_dir_input").value.trim() || ".",
      filename: $("export_filename_input").value.trim(),
      use_rk: Boolean(rad && rad.value === "RK"),
      include_events: $("include_events_chk").checked
    };
  }

  function showExportUnscored(nUns, fmtKey) {
    var el = $("export_unscored_callout");
    if (nUns > 0) {
      var tip = fmtKey === "txt"
        ? " Tip: Export as .Markers if you plan to resume mid-night scoring later so unscored epochs stay distinct from Wake (0)."
        : "";
      el.textContent = "⚠️ " + nUns + " epoch(s) are still unscored (UNS). They export as ? in text/marker formats and 0 in numeric codes." + tip;
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  }

  /* ---- Boot & Wiring ---- */
  function init() {
    applyTheme(currentTheme);

    hypnoView = window.OneraHypnogram.create({
      theme: currentTheme,
      onSelectEpochs: function (evt) {
        if (!state.session) return;
        var mx = Math.max(0, state.session.n_ep - 1);
        var validSet = state.session.is_manual_mode || state.showAll
          ? null
          : new Set(state.session.epochs.final_stage.reduce(function (acc, st, idx) {
              if (state.session.epochs.is_flagged[idx] || state.session.epochs.is_manual_edit[idx] || st === "ART") acc.push(idx);
              return acc;
            }, []));
        var idxs = (evt.epochs || []).filter(function (i) {
          return i >= 0 && i <= mx && (!validSet || validSet.has(i));
        });
        if (!idxs.length) return;
        var curSet = new Set(state.selectedEps);
        if (evt.type === "click" && idxs.length === 1) {
          var ep = idxs[0];
          if (evt.additive) {
            if (curSet.has(ep)) {
              curSet.delete(ep);
              state.selectedEps = Array.from(curSet).sort(function (a, b) { return a - b; });
              if (state.selectedEps.length && state.activeEp === ep) {
                state.activeEp = state.selectedEps[state.selectedEps.length - 1];
              }
            } else {
              curSet.add(ep);
              state.selectedEps = Array.from(curSet).sort(function (a, b) { return a - b; });
              state.activeEp = ep;
            }
          } else {
            state.selectedEps = [ep];
            state.activeEp = ep;
          }
        } else {
          idxs.forEach(function (i) { curSet.add(i); });
          state.selectedEps = Array.from(curSet).sort(function (a, b) { return a - b; });
          state.activeEp = idxs[0];
        }
        waveView.jumpToEpoch(state.activeEp);
        hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, false);
        updateSummaryAndSidebar();
        updateToolbarState();
      }
    });

    waveView = window.OneraWaveforms.create({
      theme: currentTheme,
      onPreviewEpoch: function (epIdx) {
        if (!state.session) return;
        state.activeEp = epIdx;
        state.selectedEps = [epIdx];
        hypnoView.previewEpoch(epIdx);
        updateSummaryAndSidebar();
      },
      onPanCommit: function (epIdx) {
        if (!state.session) return;
        state.activeEp = epIdx;
        state.selectedEps = [epIdx];
        hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, true);
        updateSummaryAndSidebar();
        updateToolbarState();
      },
      onChannelsChanged: function () {
        updateAddChannelSelect();
        updateSummaryAndSidebar();
      },
      onAddChannel: function (chName) {
        apiJson("/api/channels/ensure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ channel: chName })
        }).then(function (chMeta) {
          waveView.addChannel(chMeta);
          updateAddChannelSelect();
          updateSummaryAndSidebar();
        }).catch(function (err) {
          alert("Could not load channel " + chName + ": " + err.message);
        });
      }
    });

    recCombo = createComboBox($("rec_combobox"), selectRecording);
    refMs = createMultiSelect($("ref_channels_ms"));
    visualMs = createMultiSelect($("visual_channels_ms"));
    customEegMs = createMultiSelect($("custom_eeg_ms"));
    manualMs = createMultiSelect($("manual_channels_ms"));

    $("theme_toggle_btn").onclick = cycleTheme;
    $("toggle_about_btn").onclick = function () {
      $("header_features").classList.toggle("hidden");
    };
    $("toggle_config_btn").onclick = function () {
      var body = $("config_body");
      var hide = !body.classList.contains("hidden");
      body.classList.toggle("hidden", hide);
      $("toggle_config_btn").textContent = hide ? "▸ Expand" : "▾ Collapse";
    };

    $("scan_dirs_btn").onclick = scanDirectories;
    $("add_ignore_pair_btn").onclick = function () {
      state.ignorePairs.push(["", ""]);
      renderIgnorePairs();
    };
    $("remove_ignore_pair_btn").onclick = function () {
      if (state.ignorePairs.length > 1) {
        state.ignorePairs.pop();
        renderIgnorePairs();
      }
    };
    $("confidence_slider").oninput = function () {
      $("confidence_val").textContent = $("confidence_slider").value + "%";
    };
    $("resume_file_input").onchange = function () {
      var f = $("resume_file_input").files && $("resume_file_input").files[0];
      var nameEl = $("resume_file_name");
      nameEl.textContent = f ? f.name : "No file selected";
      nameEl.title = f ? f.name : "";
    };

    $("cons_s1_file_input").onchange = function () {
      var f = $("cons_s1_file_input").files && $("cons_s1_file_input").files[0];
      var el = $("cons_s1_file_name");
      el.textContent = f ? f.name : "No file selected";
      el.title = f ? f.name : "";
      syncConsensusLabelsFromFiles();
    };
    $("cons_s2_file_input").onchange = function () {
      var f = $("cons_s2_file_input").files && $("cons_s2_file_input").files[0];
      var el = $("cons_s2_file_name");
      el.textContent = f ? f.name : "No file selected";
      el.title = f ? f.name : "";
      syncConsensusLabelsFromFiles();
    };
    ["cons_s1_label_input", "cons_s2_label_input"].forEach(function (id) {
      $(id).addEventListener("input", function () {
        $(id).dataset.userEdited = $(id).value.trim() ? "1" : "";
      });
    });
    $("cons_with_yasa_chk").onchange = function () {
      var on = $("cons_with_yasa_chk").checked;
      $("cons_auto_resolve_chk").disabled = !on;
      if (!on) $("cons_auto_resolve_chk").checked = false;
    };

    $("run_auto_btn").onclick = function () { startScoring("auto"); };
    $("start_manual_btn").onclick = function () { startScoring("manual"); };
    $("resume_btn").onclick = function () { startScoring("resume"); };
    $("run_consensus_btn").onclick = function () { startScoring("consensus"); };
    $("cons_accept_s1_btn").onclick = function () { acceptConsensusScorer(1); };
    $("cons_accept_s2_btn").onclick = function () { acceptConsensusScorer(2); };

    $("btn_prev_flag").onclick = function () {
      if (!state.session) return;
      var fl = state.session.jump_indices || [];
      if (!fl.length) return;
      var cur = state.activeEp;
      var tgt = fl[fl.length - 1];
      for (var i = fl.length - 1; i >= 0; i--) { if (fl[i] < cur) { tgt = fl[i]; break; } }
      jumpSingle(tgt);
    };
    $("btn_next_flag").onclick = function () {
      if (!state.session) return;
      var fl = state.session.jump_indices || [];
      if (!fl.length) return;
      var cur = state.activeEp;
      var tgt = fl[0];
      for (var i = 0; i < fl.length; i++) { if (fl[i] > cur) { tgt = fl[i]; break; } }
      jumpSingle(tgt);
    };
    $("btn_prev_ep").onclick = function () { jumpSingle(state.activeEp - 1); };
    $("btn_next_ep").onclick = function () { jumpSingle(state.activeEp + 1); };
    $("btn_unselect").onclick = unselectEpochs;

    $("btn_stage_w").onclick = function () { applyStage("W"); };
    $("btn_stage_n1").onclick = function () { applyStage("N1"); };
    $("btn_stage_n2").onclick = function () { applyStage("N2"); };
    $("btn_stage_n3").onclick = function () { applyStage("N3"); };
    $("btn_stage_s4").onclick = function () { applyStage("S4"); };
    $("btn_stage_rem").onclick = function () { applyStage("REM"); };
    $("btn_stage_art").onclick = function () { applyStage("ART"); };
    $("btn_toggle_ma").onclick = toggleMa;
    $("btn_revert").onclick = revertLast;
    $("btn_clear_all").onclick = clearAllEdits;

    $("chk_auto_advance").onchange = function () {
      state.autoAdvance = $("chk_auto_advance").checked;
    };
    $("chk_show_all").onchange = function () {
      if (!state.session) return;
      state.showAll = $("chk_show_all").checked;
      if (!state.showAll) {
        var ep = state.session.epochs;
        state.selectedEps = state.selectedEps.filter(function (i) {
          return ep.is_flagged[i] || ep.final_stage[i] === "ART";
        });
      }
      hypnoView.updateState(computeHypnoDerived(), state.activeEp, state.selectedEps, false);
      updateSummaryAndSidebar();
      updateToolbarState();
    };

    $("onera_fs_btn").onclick = toggleFS;
    $("onera_keys_btn").onclick = function () {
      state.showKeys = !state.showKeys;
      syncFSUI();
    };

    var onFsChange = function () {
      if (!getFsEl() && state.isFS) {
        state.isFS = false;
        syncFSUI();
      }
    };
    document.addEventListener("fullscreenchange", onFsChange);
    document.addEventListener("webkitfullscreenchange", onFsChange);

    window.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && state.isFS && !getFsEl()) {
        state.isFS = false;
        syncFSUI();
        return;
      }
      if (!state.session) return;
      var tg = (e.target && e.target.tagName || "").toUpperCase();
      if (tg === "TEXTAREA" || tg === "SELECT") return;
      if (tg === "INPUT") {
        var tp = (e.target.type || "text").toLowerCase();
        if (tp !== "checkbox" && tp !== "radio" && tp !== "button" && tp !== "range" && tp !== "submit" && tp !== "reset") return;
      }
      if (e.target && e.target.isContentEditable) return;
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        revertLast();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      var c = e.code || "", k = e.key || "";
      var act = null;
      if (c === "Digit0" || c === "Numpad0" || k === "0") act = "W";
      else if (c === "Digit1" || c === "Numpad1" || k === "1") act = "N1";
      else if (c === "Digit2" || c === "Numpad2" || k === "2") act = "N2";
      else if (c === "Digit3" || c === "Numpad3" || k === "3") act = "N3";
      else if (c === "Digit5" || c === "Numpad5" || k === "5") act = "REM";
      else if (c === "KeyF" || k === "f" || k === "F") act = "ART";
      else if (c === "KeyG" || k === "g" || k === "G") act = "MA";
      else if (c === "KeyX" || k === "x" || k === "X") act = "UNSELECT";
      else if (c === "ArrowLeft" || k === "ArrowLeft") act = "PREV";
      else if (c === "ArrowRight" || k === "ArrowRight") act = "NEXT";
      if (!act) return;
      if (e.repeat && act !== "PREV" && act !== "NEXT") return;

      e.preventDefault();
      if (act === "PREV") jumpSingle(state.activeEp - 1);
      else if (act === "NEXT") jumpSingle(state.activeEp + 1);
      else if (act === "UNSELECT") unselectEpochs();
      else if (act === "MA") toggleMa();
      else applyStage(act);
    }, true);

    $("export_format_select").onchange = syncExportFormatFields;
    $("export_preview_btn").onclick = function () {
      var payload = getExportPayload();
      apiJson("/api/export/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(function (res) {
        showExportUnscored(res.n_unscored, res.fmt_key);
        if (typeof res.n_unresolved_conflicts === "number") {
          updateExportConflictCallout(res.n_unresolved_conflicts, res.s1_label);
        }
        var st = $("export_status_callout");
        st.className = "onera-callout is-info";
        st.textContent = "Target: " + res.target_path + " (" + res.n_rows + " rows)";
        st.classList.remove("hidden");
        $("export_download_btn").classList.remove("hidden");
        $("export_preview_pre").textContent = res.preview;
        $("export_preview_pre").classList.remove("hidden");
      }).catch(function (err) {
        var st = $("export_status_callout");
        st.className = "onera-callout is-danger";
        st.textContent = "❌ Preview failed: " + err.message;
        st.classList.remove("hidden");
      });
    };

    $("export_save_btn").onclick = function () {
      var payload = getExportPayload();
      apiJson("/api/export/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(function (res) {
        showExportUnscored(res.n_unscored, payload.fmt_key);
        if (typeof res.n_unresolved_conflicts === "number") {
          updateExportConflictCallout(res.n_unresolved_conflicts, res.s1_label);
        }
        var st = $("export_status_callout");
        st.className = "onera-callout is-success";
        var sideNote = res.sidecar_relpath ? (" (session state saved to " + res.sidecar_relpath + ")") : "";
        st.textContent = "✅ Saved " + res.n_rows + " rows to: " + res.target_path + sideNote;
        st.classList.remove("hidden");
        $("export_preview_pre").textContent = res.preview;
        $("export_preview_pre").classList.remove("hidden");
      }).catch(function (err) {
        var st = $("export_status_callout");
        st.className = "onera-callout is-danger";
        st.textContent = "❌ Failed to save: " + err.message;
        st.classList.remove("hidden");
      });
    };

    $("export_download_btn").onclick = function () {
      var payload = getExportPayload();
      fetch("/api/export/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(function (res) {
        if (!res.ok) throw new Error("Download failed (" + res.status + ")");
        var cd = res.headers.get("Content-Disposition") || "";
        var m = cd.match(/filename="([^"]+)"/);
        var fname = m ? m[1] : (payload.filename || "hypnogram.txt");
        return res.blob().then(function (blob) { return { blob: blob, fname: fname }; });
      }).then(function (obj) {
        var url = URL.createObjectURL(obj.blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = obj.fname;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      }).catch(function (err) {
        alert(err.message);
      });
    };

    // Hydrate initial server state on load/refresh
    apiJson("/api/state").then(function (initSt) {
      if (initSt.data_dir) $("data_dir_input").value = initSt.data_dir;
      if (initSt.output_dir) $("output_dir_input").value = initSt.output_dir;
      if (initSt.scanned && initSt.recordings && initSt.recordings.length) {
        $("recording_selector_row").classList.remove("hidden");
        $("scan_callout").className = "onera-callout is-info";
        $("scan_callout").innerHTML = "Found <strong>" + initSt.recordings.length + "</strong> recording(s).";
        var selKey = initSt.selected_recording ? initSt.selected_recording.recording_key : initSt.recordings[0];
        recCombo.setOptions(initSt.recordings, selKey);
        if (initSt.selected_recording) renderRecordingMeta(initSt.selected_recording);
      }
      if (initSt.has_session) {
        apiJson("/api/session").then(applySession).catch(function () {});
      }
    }).catch(function () {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

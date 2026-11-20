/* High-FPS Offscreen Epoch Tile Caching + Async Binary Streaming Waveform Inspector */
(function () {
  "use strict";

  var EPOCH_LRU_MAX = 1800;
  var TILE_LRU_MAX = 28;

  function createLRU(maxSize, onEvict) {
    var map = new Map();
    return {
      get: function (key) {
        if (!map.has(key)) return undefined;
        var val = map.get(key);
        map.delete(key);
        map.set(key, val);
        return val;
      },
      has: function (key) {
        return map.has(key);
      },
      set: function (key, val) {
        if (map.has(key)) map.delete(key);
        map.set(key, val);
        while (map.size > maxSize) {
          var oldestKey = map.keys().next().value;
          var oldestVal = map.get(oldestKey);
          map.delete(oldestKey);
          if (onEvict) onEvict(oldestKey, oldestVal);
        }
      },
      clear: function () {
        if (onEvict) {
          map.forEach(function (val, key) { onEvict(key, val); });
        }
        map.clear();
      }
    };
  }

  function createWaveformsController(opts) {
    var cv = document.getElementById("onera_wave_canvas");
    var wrapEl = document.getElementById("onera_wave_wrap");
    var b1 = document.getElementById("onera_cols_1");
    var b2 = document.getElementById("onera_cols_2");
    var w30 = document.getElementById("onera_win_30");
    var w60 = document.getElementById("onera_win_60");
    var ampSel = document.getElementById("onera_amp_scale");
    var addSel = document.getElementById("onera_add_ch_select");
    var titleEl = document.getElementById("onera_wave_title");

    var ctx = cv ? cv.getContext("2d") : null;
    function normTheme(t) {
      if (t === "night" || t === "dark" || t === "light") return t;
      return t ? "dark" : "light";
    }
    var themeMode = normTheme(opts && (opts.theme !== undefined ? opts.theme : opts.isDark));
    var cachedWaveTheme = null;

    var epochCache = createLRU(EPOCH_LRU_MAX);
    var tileCache = createLRU(TILE_LRU_MAX, function (_k, tCv) {
      if (tCv) { tCv.width = 0; tCv.height = 0; }
    });
    var lastTileConfigKey = "";

    var sessionGen = 0;
    var inFlightEpochs = new Set();
    var activePrefetchCount = 0;
    var lastPanDir = 0;

    var sessionMeta = null;
    var eventsByEp = {};
    var allEvents = [];
    var hasAnyEvents = false;
    var viewStates = {};

    var st = {
      chs: [],
      cols: 1,
      winSec: 30,
      ampScale: 0,
      viewStartSec: 0,
      hlEp: 0,
      hoverRemove: -1,
      hoverGrip: -1,
      dragMode: null,
      dragChIdx: -1,
      dropSlot: -1,
      panStartClientX: 0,
      panStartViewSec: 0,
      panPlotW: 600,
      panLastEp: 0,
      removeIdx: -1
    };

    var snapAnimId = null, drawRafId = null, wheelSnapTimer = null, trackBoxes = [];

    function stopSnapAnim() {
      if (wheelSnapTimer !== null) {
        clearTimeout(wheelSnapTimer);
        wheelSnapTimer = null;
      }
      if (snapAnimId !== null) {
        cancelAnimationFrame(snapAnimId);
        snapAnimId = null;
        return true;
      }
      return false;
    }

    function snapHalf(v) { return Math.floor(v) + 0.5; }
    function pad2(n) { return n < 10 ? "0" + n : String(n); }
    function fmtClk(totalSec) {
      var s = ((Math.round(totalSec) % 86400) + 86400) % 86400;
      return pad2(Math.floor(s / 3600)) + ":" + pad2(Math.floor((s % 3600) / 60)) + ":" + pad2(s % 60);
    }
    function fmtAxisSec(totalSec) {
      var s = Math.max(0, Math.round(totalSec));
      var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sc = s % 60;
      return (sessionMeta && sessionMeta.totalSec >= 3600) ? (h + ":" + pad2(m) + ":" + pad2(sc)) : (pad2(m) + ":" + pad2(sc));
    }
    function fmtScaleTick(yl, ctype) {
      if (ctype !== "eeg" && ctype !== "eog" && ctype !== "emg" && ctype !== "ecg") {
        var v = yl * 0.65;
        if (v >= 1000) return (Math.round(v / 100) / 10) + "k";
        if (v >= 100) return String(Math.round(v));
        if (v >= 10) return String(Math.round(v * 10) / 10);
        if (v >= 1) return v.toFixed(1);
        return v.toPrecision(2);
      }
      var ticks = [10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000];
      var b = Math.round(yl * 0.5);
      for (var ti = 0; ti < ticks.length; ti++) { if (ticks[ti] <= yl * 0.85) b = ticks[ti]; }
      b = b || Math.round(yl * 0.5) || 10;
      if (b >= 1000) return (b % 1000 === 0 ? (b / 1000) + "k" : (b / 1000).toFixed(1) + "k");
      return String(b);
    }

    function getWaveTheme() {
      var mode = themeMode;
      if (cachedWaveTheme && cachedWaveTheme._mode === mode) return cachedWaveTheme;
      if (mode === "dark") {
        cachedWaveTheme = {
          _mode: "dark",
          bg: "#000000", text: "#a6b3c4", label: "#c8d2df", axis: "#5a687c",
          border: "#263347", dragBorder: "#6fa3d4",
          zero: "rgba(122,136,156,0.22)", delta: "rgba(122,136,156,0.34)",
          grid5s: "rgba(122,136,156,0.14)", epBound: "rgba(166,179,196,0.30)",
          hlBg: "rgba(78,123,186,0.11)", dragBg: "rgba(78,123,186,0.16)",
          epHi: "#7aa7d6", epMuted: "rgba(122,136,156,0.78)",
          rmBg: "rgba(122,136,156,0.14)", rmBgHot: "rgba(204,114,124,0.22)",
          rmStroke: "#7a889c", rmStrokeHot: "#cc727c",
          grip: "#5a687c", gripHot: "#c8d2df", dropBar: "#6fa3d4",
          evLights: "rgba(101,173,134,0.88)", evLightsBg: "rgba(20,42,32,0.85)",
          evOther: "rgba(158,133,201,0.90)", evOtherBg: "rgba(32,25,48,0.86)",
          traces: { eeg: "#79ace0", eog: "#62bfb0", emg: "#d6858f", ecg: "#a692d9", misc: "#8e9eb3" }
        };
      } else if (mode === "night") {
        cachedWaveTheme = {
          _mode: "night",
          bg: "#161f2e", text: "#cbd5e1", label: "#e2e8f0", axis: "#64748b",
          border: "#475569", dragBorder: "#60a5fa",
          zero: "rgba(148,163,184,0.28)", delta: "rgba(148,163,184,0.42)",
          grid5s: "rgba(148,163,184,0.20)", epBound: "rgba(203,213,225,0.40)",
          hlBg: "rgba(83,134,207,0.12)", dragBg: "rgba(83,134,207,0.16)",
          epHi: "#7eb2f0", epMuted: "rgba(148,163,184,0.85)",
          rmBg: "rgba(148,163,184,0.16)", rmBgHot: "rgba(235,132,142,0.24)",
          rmStroke: "#94a3b8", rmStrokeHot: "#eb848e",
          grip: "#64748b", gripHot: "#e2e8f0", dropBar: "#60a5fa",
          evLights: "rgba(120,199,157,0.92)", evLightsBg: "rgba(28,56,44,0.78)",
          evOther: "rgba(182,158,235,0.94)", evOtherBg: "rgba(44,36,64,0.82)",
          traces: { eeg: "#8ec0fa", eog: "#73d4c5", emg: "#f299a2", ecg: "#bfa8f5", misc: "#aab8ca" }
        };
      } else {
        cachedWaveTheme = {
          _mode: "light",
          bg: "#ffffff", text: "#334155", label: "#1e293b", axis: "#475569",
          border: "#94a3b8", dragBorder: "#5b8fd9",
          zero: "rgba(148,163,184,0.45)", delta: "rgba(100,116,139,0.52)",
          grid5s: "rgba(148,163,184,0.34)", epBound: "rgba(51,65,85,0.42)",
          hlBg: "rgba(91,143,217,0.08)", dragBg: "rgba(91,143,217,0.10)",
          epHi: "#467ac4", epMuted: "rgba(100,116,139,0.85)",
          rmBg: "rgba(148,163,184,0.14)", rmBgHot: "rgba(222,107,118,0.18)",
          rmStroke: "#64748b", rmStrokeHot: "#de6b76",
          grip: "#94a3b8", gripHot: "#334155", dropBar: "#5b8fd9",
          evLights: "rgba(46,125,87,0.90)", evLightsBg: "rgba(223,242,232,0.90)",
          evOther: "rgba(109,84,166,0.92)", evOtherBg: "rgba(239,233,250,0.92)",
          traces: { eeg: "#2f63ab", eog: "#268074", emg: "#b54854", ecg: "#6c52a8", misc: "#475569" }
        };
      }
      return cachedWaveTheme;
    }

    function clampViewSec(vs) {
      var tot = sessionMeta ? sessionMeta.totalSec : 30;
      var maxV = Math.max(0, tot - st.winSec);
      return Math.max(0, Math.min(maxV, vs));
    }

    function alignedViewSec(epIdx, winSec) {
      var tot = sessionMeta ? sessionMeta.totalSec : 30;
      var maxV = Math.max(0, tot - winSec);
      var raw = epIdx * 30.0 - (winSec === 60 ? 15.0 : 0.0);
      return Math.max(0, Math.min(maxV, raw));
    }

    function computeCenterEp(vs) {
      var nEp = sessionMeta ? sessionMeta.nEp : 1;
      var tot = sessionMeta ? sessionMeta.totalSec : 30;
      var mx = Math.max(0, nEp - 1);
      var maxTot = Math.max(0, tot - st.winSec);
      if (st.winSec === 60 && mx >= 1) {
        if (vs <= 7.5) return 0;
        if (vs >= maxTot - 7.5) return mx;
      }
      return Math.max(0, Math.min(mx, Math.floor((vs + 0.5 * st.winSec) / 30.0)));
    }

    function saveSessionView() {
      if (!sessionMeta) return;
      viewStates[sessionMeta.sessionKey] = {
        chs: st.chs.slice(),
        cols: st.cols,
        winSec: st.winSec,
        ampScale: st.ampScale,
        viewStartSec: st.viewStartSec,
        hlEp: st.hlEp
      };
    }

    function syncHeaderUI() {
      if (!sessionMeta) return;
      if (titleEl) {
        var ep = Math.max(0, Math.min((sessionMeta.nEp || 1) - 1, st.hlEp));
        var c0 = (sessionMeta.ep0ClockSec || 0) + ep * 30;
        var nextTitle = st.winSec + "s Pre-Filtered Waveforms \u2014 Ep #" + (ep + 1) + " (" + fmtClk(c0) + "\u2013" + fmtClk(c0 + 30) + ")";
        if (titleEl.textContent !== nextTitle) titleEl.textContent = nextTitle;
      }
      if (b1) b1.classList.toggle("is-active", st.cols === 1);
      if (b2) b2.classList.toggle("is-active", st.cols === 2);
      if (w30) w30.classList.toggle("is-active", st.winSec === 30);
      if (w60) w60.classList.toggle("is-active", st.winSec === 60);
      if (ampSel) {
        var ampVal = String(st.ampScale || 0);
        if (ampSel.value !== ampVal) ampSel.value = ampVal;
      }
    }

    function epochCacheKey(chName, epIdx) {
      return chName + "\u0000" + epIdx;
    }

    function isEpochComplete(epIdx, chList) {
      for (var i = 0; i < chList.length; i++) {
        if (!epochCache.has(epochCacheKey(chList[i].name, epIdx))) return false;
      }
      return true;
    }

    function decodeBinaryFrame(buf) {
      if (!buf || buf.byteLength < 8) return;
      var view = new DataView(buf);
      var hdrLen = view.getUint32(0, true);
      if (4 + hdrLen > buf.byteLength) return;
      var hdrText = new TextDecoder("utf-8").decode(new Uint8Array(buf, 4, hdrLen));
      var hdr = JSON.parse(hdrText);
      var payloadBase = 4 + hdrLen;
      var nEps = hdr.endEp - hdr.startEp;
      var chs = hdr.channels || [];
      for (var ci = 0; ci < chs.length; ci++) {
        var cm = chs[ci];
        var pts = cm.ptsPerEp;
        for (var bi = 0; bi < nEps; bi++) {
          var ep = hdr.startEp + bi;
          var byteStart = payloadBase + cm.byteOffset + bi * pts * 2;
          if (byteStart + pts * 2 > buf.byteLength) continue;
          var copied = new Int16Array(buf, byteStart, pts).slice();
          epochCache.set(epochCacheKey(cm.name, ep), {
            i16: copied,
            isMinMax: Boolean(cm.isMinMax),
            encodeMax: Number(cm.encodeMax),
            ylim: Number(cm.ylim),
            type: cm.type
          });
        }
      }
    }

    function fetchEpochBatch(startEp, endEp, chList, isPrefetch) {
      if (!sessionMeta || endEp <= startEp || !chList.length) return;
      var myGen = sessionGen;
      var keys = [];
      for (var ep = startEp; ep < endEp; ep++) {
        for (var ci = 0; ci < chList.length; ci++) {
          var fk = chList[ci].name + "\u0000" + ep;
          keys.push(fk);
          inFlightEpochs.add(fk);
        }
      }
      if (isPrefetch) activePrefetchCount++;
      var qs = new URLSearchParams();
      qs.set("start_ep", String(startEp));
      qs.set("end_ep", String(endEp));
      for (var j = 0; j < chList.length; j++) {
        qs.append("channels", chList[j].name);
      }
      fetch("/api/waveforms?" + qs.toString())
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          return res.arrayBuffer();
        })
        .then(function (buf) {
          if (myGen !== sessionGen) return;
          decodeBinaryFrame(buf);
          scheduleDraw();
        })
        .catch(function () {
          // Non-fatal network/abort error; next view tick retries cleanly
        })
        .finally(function () {
          for (var k = 0; k < keys.length; k++) inFlightEpochs.delete(keys[k]);
          if (isPrefetch) {
            activePrefetchCount = Math.max(0, activePrefetchCount - 1);
            if (myGen === sessionGen) requestWindowEpochs(st.viewStartSec, lastPanDir);
          }
        });
    }

    function findMissingRuns(ep0, ep1, chList) {
      var runs = [];
      var runStart = -1;
      for (var ep = ep0; ep <= ep1; ep++) {
        var need = false;
        for (var ci = 0; ci < chList.length; ci++) {
          var ck = epochCacheKey(chList[ci].name, ep);
          if (!epochCache.has(ck) && !inFlightEpochs.has(ck)) {
            need = true;
            break;
          }
        }
        if (need) {
          if (runStart < 0) runStart = ep;
          if (ep - runStart + 1 >= 16) {
            runs.push([runStart, ep + 1]);
            runStart = -1;
          }
        } else if (runStart >= 0) {
          runs.push([runStart, ep]);
          runStart = -1;
        }
      }
      if (runStart >= 0) runs.push([runStart, ep1 + 1]);
      return runs;
    }

    function requestWindowEpochs(vs, panDir) {
      if (!sessionMeta || !st.chs.length) return;
      var nEp = sessionMeta.nEp;
      var k0 = Math.max(0, Math.floor(vs / 30.0));
      var k1 = Math.min(nEp - 1, Math.floor(Math.max(vs, vs + st.winSec - 1e-4) / 30.0));

      var visRuns = findMissingRuns(k0, k1, st.chs);
      for (var ri = 0; ri < visRuns.length; ri++) {
        fetchEpochBatch(visRuns[ri][0], visRuns[ri][1], st.chs, false);
      }

      if (activePrefetchCount < 1) {
        var back = panDir < 0 ? 8 : 2;
        var fwd = panDir > 0 ? 8 : (panDir < 0 ? 2 : 4);
        var pf0 = Math.max(0, k0 - back);
        var pf1 = Math.min(nEp - 1, k1 + fwd);
        var pfRuns = findMissingRuns(pf0, pf1, st.chs);
        if (pfRuns.length) {
          fetchEpochBatch(pfRuns[0][0], pfRuns[0][1], st.chs, true);
        }
      }
    }

    function ensureTileConfig(layout) {
      var chOrder = st.chs.map(function (c) { return c.name; }).join("\u0001");
      var key = [
        sessionGen,
        st.winSec,
        layout.cols,
        st.ampScale,
        themeMode,
        layout.dpr.toFixed(3),
        layout.tileW_dev,
        layout.H_dev,
        hasAnyEvents ? 1 : 0,
        chOrder
      ].join("|");
      if (key !== lastTileConfigKey) {
        tileCache.clear();
        lastTileConfigKey = key;
      }
    }

    function getOrBakeTile(epIdx, colIdx, layout) {
      var tileKey = epIdx + "|" + colIdx;
      var existing = tileCache.get(tileKey);
      if (existing) return existing;

      var nCh = st.chs.length;
      var colChIndices = [];
      for (var i = 0; i < nCh; i++) {
        var cIdx = layout.cols === 2 ? (i % 2) : 0;
        if (cIdx === colIdx) colChIndices.push(i);
      }
      if (!colChIndices.length) return null;

      // Only bake once every channel in this column has epoch data ready
      var entries = [];
      for (var ci = 0; ci < colChIndices.length; ci++) {
        var chObj = st.chs[colChIndices[ci]];
        var ent = epochCache.get(epochCacheKey(chObj.name, epIdx));
        if (!ent) return null;
        entries.push({ idx: colChIndices[ci], ch: chObj, data: ent });
      }

      var tCanvas = document.createElement("canvas");
      tCanvas.width = layout.tileW_dev;
      tCanvas.height = layout.H_dev;
      var tctx = tCanvas.getContext("2d");
      if (!tctx) return null;
      tctx.setTransform(layout.dpr, 0, 0, layout.dpr, 0, 0);

      var tileW = layout.tileW_dev / layout.dpr;
      var wt = getWaveTheme();

      for (var ei = 0; ei < entries.length; ei++) {
        var item = entries[ei];
        var chIdx = item.idx;
        var c = item.ch;
        var entData = item.data;
        var rowIdx = layout.cols === 2 ? Math.floor(chIdx / 2) : chIdx;
        var yTop = layout.padT + rowIdx * (layout.chH + layout.gapY);
        var yMid = yTop + layout.chH / 2;
        var isAmpScaled = (c.type === "eeg" || c.type === "eog");
        var yl = (isAmpScaled && st.ampScale > 0) ? st.ampScale : (entData.ylim || c.ylim || 100);
        var encMax = Number(entData.encodeMax || c.encodeMax || yl);
        var yScale = ((layout.chH / 2) / Math.max(1e-12, yl)) * (encMax / 32767.0);
        var cCol = wt.traces[c.type] || c.color || wt.traces.misc;
        var arr = entData.i16;

        tctx.save();
        tctx.beginPath();
        tctx.rect(0, yTop + 0.5, tileW, Math.max(1, layout.chH - 1));
        tctx.clip();

        if (arr && arr.length > 1) {
          tctx.strokeStyle = cCol;
          tctx.lineJoin = "bevel";
          tctx.lineCap = "butt";
          tctx.beginPath();
          if (entData.isMinMax) {
            tctx.lineWidth = 0.65;
            var nPairs = Math.floor(arr.length / 2);
            var nCols = Math.max(1, layout.tileW_dev);
            var pPerCol = nPairs / nCols;
            if (pPerCol > 1.0) {
              var startedM = false;
              for (var cm = 0; cm < nCols; cm++) {
                var bA = Math.floor(cm * pPerCol);
                var bB = (cm === nCols - 1) ? nPairs : Math.floor((cm + 1) * pPerCol);
                if (bB <= bA) continue;
                var mnM = arr[bA * 2], mxM = arr[bA * 2 + 1];
                for (var b = bA + 1; b < bB; b++) {
                  var loM = arr[b * 2], hiM = arr[b * 2 + 1];
                  if (loM < mnM) mnM = loM;
                  if (hiM > mxM) mxM = hiM;
                }
                var pxM = (cm + 0.5) / layout.dpr;
                var y0M = yMid - mnM * yScale, y1M = yMid - mxM * yScale;
                if (!startedM) { tctx.moveTo(pxM, y0M); startedM = true; }
                else tctx.lineTo(pxM, y0M);
                if (mxM !== mnM) tctx.lineTo(pxM, y1M);
              }
              if (startedM) tctx.stroke();
            } else {
              var denomP = Math.max(1, nPairs - 1);
              for (var b2 = 0; b2 < nPairs; b2++) {
                var px = (b2 / denomP) * tileW;
                var y0 = yMid - arr[b2 * 2] * yScale, y1 = yMid - arr[b2 * 2 + 1] * yScale;
                if (b2 === 0) tctx.moveTo(px, y0);
                else tctx.lineTo(px, y0);
                if (y1 !== y0) tctx.lineTo(px, y1);
              }
              tctx.stroke();
            }
          } else {
            tctx.lineWidth = (c.type === "emg") ? 0.65 : 0.75;
            var ns = arr.length, denomS = Math.max(1, ns - 1);
            tctx.moveTo(0, yMid - arr[0] * yScale);
            for (var k = 1; k < ns; k++) {
              tctx.lineTo((k / denomS) * tileW, yMid - arr[k] * yScale);
            }
            tctx.stroke();
          }
        }
        tctx.restore();
      }

      tileCache.set(tileKey, tCanvas);
      return tCanvas;
    }

    function scheduleDraw() {
      if (drawRafId !== null || !sessionMeta) return;
      drawRafId = requestAnimationFrame(function () {
        drawRafId = null;
        drawCanvas();
      });
    }

    function drawCanvas() {
      if (!cv || !ctx || !sessionMeta) return false;
      var curChs = st.chs, nCh = curChs ? curChs.length : 0;
      syncHeaderUI();
      var wt = getWaveTheme();
      var cols = (nCh > 1 && st.cols === 2) ? 2 : 1;
      var nRows = cols === 2 ? Math.ceil(nCh / 2) : Math.max(1, nCh);
      var isFS = Boolean(document.documentElement && document.documentElement.classList.contains("onera-fs-active"));

      var trackH = cols === 2 ? (nRows <= 4 ? 92 : Math.max(68, Math.floor(420 / nRows))) : (nRows <= 6 ? 70 : Math.max(56, Math.floor(420 / nRows)));
      var padT = hasAnyEvents ? 26 : 13, padB = 18, gapY = isFS ? 2 : 3;
      var targetH = nCh > 0 ? Math.max(220, nRows * trackH + Math.max(0, nRows - 1) * gapY + padT + padB) : 220;
      var availFS = (isFS && wrapEl) ? Math.floor(wrapEl.clientHeight) : 0;
      if (isFS && availFS > 60) {
        if (cv.style.height !== "") cv.style.height = "";
      } else {
        var tgtStr = targetH + "px";
        if (cv.style.height !== tgtStr) cv.style.height = tgtStr;
      }

      var W = cv.clientWidth || (wrapEl && wrapEl.clientWidth) || (cv.parentElement && cv.parentElement.clientWidth) || 0;
      var H = (isFS && availFS > 60) ? availFS : (cv.clientHeight || targetH);
      if (W <= 0 || H <= 0) return false;

      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var tw = Math.round(W * dpr), th = Math.round(H * dpr);
      if (cv.width !== tw || cv.height !== th) { cv.width = tw; cv.height = th; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = wt.bg;
      ctx.fillRect(0, 0, W, H);
      trackBoxes = [];

      if (!nCh) {
        ctx.fillStyle = wt.text; ctx.font = "600 13px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("Signal data unavailable.", W / 2, H / 2);
        return true;
      }

      var vs = st.viewStartSec, ws = st.winSec, ve = vs + ws;
      var padL = 58, padR = 6, colGap = 8;
      var colW = cols === 2 ? Math.max(150, (W - colGap) / 2) : Math.max(150, W - 2);
      var colOrigins = cols === 2 ? [0, colW + colGap] : [0];
      var plotW = Math.max(70, colW - padL - padR);
      var chH = Math.max(isFS ? 18 : 36, (H - padT - padB - gapY * (nRows - 1)) / nRows);
      var tileW_dev = Math.max(1, Math.round((30.0 / ws) * plotW * dpr));

      var layout = {
        cols: cols,
        nRows: nRows,
        padT: padT,
        gapY: gapY,
        chH: chH,
        dpr: dpr,
        tileW_dev: tileW_dev,
        H_dev: th
      };
      ensureTileConfig(layout);

      // Ensure visible + prefetch epochs are requested
      requestWindowEpochs(vs, lastPanDir);

      var hlStart = st.hlEp * 30.0, hlEnd = hlStart + 30.0;
      var epLineBnds = [];
      var b0 = Math.ceil(vs / 30.0) * 30.0;
      for (var bs = b0; bs <= ve + 1e-6; bs += 30.0) {
        if (bs >= vs - 1e-6 && bs <= ve + 1e-6) epLineBnds.push(bs);
      }

      // Pass 1: Track backgrounds, active-epoch highlight tint, gridlines, zero & delta lines
      for (var i = 0; i < nCh; i++) {
        var colIdx = cols === 2 ? (i % 2) : 0, rowIdx = cols === 2 ? Math.floor(i / 2) : i;
        var xBase = colOrigins[colIdx], xPlot = xBase + padL;
        var c = curChs[i], yTop = padT + rowIdx * (chH + gapY), yMid = yTop + chH / 2, yBot = yTop + chH;
        var isAmpScaled = (c.type === "eeg" || c.type === "eog");
        var yl = (isAmpScaled && st.ampScale > 0) ? st.ampScale : (c.ylim || 100);
        var rmRect = { x: xBase + 1, y: yTop + 1, w: 12, h: 12 };
        var gripRect = { x: xBase + 1, y: yTop + 14, w: 12, h: Math.max(12, chH - 15) };
        var plotRect = { x: xPlot, y: yTop, w: plotW, h: chH };
        trackBoxes.push({ idx: i, colIdx: colIdx, rowIdx: rowIdx, xBase: xBase, yTop: yTop, yBot: yBot, colW: colW, chH: chH, rm: rmRect, grip: gripRect, plot: plotRect });

        ctx.save();
        if (st.dragMode === "reorder" && st.dragChIdx === i) {
          ctx.fillStyle = wt.dragBg;
          ctx.fillRect(xBase + 1, yTop, colW - 2, chH);
        }

        var ov0 = Math.max(vs, hlStart), ov1 = Math.min(ve, hlEnd);
        if (ov1 > ov0) {
          var hx0 = xPlot + ((ov0 - vs) / ws) * plotW, hx1 = xPlot + ((ov1 - vs) / ws) * plotW;
          ctx.fillStyle = wt.hlBg;
          ctx.fillRect(hx0, yTop, hx1 - hx0, chH);
        }

        // 5s vertical sub-grid
        ctx.strokeStyle = wt.grid5s; ctx.lineWidth = 0.8; ctx.setLineDash([2, 3]);
        var g0 = Math.ceil(vs / 5.0) * 5.0;
        for (var s = g0; s <= ve + 1e-6; s += 5.0) {
          if (s <= vs + 0.1 || s >= ve - 0.1) continue;
          if (Math.abs(s % 30.0) < 0.05 || Math.abs((s % 30.0) - 30.0) < 0.05) continue;
          var gx = snapHalf(xPlot + ((s - vs) / ws) * plotW);
          ctx.beginPath(); ctx.moveTo(gx, yTop); ctx.lineTo(gx, yBot); ctx.stroke();
        }
        ctx.setLineDash([]);

        // 30s epoch boundaries
        if (epLineBnds.length) {
          ctx.strokeStyle = wt.epBound; ctx.lineWidth = 1.2;
          for (var ebi = 0; ebi < epLineBnds.length; ebi++) {
            var ebx = snapHalf(xPlot + ((epLineBnds[ebi] - vs) / ws) * plotW);
            if (ebx <= xPlot + 0.8 || ebx >= xPlot + plotW - 0.8) continue;
            ctx.beginPath(); ctx.moveTo(ebx, yTop); ctx.lineTo(ebx, yBot); ctx.stroke();
          }
        }

        // Zero & ±37.5µV delta reference lines
        var yMidSnap = snapHalf(yMid);
        ctx.strokeStyle = wt.zero; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(xPlot, yMidSnap); ctx.lineTo(xPlot + plotW, yMidSnap); ctx.stroke();
        if (c.type === "eeg" && yl > 37.5) {
          var dy37 = (37.5 / yl) * (chH / 2);
          var yD1 = snapHalf(yMid - dy37), yD2 = snapHalf(yMid + dy37);
          ctx.strokeStyle = wt.delta; ctx.lineWidth = 0.85; ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo(xPlot, yD1); ctx.lineTo(xPlot + plotW, yD1);
          ctx.moveTo(xPlot, yD2); ctx.lineTo(xPlot + plotW, yD2); ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.restore();
      }

      // Pass 2: Blit 1–3 cached offscreen epoch tiles per column snapped to device pixels
      var k0 = Math.max(0, Math.floor(vs / 30.0));
      var k1 = Math.min((sessionMeta.nEp || 1) - 1, Math.floor(Math.max(vs, ve - 1e-4) / 30.0));
      for (var colB = 0; colB < cols; colB++) {
        var rowsInC = cols === 2 ? (colB === 0 ? Math.ceil(nCh / 2) : Math.floor(nCh / 2)) : nRows;
        if (rowsInC <= 0) continue;
        var cPlotX = colOrigins[colB] + padL;
        var yBotC = padT + rowsInC * chH + (rowsInC - 1) * gapY;

        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        var clipX0 = Math.round((cPlotX + 0.5) * dpr);
        var clipW = Math.max(1, Math.round((plotW - 1) * dpr));
        var clipY0 = Math.round(padT * dpr);
        var clipH = Math.max(1, Math.round((yBotC - padT) * dpr));
        ctx.beginPath();
        ctx.rect(clipX0, clipY0, clipW, clipH);
        ctx.clip();

        var baseTileXDev = Math.round((cPlotX + ((k0 * 30.0 - vs) / ws) * plotW) * dpr);
        for (var ek = k0; ek <= k1; ek++) {
          var tileCv = getOrBakeTile(ek, colB, layout);
          if (tileCv) {
            ctx.drawImage(tileCv, baseTileXDev + (ek - k0) * tileW_dev, 0);
          }
        }
        ctx.restore();
      }

      // Pass 3: Borders, remove buttons, grips, Y-axis ticks & channel labels
      for (var j = 0; j < nCh; j++) {
        var tb = trackBoxes[j];
        var chObj = curChs[j];
        var isAmp = (chObj.type === "eeg" || chObj.type === "eog");
        var yl2 = (isAmp && st.ampScale > 0) ? st.ampScale : (chObj.ylim || 100);
        var yMid2 = tb.yTop + tb.chH / 2;
        var bx0 = snapHalf(tb.plot.x), by0 = snapHalf(tb.yTop);
        var bwTrack = Math.max(1, Math.round(plotW)), bhTrack = Math.max(1, Math.round(tb.chH));

        ctx.save();
        ctx.strokeStyle = (st.dragMode === "reorder" && st.dragChIdx === j) ? wt.dragBorder : wt.border;
        ctx.lineWidth = (st.dragMode === "reorder" && st.dragChIdx === j) ? 1.8 : 1.0;
        ctx.strokeRect(bx0, by0, bwTrack, bhTrack);

        if (nCh > 1) {
          var isRmHot = (st.hoverRemove === j);
          ctx.fillStyle = isRmHot ? wt.rmBgHot : wt.rmBg;
          ctx.beginPath(); ctx.arc(tb.rm.x + 6, tb.rm.y + 6, 5.2, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = isRmHot ? wt.rmStrokeHot : wt.rmStroke; ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(tb.rm.x + 3.8, tb.rm.y + 3.8); ctx.lineTo(tb.rm.x + 8.2, tb.rm.y + 8.2);
          ctx.moveTo(tb.rm.x + 8.2, tb.rm.y + 3.8); ctx.lineTo(tb.rm.x + 3.8, tb.rm.y + 8.2);
          ctx.stroke();
        }

        var gHot = (st.hoverGrip === j || (st.dragMode === "reorder" && st.dragChIdx === j));
        ctx.fillStyle = gHot ? wt.gripHot : wt.grip;
        var gxCenter = tb.grip.x + 6, gyCenter = tb.grip.y + tb.grip.h / 2;
        for (var dgRow = -1; dgRow <= 1; dgRow++) {
          ctx.beginPath();
          ctx.arc(gxCenter - 1.9, gyCenter + dgRow * 3.8, 1.05, 0, Math.PI * 2);
          ctx.arc(gxCenter + 1.9, gyCenter + dgRow * 3.8, 1.05, 0, Math.PI * 2);
          ctx.fill();
        }

        var tkStr = fmtScaleTick(yl2, chObj.type);
        var tkNum = (tkStr.slice(-1) === "k" ? parseFloat(tkStr) * 1000 : parseFloat(tkStr)) || yl2 * 0.5;
        var dyTk = Math.min(tb.chH * 0.42, (tkNum / Math.max(1e-12, yl2)) * (tb.chH / 2));
        var yPts = tb.chH < 36
          ? [[yMid2 - dyTk, tkStr], [yMid2 + dyTk, "-" + tkStr]]
          : [[yMid2 - dyTk, tkStr], [yMid2, "0"], [yMid2 + dyTk, "-" + tkStr]];
        ctx.fillStyle = wt.text; ctx.font = "8.5px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.strokeStyle = wt.axis; ctx.lineWidth = 1.0;
        for (var ky = 0; ky < yPts.length; ky++) {
          var yp = snapHalf(yPts[ky][0]);
          ctx.beginPath(); ctx.moveTo(tb.plot.x - 3, yp); ctx.lineTo(tb.plot.x, yp); ctx.stroke();
          ctx.fillText(yPts[ky][1], tb.plot.x - 4, yp);
        }

        if (tb.chH >= 52) {
          ctx.save(); ctx.translate(tb.xBase + 21, yMid2); ctx.rotate(-Math.PI / 2);
          ctx.fillStyle = wt.label; ctx.font = "700 9.5px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(chObj.name, 0, -4.5);
          ctx.fillStyle = wt.axis; ctx.font = "7.5px sans-serif";
          ctx.fillText(chObj.unit || "(\u00b5V)", 0, 5.5);
          ctx.restore();
        } else if (tb.chH >= 36) {
          ctx.save(); ctx.translate(tb.xBase + 21, yMid2); ctx.rotate(-Math.PI / 2);
          ctx.fillStyle = wt.label; ctx.font = "700 9.5px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(chObj.name, 0, 0);
          ctx.restore();
        } else {
          ctx.fillStyle = wt.label; ctx.font = "700 8.5px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText(chObj.name, tb.xBase + 14, yMid2, 24);
        }
        ctx.restore();
      }

      // Reorder drop insertion bar
      if (st.dragMode === "reorder" && st.dropSlot >= 0) {
        if (cols === 1) {
          var insY = (st.dropSlot >= nCh)
            ? (padT + nCh * chH + Math.max(0, nCh - 1) * gapY + 2)
            : (padT + st.dropSlot * (chH + gapY) - gapY / 2);
          ctx.strokeStyle = wt.dropBar; ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.moveTo(padL, insY); ctx.lineTo(colOrigins[0] + colW - padR, insY); ctx.stroke();
          ctx.fillStyle = wt.dropBar; ctx.beginPath(); ctx.arc(padL, insY, 4, 0, Math.PI * 2); ctx.fill();
        } else if (st.dropSlot < trackBoxes.length) {
          var tbT = trackBoxes[st.dropSlot];
          ctx.strokeStyle = wt.dropBar; ctx.lineWidth = 2.5;
          ctx.strokeRect(tbT.plot.x - 1, tbT.yTop - 1, tbT.plot.w + 2, tbT.chH + 2);
        }
      }

      // Events, Epoch # boundary chips, and bottom time axis per column
      var visEvs = [];
      for (var vei = 0; vei < allEvents.length; vei++) {
        var evA = allEvents[vei];
        if (evA.absSec >= vs && evA.absSec <= ve) visEvs.push(evA);
      }

      for (var colI = 0; colI < cols; colI++) {
        var rowsInCol = cols === 2 ? (colI === 0 ? Math.ceil(nCh / 2) : Math.floor(nCh / 2)) : nRows;
        if (rowsInCol <= 0) continue;
        var yColBot = padT + rowsInCol * chH + (rowsInCol - 1) * gapY;
        var cPlot = colOrigins[colI] + padL;

        ctx.font = "700 9px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top";
        var epY = hasAnyEvents ? 14 : 1;
        var lastEpRx = -1e9;
        for (var ek2 = k0; ek2 <= k1; ek2++) {
          var exStart = Math.max(vs, ek2 * 30.0), exEnd = Math.min(ve, (ek2 + 1) * 30.0);
          var pxBadge = cPlot + ((exStart - vs) / ws) * plotW + 4;
          var pxEpEnd = cPlot + ((exEnd - vs) / ws) * plotW;
          var epStr = "Ep #" + (ek2 + 1);
          var twEp = ctx.measureText(epStr).width;
          if (pxBadge >= lastEpRx + 6 && pxBadge + twEp + 4 <= pxEpEnd && pxBadge + twEp + 4 < cPlot + plotW) {
            ctx.fillStyle = (ek2 === st.hlEp) ? wt.epHi : wt.epMuted;
            ctx.fillText(epStr, pxBadge, epY);
            lastEpRx = pxBadge + twEp;
          }
        }

        if (visEvs.length) {
          ctx.font = "700 9px sans-serif";
          var lastRx = -1e9;
          for (var ei2 = 0; ei2 < visEvs.length; ei2++) {
            var ev = visEvs[ei2];
            var ex = snapHalf(cPlot + ((ev.absSec - vs) / ws) * plotW);
            var ecol = ev.isLights ? wt.evLights : wt.evOther;
            var ebg = ev.isLights ? wt.evLightsBg : wt.evOtherBg;
            ctx.save(); ctx.strokeStyle = ecol; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.moveTo(ex, padT); ctx.lineTo(ex, yColBot); ctx.stroke(); ctx.setLineDash([]);
            var lbl = String(ev.label || "") + " (" + Number(ev.secInEp || 0).toFixed(1) + "s)";
            var tw2 = Math.ceil(ctx.measureText(lbl).width) + 6;
            var lx = Math.max(cPlot, Math.min(cPlot + plotW - tw2, ex - tw2 / 2));
            if (lx >= lastRx + 3) {
              lastRx = lx + tw2;
              ctx.fillStyle = ebg; ctx.fillRect(lx, 1, tw2, 11);
              ctx.fillStyle = ecol; ctx.textAlign = "left"; ctx.textBaseline = "top";
              ctx.fillText(lbl, lx + 3, 2);
            }
            ctx.restore();
          }
        }

        var lblStep = 5;
        var candSteps = [5, 10, 15, 30];
        for (var si = 0; si < candSteps.length; si++) {
          if ((candSteps[si] / ws) * plotW >= 48) { lblStep = candSteps[si]; break; }
        }
        ctx.fillStyle = wt.text; ctx.strokeStyle = wt.axis; ctx.lineWidth = 1.0;
        ctx.font = "9.5px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
        var tTick0 = Math.ceil(vs / 5.0) * 5.0;
        for (var sec = tTick0; sec <= ve + 1e-6; sec += 5.0) {
          var tx = snapHalf(cPlot + ((sec - vs) / ws) * plotW);
          ctx.beginPath(); ctx.moveTo(tx, yColBot); ctx.lineTo(tx, yColBot + 3); ctx.stroke();
          if (Math.abs(sec % lblStep) < 0.05 || Math.abs((sec % lblStep) - lblStep) < 0.05) {
            if (tx >= cPlot + 14 && tx <= cPlot + plotW - 14) ctx.fillText(fmtAxisSec(sec), tx, yColBot + 4);
          }
        }
      }
      return true;
    }

    function getRelXY(e) {
      var r = cv.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    function hitTestPoint(px, py) {
      for (var i = 0; i < trackBoxes.length; i++) {
        var tb = trackBoxes[i];
        if (st.chs.length > 1 && px >= tb.rm.x - 2 && px <= tb.rm.x + tb.rm.w + 2 && py >= tb.rm.y - 2 && py <= tb.rm.y + tb.rm.h + 2) {
          return { zone: "remove", idx: i };
        }
        if (px >= tb.grip.x - 2 && px <= tb.grip.x + tb.grip.w + 4 && py >= tb.grip.y - 2 && py <= tb.grip.y + tb.grip.h + 2) {
          return { zone: "grip", idx: i };
        }
        if (px >= tb.plot.x && px <= tb.plot.x + tb.plot.w && py >= tb.plot.y && py <= tb.plot.y + tb.plot.h) {
          return { zone: "plot", idx: i, plotW: tb.plot.w };
        }
      }
      return { zone: null, idx: -1 };
    }

    function computeDropSlot(px, py) {
      var nCh = st.chs.length;
      var cols = (nCh > 1 && st.cols === 2) ? 2 : 1;
      if (!trackBoxes.length) return 0;
      if (cols === 1) {
        for (var i = 0; i < trackBoxes.length; i++) {
          if (py < trackBoxes[i].yTop + trackBoxes[i].chH * 0.5) return i;
        }
        return nCh;
      }
      var best = 0, bestD = 1e12;
      for (var j = 0; j < trackBoxes.length; j++) {
        var cx = trackBoxes[j].plot.x + trackBoxes[j].plot.w * 0.5, cy = trackBoxes[j].yTop + trackBoxes[j].chH * 0.5;
        var d = (px - cx) * (px - cx) + (py - cy) * (py - cy);
        if (d < bestD) { bestD = d; best = j; }
      }
      return best;
    }

    function updateHover(px, py) {
      var h = hitTestPoint(px, py);
      var nr = (h.zone === "remove") ? h.idx : -1, ng = (h.zone === "grip") ? h.idx : -1;
      if (cv) {
        cv.style.cursor = (h.zone === "remove") ? "pointer" : ((h.zone === "grip") ? "grab" : ((h.zone === "plot") ? "ew-resize" : "default"));
      }
      if (nr !== st.hoverRemove || ng !== st.hoverGrip) {
        st.hoverRemove = nr; st.hoverGrip = ng;
        scheduleDraw();
      }
    }

    function commitPan(finalView) {
      var cep = computeCenterEp(finalView);
      st.viewStartSec = finalView;
      st.hlEp = cep;
      saveSessionView();
      drawCanvas();
      if (opts && opts.onPanCommit) opts.onPanCommit(cep, finalView);
    }

    function snapOrCommitPan(curVs) {
      if (!sessionMeta) return;
      var off = st.winSec === 60 ? 15.0 : 0.0;
      var kBase = (curVs + off) / 30.0;
      var cands = [Math.floor(kBase), Math.round(kBase), Math.ceil(kBase)];
      var bestAlign = curVs, bestDist = 1e9, mxEp = Math.max(0, (sessionMeta.nEp || 1) - 1);
      for (var ci = 0; ci < cands.length; ci++) {
        var k = Math.max(0, Math.min(mxEp, cands[ci]));
        var a = clampViewSec(k * 30.0 - off);
        var d = Math.abs(curVs - a);
        if (d < bestDist) { bestDist = d; bestAlign = a; }
      }
      if (bestDist <= 2.5 && bestDist > 0.01) {
        var s0 = curVs, t0 = performance.now(), dur = 135;
        function animSnap(now) {
          if (!sessionMeta) { snapAnimId = null; return; }
          var pFrac = Math.min(1, (now - t0) / dur), ease = 1 - Math.pow(1 - pFrac, 3);
          st.viewStartSec = s0 + (bestAlign - s0) * ease;
          var cep = computeCenterEp(st.viewStartSec);
          st.hlEp = cep;
          if (opts && opts.onPreviewEpoch) opts.onPreviewEpoch(cep);
          drawCanvas();
          if (pFrac < 1) snapAnimId = requestAnimationFrame(animSnap);
          else { snapAnimId = null; commitPan(bestAlign); }
        }
        snapAnimId = requestAnimationFrame(animSnap);
      } else {
        commitPan(curVs);
      }
    }

    if (cv) {
      cv.addEventListener("wheel", function (e) {
        if (!sessionMeta || st.dragMode) return;
        var dx = e.deltaX || 0, dy = e.deltaY || 0;
        var isHoriz = Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 0.5;
        var isShiftWheel = e.shiftKey && Math.abs(dy) > 0.5;
        if (!isHoriz && !isShiftWheel) return;
        e.preventDefault();
        stopSnapAnim();
        var plotW = (trackBoxes[0] && trackBoxes[0].plot.w) || Math.max(120, (cv.clientWidth || 600) - 64);
        var rawDelta = isHoriz ? dx : dy;
        if (e.deltaMode === 1) rawDelta *= 16;
        else if (e.deltaMode === 2) rawDelta *= plotW;
        var nvs = clampViewSec(st.viewStartSec + (rawDelta / plotW) * st.winSec);
        if (Math.abs(nvs - st.viewStartSec) > 1e-4) {
          lastPanDir = nvs > st.viewStartSec ? 1 : -1;
        }
        st.viewStartSec = nvs;
        var cep = computeCenterEp(nvs);
        st.hlEp = cep;
        if (opts && opts.onPreviewEpoch) opts.onPreviewEpoch(cep);
        scheduleDraw();
        wheelSnapTimer = setTimeout(function () {
          wheelSnapTimer = null;
          snapOrCommitPan(st.viewStartSec);
        }, 180);
      }, { passive: false });

      cv.addEventListener("pointerdown", function (e) {
        if (!sessionMeta || e.button !== 0) return;
        var wasSnapping = stopSnapAnim();
        var p = getRelXY(e), h = hitTestPoint(p.x, p.y);
        if (!h.zone) { if (wasSnapping) scheduleDraw(); return; }
        e.preventDefault();
        try { cv.setPointerCapture(e.pointerId); } catch (err) {}
        if (h.zone === "remove") { st.dragMode = "remove"; st.removeIdx = h.idx; return; }
        if (h.zone === "grip") {
          st.dragMode = "reorder"; st.dragChIdx = h.idx; st.dropSlot = h.idx;
          cv.style.cursor = "grabbing"; scheduleDraw(); return;
        }
        if (h.zone === "plot") {
          st.dragMode = "pan"; st.panStartClientX = e.clientX; st.panStartViewSec = st.viewStartSec;
          st.panPlotW = Math.max(40, h.plotW || 600); st.panLastEp = st.hlEp;
          lastPanDir = 0;
          cv.style.cursor = "grabbing";
        }
      });

      cv.addEventListener("pointermove", function (e) {
        if (!sessionMeta) return;
        var p = getRelXY(e);
        if (!st.dragMode) { updateHover(p.x, p.y); return; }
        if (st.dragMode === "reorder") {
          var slot = computeDropSlot(p.x, p.y);
          if (slot !== st.dropSlot) { st.dropSlot = slot; scheduleDraw(); }
          return;
        }
        if (st.dragMode === "pan") {
          var dx = e.clientX - st.panStartClientX;
          var dt = -(dx / st.panPlotW) * st.winSec;
          var nvs = clampViewSec(st.panStartViewSec + dt);
          if (Math.abs(nvs - st.viewStartSec) > 1e-4) {
            lastPanDir = nvs > st.viewStartSec ? 1 : -1;
          }
          st.viewStartSec = nvs;
          var cep = computeCenterEp(nvs);
          if (cep !== st.panLastEp) {
            st.panLastEp = cep; st.hlEp = cep;
            if (opts && opts.onPreviewEpoch) opts.onPreviewEpoch(cep);
          }
          scheduleDraw();
        }
      });

      var onPointerUpOrCancel = function (e) {
        if (!st.dragMode || !sessionMeta) return;
        try { if (cv.hasPointerCapture && cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId); } catch (err) {}
        var mode = st.dragMode; st.dragMode = null;
        var p = getRelXY(e);

        if (mode === "remove") {
          var h = hitTestPoint(p.x, p.y);
          if (h.zone === "remove" && h.idx === st.removeIdx && st.chs.length > 1) {
            st.chs.splice(st.removeIdx, 1);
            saveSessionView();
            scheduleDraw();
            if (opts && opts.onChannelsChanged) opts.onChannelsChanged(st.chs.map(function (x) { return x.name; }));
          }
          st.removeIdx = -1; updateHover(p.x, p.y); return;
        }

        if (mode === "reorder") {
          var from = st.dragChIdx, slot = st.dropSlot, cols = (st.chs.length > 1 && st.cols === 2) ? 2 : 1;
          st.dragChIdx = -1; st.dropSlot = -1;
          if (from >= 0 && from < st.chs.length && slot >= 0) {
            var targetIdx = cols === 1 ? (slot > from ? slot - 1 : slot) : slot;
            targetIdx = Math.max(0, Math.min(st.chs.length - 1, targetIdx));
            if (targetIdx !== from) {
              var item = st.chs.splice(from, 1)[0];
              st.chs.splice(targetIdx, 0, item);
              saveSessionView();
              scheduleDraw();
              if (opts && opts.onChannelsChanged) opts.onChannelsChanged(st.chs.map(function (x) { return x.name; }));
            } else { scheduleDraw(); }
          } else { scheduleDraw(); }
          updateHover(p.x, p.y); return;
        }

        if (mode === "pan") {
          var curVs = st.viewStartSec, moved = Math.abs(curVs - st.panStartViewSec);
          updateHover(p.x, p.y);
          if (moved <= 0.02) { scheduleDraw(); return; }
          snapOrCommitPan(curVs);
        }
      };

      cv.addEventListener("pointerup", onPointerUpOrCancel);
      cv.addEventListener("pointercancel", onPointerUpOrCancel);
      cv.addEventListener("pointerleave", function () {
        if (!st.dragMode && (st.hoverRemove !== -1 || st.hoverGrip !== -1)) {
          st.hoverRemove = -1; st.hoverGrip = -1; scheduleDraw();
        }
      });
    }

    if (b1) b1.onclick = function () { if (st.cols !== 1) { st.cols = 1; saveSessionView(); drawCanvas(); } };
    if (b2) b2.onclick = function () { if (st.cols !== 2) { st.cols = 2; saveSessionView(); drawCanvas(); } };

    function switchWin(nw) {
      if (!sessionMeta || st.winSec === nw) return;
      stopSnapAnim();
      var oldWin = st.winSec, oldVs = st.viewStartSec, ep = st.hlEp;
      var oldAlign = alignedViewSec(ep, oldWin);
      st.winSec = nw;
      if (Math.abs(oldVs - oldAlign) < 0.05) {
        st.viewStartSec = alignedViewSec(ep, nw);
      } else {
        var mid = oldVs + 0.5 * oldWin;
        st.viewStartSec = clampViewSec(mid - 0.5 * nw);
      }
      saveSessionView();
      drawCanvas();
    }

    if (w30) w30.onclick = function () { switchWin(30); };
    if (w60) w60.onclick = function () { switchWin(60); };

    if (ampSel) {
      ampSel.onchange = function () {
        var nv = parseInt(ampSel.value, 10);
        if (isNaN(nv)) nv = 0;
        ampSel.blur();
        if (st.ampScale !== nv) {
          st.ampScale = nv;
          saveSessionView();
          drawCanvas();
        }
      };
    }

    if (addSel) {
      addSel.onchange = function () {
        var v = addSel.value;
        addSel.value = "";
        addSel.blur();
        if (v && opts && opts.onAddChannel) opts.onAddChannel(v);
      };
    }

    if (typeof ResizeObserver !== "undefined") {
      var ro = new ResizeObserver(function () { scheduleDraw(); });
      if (cv) ro.observe(cv);
      if (wrapEl) ro.observe(wrapEl);
    }
    window.addEventListener("resize", function () { scheduleDraw(); });

    return {
      setSession: function (sessionData, activeEp) {
        stopSnapAnim();
        sessionGen++;
        epochCache.clear();
        tileCache.clear();
        inFlightEpochs.clear();
        activePrefetchCount = 0;
        lastTileConfigKey = "";
        lastPanDir = 0;

        sessionMeta = {
          sessionKey: sessionData.session_key,
          nEp: sessionData.n_ep,
          totalSec: sessionData.total_sec,
          ep0ClockSec: sessionData.ep0_clock_sec || 0
        };

        eventsByEp = sessionData.epoch_events || {};
        allEvents = [];
        var epKeys = Object.keys(eventsByEp);
        for (var ki = 0; ki < epKeys.length; ki++) {
          var epIdx = parseInt(epKeys[ki], 10);
          var lst = eventsByEp[epKeys[ki]] || [];
          for (var li = 0; li < lst.length; li++) {
            allEvents.push({
              absSec: epIdx * 30.0 + Number(lst[li][0] || 0),
              secInEp: Number(lst[li][0] || 0),
              label: String(lst[li][1] || ""),
              isLights: String(lst[li][1] || "").toLowerCase().indexOf("lights") >= 0
            });
          }
        }
        allEvents.sort(function (a, b) { return a.absSec - b.absSec; });
        hasAnyEvents = allEvents.length > 0;

        var saved = viewStates[sessionMeta.sessionKey];
        if (saved && saved.chs && saved.chs.length) {
          st.chs = saved.chs.slice();
          st.cols = saved.cols === 2 ? 2 : 1;
          st.winSec = saved.winSec === 60 ? 60 : 30;
          st.ampScale = Number(saved.ampScale || 0);
        } else {
          var chList = [];
          var dChs = sessionData.display_chs || [];
          var metaMap = sessionData.channel_meta || {};
          for (var ci = 0; ci < dChs.length; ci++) {
            if (metaMap[dChs[ci]]) chList.push(metaMap[dChs[ci]]);
          }
          st.chs = chList;
          st.cols = 1;
          st.winSec = 30;
          st.ampScale = 0;
        }
        st.hlEp = Math.max(0, Math.min(sessionMeta.nEp - 1, activeEp || 0));
        st.viewStartSec = alignedViewSec(st.hlEp, st.winSec);
        st.dragMode = null; st.hoverRemove = -1; st.hoverGrip = -1;
        saveSessionView();
        drawCanvas();
      },

      jumpToEpoch: function (epIdx) {
        if (!sessionMeta) return;
        stopSnapAnim();
        var ep = Math.max(0, Math.min(sessionMeta.nEp - 1, Math.round(Number(epIdx) || 0)));
        lastPanDir = ep > st.hlEp ? 1 : (ep < st.hlEp ? -1 : 0);
        st.hlEp = ep;
        st.viewStartSec = alignedViewSec(ep, st.winSec);
        saveSessionView();
        drawCanvas();
      },

      addChannel: function (chMeta) {
        if (!sessionMeta || !chMeta) return;
        for (var i = 0; i < st.chs.length; i++) {
          if (st.chs[i].name === chMeta.name) return;
        }
        st.chs.push(chMeta);
        saveSessionView();
        drawCanvas();
      },

      getVisibleChannelNames: function () {
        return st.chs.map(function (c) { return c.name; });
      },

      setTheme: function (theme) {
        themeMode = normTheme(theme);
        cachedWaveTheme = null;
        scheduleDraw();
      },

      resizeAndDraw: function () {
        scheduleDraw();
      }
    };
  }

  window.OneraWaveforms = { create: createWaveformsController };
})();

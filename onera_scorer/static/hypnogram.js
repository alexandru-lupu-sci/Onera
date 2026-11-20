/* Persistent HTML5 Canvas Hypnogram Controller for Onera SleepScorer */
(function () {
  "use strict";

  function createHypnogramController(opts) {
    var canvas = document.getElementById("onera_hypno_canvas");
    var wrap = document.getElementById("onera_hypno_wrap");
    var tip = document.getElementById("onera_hypno_tip");
    var pBtnEl = document.getElementById("onera_hyp_mode_pan");
    var bBtnEl = document.getElementById("onera_hyp_mode_box");
    var rBtnEl = document.getElementById("onera_hyp_reset");
    var cBtnEl = document.getElementById("onera_hyp_tog_conf");
    var eBtnEl = document.getElementById("onera_hyp_tog_ev");
    var lBtnEl = document.getElementById("onera_hyp_tog_lbl");

    var ctx = canvas ? canvas.getContext("2d") : null;
    var viewStates = {};
    var curRev = null;
    var st = null;
    var payload = null;
    var isArt = new Uint8Array(0);

    var padL = 58, padR = 10, padT = 8, padB = 20;
    var yMin = -0.45, yMax = 4.35;
    var hoverKind = null, hoverIdx = -1;
    var boxRect = null;
    var isDragging = false, dragKind = null, dragStartPx = null, dragStartX0 = 0, dragStartX1 = 0, dragMoved = false;
    var drawRaf = null, panAnimId = null;
    function normTheme(t) {
      if (t === "night" || t === "dark" || t === "light") return t;
      return t ? "dark" : "light";
    }
    var themeMode = normTheme(opts && (opts.theme !== undefined ? opts.theme : opts.isDark));
    var cachedTheme = null;
    var lastCursor = "";
    var tipVisible = false, lastTipHtml = "", tipW = 180, tipH = 60, lastTipLeft = null, lastTipTop = null;

    function getFullBounds() {
      var fullMin = payload ? payload.fullMin : 0.5;
      var fullMax = payload ? payload.fullMax : 1.5;
      return { fullMin: fullMin, fullMax: fullMax, fullSpan: Math.max(1, fullMax - fullMin) };
    }

    function clampRange(a, b) {
      var fb = getFullBounds();
      var minSpan = Math.min(fb.fullSpan, 10);
      var span = b - a;
      if (!isFinite(span) || span <= 0) span = fb.fullSpan;
      if (span < minSpan) span = minSpan;
      if (span > fb.fullSpan) span = fb.fullSpan;
      if (a < fb.fullMin) { a = fb.fullMin; b = fb.fullMin + span; }
      if (b > fb.fullMax) { b = fb.fullMax; a = fb.fullMax - span; }
      return [Math.max(fb.fullMin, a), Math.min(fb.fullMax, b)];
    }

    function stopPanAnim() {
      if (panAnimId !== null) {
        cancelAnimationFrame(panAnimId);
        panAnimId = null;
      }
    }

    function snapHalf(v) { return Math.floor(v) + 0.5; }

    function getMetrics() {
      var W = (wrap && wrap.clientWidth) || (canvas && canvas.clientWidth) || 800;
      var H = (wrap && wrap.clientHeight) || (canvas && canvas.clientHeight) || 280;
      return { W: W, H: H, plotW: Math.max(40, W - padL - padR), plotH: Math.max(40, H - padT - padB) };
    }

    function xToPx(x, m) { return padL + ((x - st.x0) / (st.x1 - st.x0)) * m.plotW; }
    function pxToX(px, m) { return st.x0 + ((px - padL) / m.plotW) * (st.x1 - st.x0); }
    function yToPx(y, m) { return padT + m.plotH - ((y - yMin) / (yMax - yMin)) * m.plotH; }
    function pxToY(py, m) { return yMin + ((padT + m.plotH - py) / m.plotH) * (yMax - yMin); }

    function pad2(n) { return n < 10 ? "0" + n : String(n); }
    function fmtClock(totalSec, withSec) {
      var s = Math.round(totalSec);
      var sec = ((s % 86400) + 86400) % 86400;
      var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), sc = sec % 60;
      return withSec ? (pad2(h) + ":" + pad2(m) + ":" + pad2(sc)) : (pad2(h) + ":" + pad2(m));
    }
    function fmtElapsed(totalSec, withSec) {
      var sec = Math.max(0, Math.round(totalSec));
      var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), sc = sec % 60;
      return withSec ? (pad2(h) + ":" + pad2(m) + ":" + pad2(sc)) : (pad2(h) + ":" + pad2(m));
    }

    function getHypTheme() {
      var mode = themeMode;
      if (cachedTheme && cachedTheme._mode === mode) return cachedTheme;
      if (mode === "dark") {
        cachedTheme = {
          _mode: "dark",
          bg: "#000000", grid: "rgba(122,136,156,0.14)", frame: "rgba(122,136,156,0.28)",
          text: "#a6b3c4", tickStroke: "rgba(122,136,156,0.34)",
          stageLine: "#6fa3d4", s2Ghost: "rgba(176,148,214,0.72)", art: "#cc727c", artStroke: "#000000",
          conf: "rgba(166,179,196,0.30)", wake: "rgba(208,152,74,0.12)",
          evOtherLine: "rgba(160,136,204,0.26)", evLightsLine: "rgba(101,173,134,0.30)",
          evOther: "#9e85c9", evLights: "#65ad86",
          selDot: "#c99d62", selRing: "#b88442", hoverHalo: "rgba(200,210,223,0.45)",
          boxFill: "rgba(111,163,212,0.16)", boxStroke: "#6fa3d4",
          lblHalo: "rgba(0,0,0,0.92)", lblArt: "#cc727c", lblEdit: "#65ad86", lblFlag: "#cc727c",
          dots: { art: "#cc727c", sel: "#c99d62", edit: "#65ad86", flag: "#cc727c", imp: "#6fa3d4", wake: "#b88442", uns: "#5c6878", norm: "#6fa3d4" }
        };
      } else if (mode === "night") {
        cachedTheme = {
          _mode: "night",
          bg: "#161f2e", grid: "rgba(148,163,184,0.16)", frame: "rgba(148,163,184,0.34)",
          text: "#cbd5e1", tickStroke: "rgba(148,163,184,0.42)",
          stageLine: "#85b6e5", s2Ghost: "rgba(198,170,242,0.74)", art: "#eb848e", artStroke: "#161f2e",
          conf: "rgba(203,213,225,0.36)", wake: "rgba(245,198,134,0.18)",
          evOtherLine: "rgba(186,159,224,0.32)", evLightsLine: "rgba(119,201,157,0.36)",
          evOther: "#ba9fe0", evLights: "#77c99d",
          selDot: "#f5c686", selRing: "#e6994d", hoverHalo: "rgba(226,232,240,0.55)",
          boxFill: "rgba(121,179,232,0.18)", boxStroke: "#79b3e8",
          lblHalo: "rgba(22,31,46,0.90)", lblArt: "#f099a2", lblEdit: "#86d4aa", lblFlag: "#f099a2",
          dots: { art: "#eb848e", sel: "#f5c686", edit: "#78c79d", flag: "#eb848e", imp: "#79b3e8", wake: "#f0b873", uns: "#7d8da1", norm: "#79b3e8" }
        };
      } else {
        cachedTheme = {
          _mode: "light",
          bg: "#ffffff", grid: "rgba(100,116,139,0.18)", frame: "rgba(100,116,139,0.34)",
          text: "#334155", tickStroke: "rgba(100,116,139,0.42)",
          stageLine: "#4778a8", s2Ghost: "rgba(138,108,186,0.68)", art: "#de6b76", artStroke: "#ffffff",
          conf: "rgba(71,85,105,0.38)", wake: "rgba(242,185,116,0.22)",
          evOtherLine: "rgba(162,132,201,0.30)", evLightsLine: "rgba(93,176,133,0.36)",
          evOther: "#9d7ec7", evLights: "#59ad82",
          selDot: "#f2b974", selRing: "#d98236", hoverHalo: "rgba(51,65,85,0.48)",
          boxFill: "rgba(91,155,213,0.14)", boxStroke: "#5b9bd5",
          lblHalo: "rgba(255,255,255,0.88)", lblArt: "#c95461", lblEdit: "#3d8f66", lblFlag: "#c95461",
          dots: { art: "#de6b76", sel: "#f2b974", edit: "#5fb388", flag: "#de6b76", imp: "#5b9bd5", wake: "#e8a85c", uns: "#9aa7b8", norm: "#5b9bd5" }
        };
      }
      return cachedTheme;
    }

    function setCanvasCursor(c) {
      if (canvas && c !== lastCursor) { lastCursor = c; canvas.style.cursor = c; }
    }

    function syncCanvasCursor(shiftHeld) {
      if (!st) return;
      if (isDragging) {
        setCanvasCursor(dragKind === "box" ? "crosshair" : "grabbing");
      } else if (hoverKind === "epoch" || hoverKind === "event") {
        setCanvasCursor("pointer");
      } else {
        setCanvasCursor((st.dragMode === "select" || shiftHeld) ? "crosshair" : "grab");
      }
    }

    function updateButtonsUI() {
      if (!st) return;
      if (pBtnEl) pBtnEl.classList.toggle("is-active", st.dragMode === "pan");
      if (bBtnEl) bBtnEl.classList.toggle("is-active", st.dragMode === "select");
      if (cBtnEl) cBtnEl.classList.toggle("is-active", !!st.showConf);
      if (eBtnEl) eBtnEl.classList.toggle("is-active-lavender", !!st.showEvents);
      if (lBtnEl) lBtnEl.classList.toggle("is-active", !!st.showLabels);
      syncCanvasCursor(false);
    }

    function hideTip() {
      if (tip && tipVisible) { tip.style.display = "none"; tipVisible = false; }
    }

    function showTip(htmlStr, cx, cy, m) {
      if (!tip) return;
      var needMeasure = !tipVisible || htmlStr !== lastTipHtml;
      if (needMeasure) {
        tip.style.visibility = "hidden";
        if (tip.style.left !== "0px") { tip.style.left = "0px"; lastTipLeft = 0; }
        if (tip.style.top !== "0px") { tip.style.top = "0px"; lastTipTop = 0; }
        if (htmlStr !== lastTipHtml) { tip.innerHTML = htmlStr; lastTipHtml = htmlStr; }
        if (!tipVisible) { tip.style.display = "block"; tipVisible = true; }
        tipW = tip.offsetWidth || 180;
        tipH = tip.offsetHeight || 60;
      }
      var pad = 6;
      var left = cx + 12, top = cy + 12;
      if (left + tipW > m.W - pad) left = cx - tipW - 12;
      if (top + (tipH || 60) > m.H - pad) top = cy - (tipH || 60) - 10;
      left = Math.round(Math.max(pad, Math.min(Math.max(pad, m.W - tipW - pad), left)));
      top = Math.round(Math.max(pad, Math.min(Math.max(pad, m.H - (tipH || 60) - pad), top)));
      if (left !== lastTipLeft) { tip.style.left = left + "px"; lastTipLeft = left; }
      if (top !== lastTipTop) { tip.style.top = top + "px"; lastTipTop = top; }
      if (needMeasure) tip.style.visibility = "visible";
    }

    function draw() {
      if (!canvas || !ctx || !payload || !st) return false;
      var m = getMetrics();
      if (m.W <= 0 || m.H <= 0) return false;
      var th = getHypTheme();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var bw = Math.round(m.W * dpr), bh = Math.round(m.H * dpr);
      if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, m.W, m.H);
      ctx.fillStyle = th.bg;
      ctx.fillRect(0, 0, m.W, m.H);

      var plotW = m.plotW, plotH = m.plotH;
      var xSpan = st.x1 - st.x0;
      var xScale = plotW / xSpan;
      var yScale = plotH / (yMax - yMin);
      var toX = function (xv) { return padL + (xv - st.x0) * xScale; };
      var toY = function (yv) { return padT + plotH - (yv - yMin) * yScale; };

      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.clip();

      // 1. Wake override shaded intervals
      var wake = payload.wake || [];
      if (wake.length) {
        ctx.fillStyle = th.wake;
        for (var wi = 0; wi < wake.length; wi++) {
          var wx0 = wake[wi][0], wx1 = wake[wi][1];
          if (wx1 < st.x0 || wx0 > st.x1) continue;
          var px0 = toX(wx0), px1 = toX(wx1);
          ctx.fillRect(px0, padT, Math.max(1, px1 - px0), plotH);
        }
      }

      // 2. Horizontal stage gridlines
      ctx.strokeStyle = th.grid;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      for (var yi = 0; yi < payload.yTicks.length; yi++) {
        var gy = snapHalf(toY(payload.yTicks[yi]));
        ctx.moveTo(padL, gy);
        ctx.lineTo(padL + plotW, gy);
      }
      ctx.stroke();

      // 3. Event vertical lines
      if (st.showEvents && payload.lEv.length) {
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        var hasOther = false;
        for (var li = 0; li < payload.lEv.length; li++) {
          var evL = payload.lEv[li], ex = evL[0];
          if (ex < st.x0 || ex > st.x1 || evL[1]) continue;
          var epx = toX(ex);
          ctx.moveTo(epx, padT);
          ctx.lineTo(epx, padT + plotH);
          hasOther = true;
        }
        if (hasOther) { ctx.strokeStyle = th.evOtherLine; ctx.stroke(); }
        ctx.beginPath();
        var hasLights = false;
        for (var lj = 0; lj < payload.lEv.length; lj++) {
          var evL2 = payload.lEv[lj], ex2 = evL2[0];
          if (ex2 < st.x0 || ex2 > st.x1 || !evL2[1]) continue;
          var epx2 = toX(ex2);
          ctx.moveTo(epx2, padT);
          ctx.lineTo(epx2, padT + plotH);
          hasLights = true;
        }
        if (hasLights) { ctx.strokeStyle = th.evLightsLine; ctx.stroke(); }
      }

      // 4. Optional Confidence (%) curve (Auto mode)
      if (!payload.isManual && st.showConf && payload.conf.length) {
        var c0 = Math.max(0, Math.floor(st.x0) - 2);
        var c1 = Math.min(payload.nEp - 1, Math.ceil(st.x1) + 1);
        if (c1 >= c0) {
          ctx.strokeStyle = th.conf;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(toX(c0 + 1), toY(payload.conf[c0]));
          for (var ci = c0 + 1; ci <= c1; ci++) {
            ctx.lineTo(toX(ci + 1), toY(payload.conf[ci]));
          }
          ctx.stroke();
        }
      }

      // 5. Step hypnogram polyline (non-ART segments in pastel slate-blue, ART segments in coral)
      var sY = payload.stageY;
      var sStart = Math.max(0, Math.floor(st.x0) - 2);
      var sEnd = Math.min(payload.nEp - 1, Math.ceil(st.x1) + 1);
      if (sEnd >= sStart) {
        ctx.strokeStyle = th.stageLine;
        ctx.lineWidth = 2.2;
        ctx.lineCap = "butt";
        ctx.lineJoin = "miter";
        ctx.beginPath();
        var inRun = false, needMove = true, prevYVal = 0;
        for (var si = sStart; si <= sEnd; si++) {
          var sv = sY[si];
          if (payload.isManual && sv < -0.5) {
            inRun = false; needMove = true;
          } else if (isArt[si]) {
            prevYVal = sv; inRun = true; needMove = true;
          } else {
            var xL = toX(si + 1), xR = toX(si + 2), sy = toY(sv);
            if (!inRun) {
              ctx.moveTo(xL, sy);
              inRun = true; needMove = false;
            } else {
              if (needMove) { ctx.moveTo(xL, toY(prevYVal)); needMove = false; }
              if (sv !== prevYVal) ctx.lineTo(xL, sy);
            }
            ctx.lineTo(xR, sy);
            prevYVal = sv;
          }
        }
        ctx.stroke();

        ctx.strokeStyle = th.art;
        ctx.lineWidth = 3.0;
        ctx.lineCap = "butt";
        ctx.beginPath();
        var hasArtSeg = false, siA = sStart;
        while (siA <= sEnd) {
          if (!isArt[siA]) { siA++; continue; }
          var rStart = siA, ay = toY(sY[siA]);
          while (siA + 1 <= sEnd && isArt[siA + 1] && sY[siA + 1] === sY[rStart]) siA++;
          var ax0 = toX(rStart + 1), ax1 = toX(siA + 2);
          if ((ax1 - ax0) < 2.5) { var amid = 0.5 * (ax0 + ax1); ax0 = amid - 1.25; ax1 = amid + 1.25; }
          ctx.moveTo(ax0, ay);
          ctx.lineTo(ax1, ay);
          hasArtSeg = true;
          siA++;
        }
        if (hasArtSeg) ctx.stroke();

        // Subtle dashed ghost line for Scorer 2 differing segments in Two-Scorer Consensus Mode
        if (payload.s2StageY && payload.s2DiffMask) {
          var s2Y = payload.s2StageY;
          var s2Mask = payload.s2DiffMask;
          ctx.save();
          ctx.strokeStyle = th.s2Ghost;
          ctx.lineWidth = 1.65;
          ctx.lineCap = "butt";
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          var hasGhost = false;
          for (var gi = sStart; gi <= sEnd; gi++) {
            if (!s2Mask[gi] || s2Y[gi] < -0.2) continue;
            var gxL = toX(gi + 1), gxR = toX(gi + 2), gy = toY(s2Y[gi]);
            if (gi > 0 && s2Mask[gi - 1] && s2Y[gi - 1] >= -0.2) {
              if (s2Y[gi] !== s2Y[gi - 1]) ctx.lineTo(gxL, gy);
            } else {
              var gPrevY = toY(sY[gi]);
              ctx.moveTo(gxL, gPrevY);
              ctx.lineTo(gxL, gy);
            }
            ctx.lineTo(gxR, gy);
            if (gi + 1 >= payload.nEp || !s2Mask[gi + 1] || s2Y[gi + 1] < -0.2) {
              ctx.lineTo(gxR, toY(sY[gi]));
            }
            hasGhost = true;
          }
          if (hasGhost) ctx.stroke();
          ctx.restore();
        }
      }

      // 6. Interactive epoch dots batched by theme-aware pastel color (ART as coral squares)
      var visIdx = payload.visIdx, visKind = payload.visKind || [], visR = payload.visR;
      var colGroups = {}, artDots = [];
      for (var vi = 0; vi < visIdx.length; vi++) {
        var ep1 = visIdx[vi] + 1;
        if (ep1 < st.x0 - 1 || ep1 > st.x1 + 1) continue;
        var origIdx = visIdx[vi];
        if (isArt[origIdx]) {
          artDots.push(toX(ep1), toY(sY[origIdx]), st.selSet.has(origIdx) ? 4.8 : 4.0);
          continue;
        }
        var vk = visKind[vi] || "norm";
        var col = th.dots[vk] || th.dots.norm;
        var rad = st.selSet.has(origIdx) ? 5.4 : visR[vi];
        if (payload.isManual && st.selSet.has(origIdx)) col = th.selDot;
        var grp = colGroups[col];
        if (!grp) { grp = []; colGroups[col] = grp; }
        grp.push(toX(ep1), toY(sY[origIdx]), rad, origIdx);
      }
      var colKeys = Object.keys(colGroups);
      for (var cki = 0; cki < colKeys.length; cki++) {
        var ck = colKeys[cki], arr = colGroups[ck];
        ctx.fillStyle = ck;
        ctx.beginPath();
        for (var ai = 0; ai < arr.length; ai += 4) {
          var dx = arr[ai], dy = arr[ai + 1], dr = arr[ai + 2];
          ctx.moveTo(dx + dr, dy);
          ctx.arc(dx, dy, dr, 0, Math.PI * 2);
        }
        ctx.fill();
      }
      if (artDots.length) {
        ctx.fillStyle = th.art;
        ctx.strokeStyle = th.artStroke;
        ctx.lineWidth = 1.1;
        for (var adi = 0; adi < artDots.length; adi += 3) {
          var ax = artDots[adi], ay2 = artDots[adi + 1], hs = artDots[adi + 2];
          ctx.fillRect(ax - hs, ay2 - hs, hs * 2, hs * 2);
          ctx.strokeRect(ax - hs, ay2 - hs, hs * 2, hs * 2);
        }
      }

      // Hover halo
      if (hoverKind === "epoch" && hoverIdx >= 0 && hoverIdx < visIdx.length) {
        var hOrig = visIdx[hoverIdx];
        var hx = toX(hOrig + 1), hy = toY(sY[hOrig]);
        ctx.strokeStyle = th.hoverHalo;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(hx, hy, (visR[hoverIdx] || 3.5) + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 7. Highlight dots + pastel amber rings on all selected epochs
      if (st.selSet.size > 0) {
        ctx.fillStyle = th.selDot;
        ctx.beginPath();
        st.selSet.forEach(function (selI) {
          if (isArt[selI]) return;
          var sx1 = selI + 1;
          if (sx1 < st.x0 - 2 || sx1 > st.x1 + 2) return;
          var rx = toX(sx1), ry = toY(sY[selI]);
          ctx.moveTo(rx + 5.0, ry);
          ctx.arc(rx, ry, 5.0, 0, Math.PI * 2);
        });
        ctx.fill();
        ctx.strokeStyle = th.selRing;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        st.selSet.forEach(function (selI) {
          var sx1 = selI + 1;
          if (sx1 < st.x0 - 2 || sx1 > st.x1 + 2) return;
          var rx = toX(sx1), ry = toY(sY[selI]);
          ctx.moveTo(rx + 7.5, ry);
          ctx.arc(rx, ry, 7.5, 0, Math.PI * 2);
        });
        ctx.stroke();
      }

      // 8. Dynamically decluttered Epoch # labels (Auto mode)
      if (!payload.isManual && st.showLabels && payload.lblIdx.length) {
        var lIdx = payload.lblIdx, lEd = payload.lblEd;
        var lastXByRow = {};
        ctx.font = "600 9.5px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.lineWidth = 2.6;
        ctx.strokeStyle = th.lblHalo;
        for (var bi = 0; bi < lIdx.length; bi++) {
          var lOrig = lIdx[bi], epN = lOrig + 1;
          if (epN < st.x0 - 1 || epN > st.x1 + 1) continue;
          var lx = toX(epN), lyVal = sY[lOrig], ly = toY(lyVal) + 7.5;
          var rowKey = lyVal.toFixed(2);
          var prevPx = (rowKey in lastXByRow) ? lastXByRow[rowKey] : -1e9;
          var txt = String(epN);
          var tw = txt.length * 5.8;
          var minGap = Math.max(28, tw + 10);
          if ((lx - prevPx) < minGap) continue;
          lastXByRow[rowKey] = lx;
          ctx.strokeText(txt, lx, ly);
          ctx.fillStyle = (lEd[bi] === 2) ? th.lblArt : (lEd[bi] ? th.lblEdit : th.lblFlag);
          ctx.fillText(txt, lx, ly);
        }
      }

      // 9. Top-border Event Marker diamonds
      if (st.showEvents && payload.mEv.length) {
        var evTopY = Math.max(padT + 6, toY(4.25));
        for (var mi = 0; mi < payload.mEv.length; mi++) {
          var mev = payload.mEv[mi], mxv = mev[0];
          if (mxv < st.x0 - 1 || mxv > st.x1 + 1) continue;
          var mpx = toX(mxv);
          var dRad = (hoverKind === "event" && hoverIdx === mi) ? 5.8 : 4.2;
          ctx.beginPath();
          ctx.moveTo(mpx, evTopY - dRad);
          ctx.lineTo(mpx + dRad, evTopY);
          ctx.lineTo(mpx, evTopY + dRad);
          ctx.lineTo(mpx - dRad, evTopY);
          ctx.closePath();
          ctx.fillStyle = mev[1] ? th.evLights : th.evOther;
          ctx.fill();
          ctx.strokeStyle = th.artStroke;
          ctx.lineWidth = 0.9;
          ctx.stroke();
        }
      }

      // 10. Box-select drag overlay
      if (boxRect) {
        var bx = Math.min(boxRect.x0, boxRect.x1), by = Math.min(boxRect.y0, boxRect.y1);
        var bw2 = Math.abs(boxRect.x1 - boxRect.x0), bh2 = Math.abs(boxRect.y1 - boxRect.y0);
        ctx.fillStyle = th.boxFill;
        ctx.fillRect(bx, by, bw2, bh2);
        ctx.save();
        ctx.strokeStyle = th.boxStroke;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx, by, bw2, bh2);
        ctx.restore();
      }

      ctx.restore();

      // Plot frame border
      var fX0 = snapHalf(padL), fY0 = snapHalf(padT), fX1 = snapHalf(padL + plotW), fY1 = snapHalf(padT + plotH);
      ctx.strokeStyle = th.frame;
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(fX0, fY0);
      ctx.lineTo(fX0, fY1);
      ctx.lineTo(fX1, fY1);
      ctx.stroke();

      // Y-axis stage labels
      ctx.fillStyle = th.text;
      ctx.font = "600 10.5px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (var yi2 = 0; yi2 < payload.yTicks.length; yi2++) {
        ctx.fillText(payload.yLabels[yi2], padL - 6, toY(payload.yTicks[yi2]));
      }

      // Dynamic zoom-adaptive clock/elapsed X-axis ticks
      var stepCands = [30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400];
      var visSec = xSpan * 30.0;
      var maxTicks = Math.max(4, Math.floor(plotW / 76));
      var stepSec = stepCands[stepCands.length - 1];
      for (var sci = 0; sci < stepCands.length; sci++) {
        if ((visSec / stepCands[sci]) <= maxTicks) { stepSec = stepCands[sci]; break; }
      }
      var withSec = stepSec < 60;
      var hasClock = (payload.measStartSec !== null && payload.measStartSec !== undefined);
      var tStartSec = (st.x0 - 1.0) * 30.0;
      var lastTickRight = -1e9;
      ctx.font = "10.5px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = th.text;
      ctx.strokeStyle = th.tickStroke;
      ctx.lineWidth = 1.0;

      var drawTick = function (xVal, lblStr) {
        if (xVal < st.x0 - 1e-6 || xVal > st.x1 + 1e-6) return;
        var tx = snapHalf(toX(xVal));
        if (tx < padL - 2 || tx > padL + plotW + 2) return;
        ctx.beginPath();
        ctx.moveTo(tx, padT + plotH);
        ctx.lineTo(tx, padT + plotH + 3);
        ctx.stroke();
        var lw = lblStr.length * 6.2;
        if ((tx - lw * 0.5) < (lastTickRight + 10)) return;
        if ((tx - lw * 0.5) < 4 || (tx + lw * 0.5) > (m.W - 4)) return;
        lastTickRight = tx + lw * 0.5;
        ctx.fillText(lblStr, tx, padT + plotH + 5);
      };

      if (hasClock) {
        var absStart = payload.measStartSec + tStartSec;
        var k0 = Math.ceil((absStart - 1e-6) / stepSec);
        for (var k = k0; k < k0 + maxTicks + 4; k++) {
          var absS = k * stepSec;
          var xV = 1.0 + (absS - payload.measStartSec) / 30.0;
          if (xV > st.x1 + 1e-6) break;
          drawTick(xV, fmtClock(absS, withSec));
        }
      } else {
        var relStart = Math.max(0, tStartSec);
        var k0e = Math.ceil((relStart - 1e-6) / stepSec);
        for (var ke = k0e; ke < k0e + maxTicks + 4; ke++) {
          var relS = ke * stepSec;
          var xVe = 1.0 + relS / 30.0;
          if (xVe > st.x1 + 1e-6) break;
          drawTick(xVe, fmtElapsed(relS, withSec));
        }
      }
      return true;
    }

    function scheduleDraw() {
      if (drawRaf !== null || !payload) return;
      drawRaf = requestAnimationFrame(function () {
        drawRaf = null;
        draw();
      });
    }

    function hitTest(cx, cy, m) {
      if (!payload || !st) return { kind: null, idx: -1 };
      if (cx < padL - 8 || cx > padL + m.plotW + 8 || cy < padT - 8 || cy > padT + m.plotH + 8) {
        return { kind: null, idx: -1 };
      }
      if (st.showEvents && payload.mEv.length) {
        var evTopY = Math.max(padT + 6, yToPx(4.25, m));
        if (Math.abs(cy - evTopY) <= 12) {
          var bestEv = -1, bestEvD = 12;
          for (var mi = 0; mi < payload.mEv.length; mi++) {
            var mxv = payload.mEv[mi][0];
            if (mxv < st.x0 - 1 || mxv > st.x1 + 1) continue;
            var d = Math.hypot(cx - xToPx(mxv, m), cy - evTopY);
            if (d < bestEvD) { bestEvD = d; bestEv = mi; }
          }
          if (bestEv >= 0) return { kind: "event", idx: bestEv };
        }
      }
      var bestEp = -1, bestEpD = 14;
      var visIdx = payload.visIdx, sY = payload.stageY;
      for (var vi = 0; vi < visIdx.length; vi++) {
        var origIdx = visIdx[vi], ep1 = origIdx + 1;
        if (ep1 < st.x0 - 1 || ep1 > st.x1 + 1) continue;
        var d2 = Math.hypot(cx - xToPx(ep1, m), cy - yToPx(sY[origIdx], m));
        if (d2 < bestEpD) { bestEpD = d2; bestEp = vi; }
      }
      if (bestEp >= 0) return { kind: "epoch", idx: bestEp };
      return { kind: null, idx: -1 };
    }

    function localXY(e) {
      var rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function animateEnsureVisible(activeEp) {
      if (!payload || !st) return;
      var fb = getFullBounds();
      var curSpan = st.x1 - st.x0;
      if (curSpan >= fb.fullSpan - 1e-6) return;
      var pad = curSpan * 0.08;
      var tx = activeEp + 1;
      if (tx >= (st.x0 + pad) && tx <= (st.x1 - pad)) return;
      var tgt = clampRange(tx - curSpan * 0.5, tx + curSpan * 0.5);
      var sX0 = st.x0, sX1 = st.x1, eX0 = tgt[0], eX1 = tgt[1];
      if (Math.abs(eX0 - sX0) <= 1e-3 && Math.abs(eX1 - sX1) <= 1e-3) return;
      var t0 = null;
      stopPanAnim();
      function step(ts) {
        if (t0 === null) t0 = ts;
        var p = Math.min(1, (ts - t0) / 160);
        var ease = 1 - Math.pow(1 - p, 3);
        st.x0 = sX0 + (eX0 - sX0) * ease;
        st.x1 = sX1 + (eX1 - sX1) * ease;
        draw();
        if (p < 1) panAnimId = requestAnimationFrame(step);
        else panAnimId = null;
      }
      panAnimId = requestAnimationFrame(step);
    }

    if (canvas) {
      canvas.addEventListener("contextmenu", function (e) {
        if (e.ctrlKey || e.metaKey) e.preventDefault();
      });

      canvas.addEventListener("wheel", function (e) {
        if (!payload || !st) return;
        var m = getMetrics();
        var pt = localXY(e);
        if (pt.x < padL || pt.x > padL + m.plotW || pt.y < padT || pt.y > padT + m.plotH) return;
        e.preventDefault();
        stopPanAnim();
        hideTip();
        var fb = getFullBounds();
        var dx = e.deltaX || 0;
        var dy = e.deltaY || 0;
        if (e.deltaMode === 1) { dx *= 16; dy *= 16; }
        else if (e.deltaMode === 2) { dx *= Math.max(100, m.W || 600); dy *= Math.max(100, m.H || 300); }

        var curSpan = st.x1 - st.x0;
        if (!e.ctrlKey && Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 0.5) {
          var shiftEp = (dx / Math.max(80, m.plotW)) * curSpan;
          var pr = clampRange(st.x0 + shiftEp, st.x1 + shiftEp);
          st.x0 = pr[0]; st.x1 = pr[1];
          scheduleDraw();
          return;
        }

        var gain = e.ctrlKey ? 0.006 : 0.0014;
        var factor = (dy === 0) ? 1 : Math.exp((dy > 0 ? 1 : -1) * Math.min(0.28, Math.abs(dy) * gain));
        var minSpan = Math.min(fb.fullSpan, 10);
        var newSpan = Math.max(minSpan, Math.min(fb.fullSpan, curSpan * factor));
        var frac = (pt.x - padL) / m.plotW;
        var anchorX = st.x0 + frac * curSpan;
        var nr = clampRange(anchorX - frac * newSpan, anchorX + (1 - frac) * newSpan);
        st.x0 = nr[0]; st.x1 = nr[1];
        scheduleDraw();
      }, { passive: false });

      canvas.addEventListener("pointerdown", function (e) {
        if (!payload || !st || e.button !== 0) return;
        stopPanAnim();
        var m = getMetrics();
        var pt = localXY(e);
        if (pt.x < padL - 4 || pt.x > padL + m.plotW + 4 || pt.y < padT - 4 || pt.y > padT + m.plotH + 4) return;
        isDragging = true;
        dragMoved = false;
        dragStartPx = pt;
        dragStartX0 = st.x0;
        dragStartX1 = st.x1;
        dragKind = (st.dragMode === "select" || e.shiftKey) ? "box" : "pan";
        syncCanvasCursor(e.shiftKey);
        try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
        hideTip();
      });

      canvas.addEventListener("pointermove", function (e) {
        if (!payload || !st) return;
        var m = getMetrics();
        var pt = localXY(e);
        if (isDragging && dragStartPx) {
          var dx = pt.x - dragStartPx.x, dy = pt.y - dragStartPx.y;
          if (!dragMoved && Math.hypot(dx, dy) > 4) dragMoved = true;
          if (dragMoved) {
            if (dragKind === "pan") {
              var span = dragStartX1 - dragStartX0;
              var shift = (dx / m.plotW) * span;
              var nr = clampRange(dragStartX0 - shift, dragStartX1 - shift);
              st.x0 = nr[0]; st.x1 = nr[1];
              scheduleDraw();
            } else {
              boxRect = {
                x0: Math.max(padL, Math.min(padL + m.plotW, dragStartPx.x)),
                y0: Math.max(padT, Math.min(padT + m.plotH, dragStartPx.y)),
                x1: Math.max(padL, Math.min(padL + m.plotW, pt.x)),
                y1: Math.max(padT, Math.min(padT + m.plotH, pt.y))
              };
              scheduleDraw();
            }
          }
          return;
        }
        var hit = hitTest(pt.x, pt.y, m);
        if (hit.kind !== hoverKind || hit.idx !== hoverIdx) {
          hoverKind = hit.kind; hoverIdx = hit.idx;
          scheduleDraw();
        }
        syncCanvasCursor(e.shiftKey);
        if (hit.kind === "event") {
          showTip(payload.mEv[hit.idx][2], pt.x, pt.y, m);
        } else if (hit.kind === "epoch") {
          showTip(payload.visHover[hit.idx], pt.x, pt.y, m);
        } else {
          hideTip();
        }
      });

      var finishDrag = function (e) {
        if (!isDragging || !payload || !st) return;
        var m = getMetrics();
        var pt = localXY(e);
        try { canvas.releasePointerCapture(e.pointerId); } catch (err) {}
        var wasMoved = dragMoved, wasKind = dragKind;
        isDragging = false; dragKind = null; dragStartPx = null; dragMoved = false;
        var curBox = boxRect;
        boxRect = null;
        var hit = hitTest(pt.x, pt.y, m);
        hoverKind = hit.kind; hoverIdx = hit.idx;
        syncCanvasCursor(e.shiftKey);

        if (!wasMoved) {
          if (hit.kind === "epoch") {
            var clickedOrig = payload.visIdx[hit.idx];
            var isAdditive = Boolean(e.ctrlKey || e.metaKey);
            if (isAdditive) {
              if (st.selSet.has(clickedOrig)) st.selSet.delete(clickedOrig);
              else st.selSet.add(clickedOrig);
            } else {
              st.selSet = new Set([clickedOrig]);
            }
            draw();
            if (opts && opts.onSelectEpochs) {
              opts.onSelectEpochs({ type: "click", epochs: [clickedOrig], additive: isAdditive });
            }
          } else {
            scheduleDraw();
          }
          return;
        }

        if (wasKind === "box" && curBox) {
          var xMin = Math.min(pxToX(curBox.x0, m), pxToX(curBox.x1, m));
          var xMax = Math.max(pxToX(curBox.x0, m), pxToX(curBox.x1, m));
          var yLow = Math.min(pxToY(curBox.y0, m), pxToY(curBox.y1, m));
          var yHigh = Math.max(pxToY(curBox.y0, m), pxToY(curBox.y1, m));
          var matched = [], visIdx = payload.visIdx, sY = payload.stageY;
          for (var vi = 0; vi < visIdx.length; vi++) {
            var oIdx = visIdx[vi], epN = oIdx + 1, yv = sY[oIdx];
            if (epN >= xMin && epN <= xMax && yv >= yLow && yv <= yHigh) {
              matched.push(oIdx);
              st.selSet.add(oIdx);
            }
          }
          draw();
          if (matched.length && opts && opts.onSelectEpochs) {
            opts.onSelectEpochs({ type: "box", epochs: matched, additive: true });
          }
        } else if (wasKind === "pan") {
          draw();
        }
      };

      canvas.addEventListener("pointerup", finishDrag);
      canvas.addEventListener("pointercancel", finishDrag);
      canvas.addEventListener("pointerleave", function () {
        if (!isDragging) {
          hoverKind = null; hoverIdx = -1;
          syncCanvasCursor(false);
          hideTip();
          scheduleDraw();
        }
      });
      canvas.addEventListener("dblclick", function () {
        if (!payload || !st) return;
        stopPanAnim();
        hideTip();
        var fb = getFullBounds();
        st.x0 = fb.fullMin; st.x1 = fb.fullMax;
        draw();
      });
    }

    if (pBtnEl) pBtnEl.onclick = function () { if (st) { st.dragMode = "pan"; updateButtonsUI(); } };
    if (bBtnEl) bBtnEl.onclick = function () { if (st) { st.dragMode = "select"; updateButtonsUI(); } };
    if (rBtnEl) rBtnEl.onclick = function () {
      if (!payload || !st) return;
      stopPanAnim(); hideTip();
      var fb = getFullBounds();
      st.x0 = fb.fullMin; st.x1 = fb.fullMax;
      draw();
    };
    if (cBtnEl) cBtnEl.onclick = function () { if (st) { st.showConf = !st.showConf; updateButtonsUI(); draw(); } };
    if (eBtnEl) eBtnEl.onclick = function () { if (st) { st.showEvents = !st.showEvents; updateButtonsUI(); draw(); } };
    if (lBtnEl) lBtnEl.onclick = function () { if (st) { st.showLabels = !st.showLabels; updateButtonsUI(); draw(); } };

    if (typeof ResizeObserver !== "undefined" && wrap) {
      new ResizeObserver(function () { scheduleDraw(); }).observe(wrap);
    }
    window.addEventListener("resize", function () { scheduleDraw(); });

    function applyDerived(derived) {
      payload.stageY = derived.stageY;
      payload.s2StageY = derived.s2StageY || null;
      payload.s2DiffMask = derived.s2DiffMask || null;
      payload.artIdx = derived.artIdx;
      payload.visIdx = derived.visIdx;
      payload.visKind = derived.visKind;
      payload.visR = derived.visR;
      payload.visHover = derived.visHover;
      payload.lblIdx = derived.lblIdx;
      payload.lblEd = derived.lblEd;
      isArt = new Uint8Array(payload.nEp);
      for (var i = 0; i < payload.artIdx.length; i++) {
        var idx = payload.artIdx[i];
        if (idx >= 0 && idx < payload.nEp) isArt[idx] = 1;
      }
      if (payload.isManual) {
        payload.yTicks = [-0.65, 0, 1, 2, 3, 4];
        payload.yLabels = ["Unscored", derived.hasS4 ? "N3/S4" : "N3", "N2", "N1", "REM", "W"];
        yMin = -0.95; yMax = 4.35;
      } else {
        payload.yTicks = [0, 1, 2, 3, 4];
        payload.yLabels = [derived.hasS4 ? "N3/S4" : "N3", "N2", "N1", "REM", "W (100%)"];
        yMin = -0.45; yMax = 4.35;
      }
    }

    return {
      setSession: function (sessionData, derived, activeEp, selEps) {
        stopPanAnim();
        hideTip();
        hoverKind = null; hoverIdx = -1; boxRect = null;
        var nEp = sessionData.n_ep;
        var isManual = Boolean(sessionData.is_manual_mode);
        var hasYasa = sessionData.has_yasa !== undefined ? Boolean(sessionData.has_yasa) : !isManual;
        padL = isManual ? 60 : 58;
        curRev = sessionData.recording_id + "__" + sessionData.scoring_mode + "__" + sessionData.thresh;

        var confScaled = [];
        if (hasYasa && sessionData.epochs && sessionData.epochs.confidence_pct) {
          var cp = sessionData.epochs.confidence_pct;
          for (var i = 0; i < cp.length; i++) {
            confScaled.push(cp[i] === null ? 0 : Math.round((cp[i] / 25.0) * 100) / 100);
          }
        }
        payload = {
          uiRev: curRev,
          isManual: isManual,
          hasYasa: hasYasa,
          nEp: nEp,
          fullMin: 0.5,
          fullMax: nEp + 1.5,
          measStartSec: sessionData.meas_start_sec,
          wake: sessionData.wake_overrides || [],
          mEv: (sessionData.hypno_events && sessionData.hypno_events.mEv) || [],
          lEv: (sessionData.hypno_events && sessionData.hypno_events.lEv) || [],
          conf: confScaled,
          yTicks: [],
          yLabels: []
        };
        applyDerived(derived);

        var prevSt = viewStates[curRev];
        var fb = getFullBounds();
        if (!prevSt || !prevSt.initialized) {
          var initSpan = 240.0;
          var initEpX = Math.max(1, Math.min(nEp, (activeEp || 0) + 1));
          var ix0, ix1;
          if (fb.fullSpan <= initSpan || initEpX <= 180.0) {
            ix0 = fb.fullMin;
            ix1 = Math.min(fb.fullMax, fb.fullMin + initSpan);
          } else {
            ix0 = initEpX - initSpan * 0.5;
            ix1 = initEpX + initSpan * 0.5;
            if (ix0 < fb.fullMin) { ix0 = fb.fullMin; ix1 = Math.min(fb.fullMax, fb.fullMin + initSpan); }
            else if (ix1 > fb.fullMax) { ix1 = fb.fullMax; ix0 = Math.max(fb.fullMin, fb.fullMax - initSpan); }
          }
          st = {
            initialized: true,
            x0: ix0,
            x1: ix1,
            dragMode: "pan",
            showConf: false,
            showEvents: payload.mEv.length > 0,
            showLabels: !isManual,
            selSet: new Set(selEps || [])
          };
          viewStates[curRev] = st;
        } else {
          st = prevSt;
          var cr = clampRange(st.x0, st.x1);
          st.x0 = cr[0]; st.x1 = cr[1];
          st.selSet = new Set(selEps || []);
          if (payload.mEv.length === 0) st.showEvents = false;
        }

        if (cBtnEl) cBtnEl.classList.toggle("hidden", !hasYasa);
        if (lBtnEl) lBtnEl.classList.toggle("hidden", isManual);
        if (eBtnEl) {
          eBtnEl.classList.toggle("hidden", payload.mEv.length === 0);
          eBtnEl.textContent = "Events (" + payload.mEv.length + ")";
        }

        updateButtonsUI();
        draw();
      },

      updateState: function (derived, activeEp, selEps, ensureVisible) {
        if (!payload || !st) return;
        applyDerived(derived);
        st.selSet = new Set(selEps || []);
        if (ensureVisible) animateEnsureVisible(activeEp);
        scheduleDraw();
      },

      previewEpoch: function (epIdx) {
        if (!payload || !st) return;
        var mx = Math.max(0, (payload.nEp || 1) - 1);
        var ep = Math.max(0, Math.min(mx, Math.round(Number(epIdx) || 0)));
        stopPanAnim();
        st.selSet = new Set([ep]);
        var span = st.x1 - st.x0;
        var sx1 = ep + 1;
        var padX = Math.min(Math.max(1.0, span * 0.08), span * 0.4);
        if (sx1 < st.x0 + padX || sx1 > st.x1 - padX) {
          var fb = getFullBounds();
          if (span < fb.fullSpan - 1e-6) {
            var nx0 = sx1 - 0.5 * span, nx1 = nx0 + span;
            if (nx0 < fb.fullMin) { nx0 = fb.fullMin; nx1 = nx0 + span; }
            if (nx1 > fb.fullMax) { nx1 = fb.fullMax; nx0 = nx1 - span; }
            st.x0 = nx0; st.x1 = nx1;
          }
        }
        scheduleDraw();
      },

      setTheme: function (theme) {
        themeMode = normTheme(theme);
        cachedTheme = null;
        updateButtonsUI();
        scheduleDraw();
      },

      resizeAndDraw: function () {
        scheduleDraw();
      }
    };
  }

  window.OneraHypnogram = { create: createHypnogramController };
})();

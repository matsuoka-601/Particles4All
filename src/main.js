
import { Sim } from './sim.js';
import { Renderer, Camera, screenRay } from './render.js';
import { SurfaceMesh } from './mesh.js';
import { Solids, halfExtent } from './solids.js';
import { RayMarch } from './ray.js';
import { FluidSSFR } from './ssfr.js';
import { GpuTimer } from './gputimer.js';
import { Environment } from './env.js';
import { PRESETS, defaultParams, blockCapacity } from './scene.js';
import { iniToQuery, UNSUPPORTED } from './settingsini.js';

const canvas = document.getElementById('view');
const panel = {
  stats: document.getElementById('stats'),
  adapter: document.getElementById('adapter'),
  scene: document.getElementById('sceneinfo'),
  timing: document.getElementById('timing'),
};

function bodyHalfExtent(b) {
  const h = halfExtent(b);
  if (b.shape === 'sphere') return [h[0], h[0], h[0]];
  if (b.shape === 'torus') return [h[0] + h[1], h[1], h[0] + h[1]];
  return h;
}

function dragTarget(want, b, pose, box) {
  const h = bodyHalfExtent(b);
  const R = pose.rot;
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const ext = Math.abs(R[0 * 3 + i]) * h[0] + Math.abs(R[1 * 3 + i]) * h[1] +
                Math.abs(R[2 * 3 + i]) * h[2];
    out[i] = Math.min(Math.max(want[i], ext), box[i] - ext);
  }
  return out;
}

function pickBody(sim, ro, rd) {
  let best = -1, bestT = 1e30;
  const bodies = sim.bodies || [];
  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    const pose = sim.bodyPose[i];
    const R = pose.rot;
    const w = [ro[0] - pose.centre[0], ro[1] - pose.centre[1], ro[2] - pose.centre[2]];

    const o = [R[0] * w[0] + R[1] * w[1] + R[2] * w[2],
               R[3] * w[0] + R[4] * w[1] + R[5] * w[2],
               R[6] * w[0] + R[7] * w[1] + R[8] * w[2]];
    const d = [R[0] * rd[0] + R[1] * rd[1] + R[2] * rd[2],
               R[3] * rd[0] + R[4] * rd[1] + R[5] * rd[2],
               R[6] * rd[0] + R[7] * rd[1] + R[8] * rd[2]];
    const he = halfExtent(b);
    let t = -1;
    if (b.shape === 'box') {
      let t0 = -1e30, t1 = 1e30, miss = false;
      for (let a = 0; a < 3; a++) {
        if (Math.abs(d[a]) < 1e-9) {
          if (o[a] < -he[a] || o[a] > he[a]) { miss = true; break; }
          continue;
        }
        let ta = (-he[a] - o[a]) / d[a], tb = (he[a] - o[a]) / d[a];
        if (ta > tb) { const s = ta; ta = tb; tb = s; }
        t0 = Math.max(t0, ta);
        t1 = Math.min(t1, tb);
      }
      if (!miss && t1 > Math.max(t0, 0)) t = t0 > 0 ? t0 : t1;
    } else {
      const rad = b.shape === 'sphere' ? he[0] : he[0] + he[1];
      const bq = o[0] * d[0] + o[1] * d[1] + o[2] * d[2];
      const cq = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - rad * rad;
      const disc = bq * bq - cq;
      if (disc > 0) {
        const sq = Math.sqrt(disc);
        const ta = -bq - sq, tb = -bq + sq;
        if (tb > 0) t = ta > 0 ? ta : tb;
      }
    }
    if (t > 0 && t < bestT) { bestT = t; best = i + 1; }
  }
  if (best > 0)
    return { body: best, t: bestT,
             hit: [ro[0] + rd[0] * bestT, ro[1] + rd[1] * bestT, ro[2] + rd[2] * bestT] };
  return { body: -1, t: 0, hit: [0, 0, 0] };
}

const POUR_SLACK = 1.25;

function pourLatticeAcross(width, spacing) {
  return Math.max(1, Math.round(width / Math.max(spacing, 1e-4)));
}

function pourRatePerSecond(ui, spacing) {
  const k = pourLatticeAcross(ui.pourWidth, spacing);
  return Math.floor(k * k * ui.pourSpeed / Math.max(spacing * POUR_SLACK, 1e-4));
}

let pourSeed = 22222;
function pourJitter(step) {
  pourSeed = (pourSeed * 1103515245 + 12345) & 0x7fffffff;
  return ((pourSeed >>> 8 & 0xffff) / 65535 - 0.5) * 0.002 * step;
}

function pourStep(sim, ui, simDt) {
  if (!ui.pouring || ui.pourLeft <= 0 || simDt <= 0) return 0;
  const p = sim.params;
  const d = p.spacing;
  const box = p.box;
  const step = d;
  const layerStep = d * POUR_SLACK;
  ui.pourExtruded += ui.pourSpeed * simDt;
  let layers = 0;
  for (let m = ui.pourNextLayer; m * layerStep <= ui.pourExtruded; ++m) ++layers;
  if (layers <= 0) return 0;

  layers = Math.min(layers, 32);
  const k = pourLatticeAcross(ui.pourWidth, d);
  const span = (k - 1) * step;

  const nozzle = [sim.h + d, ui.pourHeight * box[1], 0.5 * box[2]];
  const tilt = ui.pourTilt * Math.PI / 180;
  const jet = [ui.pourSpeed * Math.cos(tilt), -ui.pourSpeed * Math.sin(tilt), 0];
  const jl = Math.hypot(jet[0], jet[1], jet[2]) || 1;
  const dir = [jet[0] / jl, jet[1] / jl, jet[2] / jl];
  const pos = [], vel = [];

  const g = p.gravity;
  for (let l = 0; l < layers && pos.length / 3 < ui.pourLeft; ++l) {
    const m = ui.pourNextLayer + l;
    const behind = ui.pourExtruded - m * layerStep;
    const t = behind / Math.max(ui.pourSpeed, 1e-6);
    const cx = nozzle[0] + jet[0] * t;
    const cy = nozzle[1] + jet[1] * t - 0.5 * g * t * t;
    const vy = jet[1] - g * t;
    if (cx >= box[0] - d || cy <= d) break;
    for (let iy = 0; iy < k && pos.length / 3 < ui.pourLeft; ++iy)
      for (let iz = 0; iz < k && pos.length / 3 < ui.pourLeft; ++iz) {
        const y = cy - 0.5 * span + iy * step + pourJitter(step);
        if (y <= d || y >= box[1] - d) continue;
        pos.push(cx + pourJitter(step), y,
                 nozzle[2] - 0.5 * span + iz * step + pourJitter(step));
        vel.push(jet[0], vy, 0);
      }
  }
  ui.pourNextLayer += layers;
  if (!pos.length) return 0;
  if (ui.pourTrace > 0) {
    let lo = 1e9, hi = -1e9;
    for (let i = 0; i < pos.length; i += 3) { lo = Math.min(lo, pos[i]); hi = Math.max(hi, pos[i]); }
    console.info(`POUR dt=${simDt.toFixed(4)} layers=${layers} k=${k} n=${pos.length / 3}` +
                 ` x ${lo.toFixed(3)}..${hi.toFixed(3)} travel=${(ui.pourSpeed * simDt).toFixed(3)}`);
    --ui.pourTrace;
  }
  const added = sim.appendFluid(pos, vel);
  ui.pourLeft = Math.max(0, ui.pourLeft - added);
  if (ui.pourLeft === 0) ui.pouring = false;
  return added;
}

const loading = {
  el: document.getElementById('loading'),
  say(what) {
    const n = document.getElementById('loadnote');
    if (n) n.textContent = what;
  },

  done() {
    if (!this.el || this.el.classList.contains('gone')) return;
    this.el.classList.add('gone');
    setTimeout(() => { if (this.el) this.el.style.display = 'none'; }, 400);
  },
};

function fail(msg) {
  loading.done();
  document.getElementById('nogpu').style.display = 'block';
  document.getElementById('nogpu').textContent = msg;
  window.__result = { error: msg };
  window.__done = true;
}

if (!navigator.gpu) {
  fail('This browser has no WebGPU. Chrome or Edge 113+, or Safari 18+, and a page served over http(s).');
} else {
  loading.say('asking for the GPU…');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
               || await navigator.gpu.requestAdapter();
  if (!adapter) {
    fail('WebGPU is present but the browser gave no adapter.\n\n' +
         '1. Restart the browser. After a few GPU process crashes it falls back\n' +
         '   to software for the rest of the session, and WebGPU goes with it.\n\n' +
         '2. Open chrome://gpu (edge://gpu) and look at the WebGPU line and the\n' +
         '   "Problems Detected" list.\n\n' +
         '3. chrome://settings/system - "Use graphics acceleration when\n' +
         '   available" has to be on.\n\n' +
         'Failing that, start the browser with\n' +
         '  --enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan');
  } else {
    const L = adapter.limits;

    const canTime = new URLSearchParams(location.search).get('timing') !== '0'
                 && adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: canTime ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBuffersPerShaderStage:
          Math.min(16, L.maxStorageBuffersPerShaderStage),
        maxStorageBufferBindingSize:
          Math.min(1 << 30, L.maxStorageBufferBindingSize),
        maxBufferSize: Math.min(1 << 30, L.maxBufferSize),
        maxComputeInvocationsPerWorkgroup:
          Math.min(256, L.maxComputeInvocationsPerWorkgroup),
      },
    });
    device.addEventListener?.('uncapturederror', e => {
      console.error('WebGPU error:', e.error?.message || e.error);
      window.__gpuError = String(e.error?.message || e.error);
    });
    const ctx = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: 'opaque' });
    const info = adapter.info || {};
    panel.adapter.textContent = `${info.vendor || 'gpu'} ${info.architecture || ''}`.trim();

    const sim = new Sim(device);
    const renderer = new Renderer(device, ctx, format);
    const mesh = new SurfaceMesh(device, format);
    const solids = new Solids(device, format);
    const ray = new RayMarch(device, format);
    const ssfr = new FluidSSFR(device, format);

    const env = new Environment(device);
    ray.env = env;
    ssfr.env = env;
    const cam = new Camera();

    const simTimer = new GpuTimer(device, 512, 'simTimer');
    const renderTimer = new GpuTimer(device, 48, 'renderTimer');
    sim.timer = simTimer;

    sim.bodyPhases = new URLSearchParams(location.search).get('bodyphases') === '1';
    renderer.timer = renderTimer;
    mesh.timer = renderTimer;
    solids.timer = renderTimer;
    ray.timer = renderTimer;
    ssfr.timer = renderTimer;
    window.__timers = { sim: simTimer, render: renderTimer };

    const q = new URLSearchParams(location.search);

    let qualityTouched = false;

    const startPreset = PRESETS[q.get('preset')] ? q.get('preset') : 'small';

    let presetNote = null;
    if (PRESETS[startPreset]?.ini) {
      const path = PRESETS[startPreset].ini;
      try {
        const r = await fetch(path, { cache: 'no-cache' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const { query, applied, skipped, missingPanorama } = iniToQuery(await r.text());
        let taken = 0;
        for (const [k, v] of new URLSearchParams(query))
          if (!q.has(k)) { q.set(k, v); taken++; }
        const unknown = skipped.filter(k => !UNSUPPORTED.includes(k));
        console.info(`${path}: ${applied.length} settings read, ${taken} applied` +
          (taken < applied.length ? ` (${applied.length - taken} overridden by the URL)` : '') +
          (skipped.length ? `, skipped ${skipped.join(', ')}` : ''));
        if (unknown.length) presetNote = `${path}: ${unknown.length} unknown keys (${unknown.join(', ')})`;
        if (missingPanorama && !q.has('cubemap'))
          console.info(`${path}: the panorama has to be at env/${missingPanorama} on this server`);
      } catch (e) {
        presetNote = `${path} could not be read (${e.message}) - this preset's ` +
                     `own tuning is NOT applied`;
        console.warn(presetNote);
      }
    }
    const qNum = (k, d) => (q.has(k) ? Number(q.get(k)) : d);
    const ui = {
      preset: startPreset,
      params: defaultParams(startPreset),
      radius: qNum('radius', 0.5),

      speedMax: qNum('speedmax', 3),
      timeScale: qNum('timescale', 1),
      paused: false,
      bodies: q.get('bodies') !== '0',
      reportAt: qNum('report', 3.0),

      force: q.get('force') !== '0',
      forceRadius: qNum('fradius', 0.12),
      forceStrength: qNum('fstrength', 30),
      forceLimit: qNum('flimit', 4),

      grabStrength: qNum('grabstrength', 10),

      grab: q.get('grab') === '1',

      pouring: false,
      pourLeft: 0,
      pourBudget: 0,
      pourSpeed: qNum('pourspeed', 3.0),
      pourWidth: qNum('pourwidth', 0.24),
      pourHeight: qNum('pourheight', 0.85),
      pourTilt: qNum('pourtilt', 0),
      pourExtruded: 0,
      pourNextLayer: 0,

      pourTrace: qNum('pourtrace', 0),

      boxScaleX: Math.min(2, Math.max(0.5, qNum('boxx', 1))),

      camSpeed: qNum('camspeed', 1.0),

      display: q.has('view')
        ? (q.get('view') === 'mesh' ? 1 : q.get('view') === 'ray' ? 2
           : q.get('view') === 'ssfr' ? 3 : 0) : 0,
      meshRes: qNum('meshres', 128),
      meshIso: qNum('meshiso', 0.4),

      fieldSmooth: qNum('fieldsmooth', 0),
      normalSmooth: qNum('normalsmooth', 2),
      raySurface: qNum('raysurface', 1),
    };
    mesh.normalSmoothPasses = ui.normalSmooth;
    ray.debug = qNum('raydebug', 0);
    ray.ior = qNum('ior', ray.ior);
    ray.absorption = qNum('absorption', ray.absorption);
    ray.thicknessSteps = qNum('thicksteps', ray.thicknessSteps);
    ray.groundReflection = qNum('groundrefl', ray.groundReflection);
    ray.roughness = qNum('roughness', ray.roughness);
    ray.sunIntensity = qNum('sunint', ray.sunIntensity);
    ray.sunElevation = qNum('sunelev', ray.sunElevation);
    ray.sunAzimuth = qNum('sunazim', ray.sunAzimuth);
    ray.exposure = qNum('exposure', ray.exposure);

    ssfr.debug = qNum('ssfrdebug', 0);
    ssfr.splatRadius = qNum('ssfrradius', ssfr.splatRadius);
    ssfr.renderScale = qNum('ssfrscale', ssfr.renderScale);
    ssfr.filter = qNum('ssfrfilter', ssfr.filter);
    ssfr.filterIterations = qNum('ssfriters', ssfr.filterIterations);
    ssfr.filterSigma = qNum('ssfrsigma', ssfr.filterSigma);
    ssfr.narrowDelta = qNum('ssfrdelta', ssfr.narrowDelta);
    ssfr.narrowMu = qNum('ssfrmu', ssfr.narrowMu);
    ssfr.thicknessRadius = qNum('ssfrthickr', ssfr.thicknessRadius);
    ssfr.thicknessScale = qNum('ssfrthick', ssfr.thicknessScale);
    ssfr.thicknessFilterSize = qNum('ssfrthickblur', ssfr.thicknessFilterSize);
    ssfr.depthCullFraction = qNum('ssfrdepthcull', ssfr.depthCullFraction);
    ssfr.cleanupPass = q.get('ssfrcleanup') !== '0';
    mesh.anisoRatio = qNum('anisoratio', mesh.anisoRatio);
    mesh.anisoRadiusScale = qNum('anisoradius', mesh.anisoRadiusScale);
    mesh.anisoLambda = qNum('anisolambda', mesh.anisoLambda);
    mesh.anisoMinNeighbours = qNum('anisonbrs', mesh.anisoMinNeighbours);
    mesh.anisoKs = qNum('anisoks', mesh.anisoKs);
    mesh.anisoStretch = qNum('anisostretch', mesh.anisoStretch);
    mesh.anisoLonely = qNum('anisolonely', mesh.anisoLonely);
    if (!ui.bodies) ui.params.bodies = [];

    ui.params.targetParticleCount = qNum('particles', ui.params.targetParticleCount);
    if (ui.bodies && q.get('body')) ui.params.bodies = q.get('body').split(',');

    ui.presetBodies = (q.get('body') ? q.get('body').split(',')
                                     : defaultParams(ui.preset).bodies).slice();
    ui.params.bodySize = qNum('bodysize', ui.params.bodySize);
    ui.params.substeps = qNum('substeps', ui.params.substeps);
    ui.params.iterations = qNum('iters', ui.params.iterations);
    ui.params.omega = qNum('omega', ui.params.omega);
    ui.params.xsphC = qNum('xsph', ui.params.xsphC);
    ui.params.sCorrK = qNum('scorr', ui.params.sCorrK);
    ui.params.spacing = qNum('spacing', ui.params.spacing);
    ui.params.gravity = qNum('gravity', ui.params.gravity);
    ui.params.surfaceTensionK = qNum('tension', ui.params.surfaceTensionK);
    ui.params.cfmEpsilonRel = qNum('cfm', ui.params.cfmEpsilonRel);
    ui.params.sCorrDq = qNum('scorrdq', ui.params.sCorrDq);
    ui.params.noBoundary = q.get('boundary') === '0';

    const qVec = (k, n) => {
      const v = q.get(k);
      if (!v) return null;
      const a = v.split(/[ ,]+/).map(Number);
      return a.length >= n && a.every(x => Number.isFinite(x)) ? a : null;
    };
    {
      const b = qVec('box', 3);
      if (b) ui.params.box = b.slice(0, 3);

      ui.boxBaseX = ui.params.box[0];
      if (ui.boxScaleX !== 1) ui.params.box[0] = ui.boxBaseX * ui.boxScaleX;
      const t = qVec('transmit', 3);
      if (t) { ray.transmit = t.slice(0, 3); ssfr.transmit = t.slice(0, 3); }
      ssfr.bilateralRange = qNum('ssfrrange', ssfr.bilateralRange);

      env.intensity = qNum('cubemapintensity', env.intensity);
      env.yaw = qNum('cubemapyaw', 0) * Math.PI / 180;
      if (q.has('floorplane')) {
        const on = q.get('floorplane') !== '0';
        env.floorPlane = on; ray.floorPlane = on; ssfr.floorPlane = on;
        env.floorPlaneExplicit = true;
      }
      const c = qVec('camera', 3);
      if (c) {
        cam.az = c[0]; cam.el = c[1]; cam.dist = c[2];
        if (c.length >= 6) cam.target = c.slice(3, 6);
      }
    }

    function applyScene() {
      sim.reset(ui.params);
      renderer.bindSim(sim);
      renderer.setBox(ui.params.box);
      solids.build(sim);

      ui.pourBudget = sim.pourBudget;
      ui.pourLeft = sim.pourBudget;
      ui.pourExtruded = 0;
      ui.pourNextLayer = 0;
      ui.pouring = false;
      syncPour();
      sceneLine();
    }

    function sceneLine() {
      const cap = blockCapacity(ui.params, ui.params.spacing);
      panel.scene.textContent =
        `${sim.n.toLocaleString()} particles ` +
        `(${sim.scene.nFluid.toLocaleString()} fluid + ${sim.scene.nBody.toLocaleString()} solid), ` +
        `${ui.params.box.map(v => v.toFixed(2)).join(' x ')} m, ` +
        `block holds ${cap.toLocaleString()}, ${sim.nBoundary.toLocaleString()} boundary samples`;
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.floor(canvas.clientWidth * dpr);
      const h = Math.floor(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        renderer.resize(w, h);
      }
    }

    const setPreset = name => {
      const here = new URLSearchParams(location.search);
      const next = new URLSearchParams({ preset: name });
      for (const k of ['spacing', 'bodies'])
        if (here.has(k)) next.set(k, here.get(k));

      if (qualityTouched) next.set('ssfrscale', String(ssfr.renderScale));
      location.search = '?' + next.toString();
    };

    for (const b of document.querySelectorAll('[data-preset]')) {
      b.classList.toggle('on', b.dataset.preset === ui.preset);
      b.onclick = () => setPreset(b.dataset.preset);
    }

    const setPaused = v => {
      ui.paused = v;
      document.getElementById('pause').textContent = v ? 'Resume' : 'Pause';
      const el = document.getElementById('boxx');
      el.disabled = v;
      el.title = v ? 'resume to move the wall' : '';
    };
    document.getElementById('pause').onclick = () => setPaused(!ui.paused);
    document.getElementById('reset').onclick = () => applyScene();

    function syncPour() {
      const b = document.getElementById('pour');
      const left = document.getElementById('pourleft');
      const rate = document.getElementById('pourrate');
      if (b) {
        b.textContent = ui.pouring ? 'Stop pouring' : 'Pour water';
        b.classList.toggle('on', ui.pouring);
        b.disabled = ui.pourLeft <= 0;
      }
      if (left) left.textContent = ui.pourBudget
        ? `${ui.pourLeft.toLocaleString()} of ${ui.pourBudget.toLocaleString()} left` : '';
      if (rate) {
        const r = pourRatePerSecond(ui, ui.params.spacing);
        rate.textContent = `${r.toLocaleString()} particles/s, ` +
          `${(ui.pourLeft / Math.max(1, r)).toFixed(1)} s to fill`;
      }
    }
    document.getElementById('pour').onclick = () => {
      ui.pouring = !ui.pouring && ui.pourLeft > 0;
      syncPour();
    };
    document.getElementById('resetcam').onclick = () => cam.frame(ui.params.box);

    {
      const note = document.getElementById('ininote');
      const file = document.getElementById('inifile');
      const takeIni = async (f) => {
        if (!f) return;
        const text = await f.text();

        const here = new URLSearchParams(location.search);
        const { query, applied, skipped, missingPanorama } = iniToQuery(text, {
          keep: { body: here.get('body'), bodysize: here.get('bodysize'),
                  bodies: here.get('bodies'), preset: here.get('preset') },
        });
        if (!applied.length) {
          note.style.display = 'block';
          note.textContent = `${f.name}: no settings recognised - is that a pbf_settings.ini?`;
          return;
        }
        const unknown = skipped.filter(k => !UNSUPPORTED.includes(k));
        console.info(`pbf_settings.ini: applied ${applied.length} keys` +
          (skipped.length ? `, skipped ${skipped.join(', ')}` : ''));
        if (missingPanorama)
          console.info(`pbf_settings.ini: the panorama has to be at env/${missingPanorama} ` +
                       `on this server - the file's own path is not reachable from a page`);
        sessionStorage.setItem('iniNote',
          `${f.name}: ${applied.length} settings applied` +
          (unknown.length ? `, ${unknown.length} unknown (${unknown.join(', ')})` : '') +
          (missingPanorama ? `. Panorama expected at env/${missingPanorama}` : ''));
        location.search = '?' + query;
      };
      document.getElementById('loadini').onclick = () => file.click();
      file.onchange = () => takeIni(file.files && file.files[0]);

      const stop = e => { e.preventDefault(); e.stopPropagation(); };
      canvas.addEventListener('dragover', stop);
      canvas.addEventListener('drop', e => {
        stop(e);
        takeIni(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
      });
      const carried = sessionStorage.getItem('iniNote');
      if (carried) sessionStorage.removeItem('iniNote');

      const lines = [carried, presetNote].filter(Boolean);
      if (lines.length) {
        note.style.display = 'block';
        note.textContent = lines.join(' · ');
      }
    }

    const surfaceGroup = document.getElementById('surfacegroup');
    const rayGroup = document.getElementById('raygroup');
    const normalRow = document.getElementById('normalsmooth').parentElement;
    const ssfrGroup = document.getElementById('ssfrgroup');
    const envGroup = document.getElementById('envgroup');
    function syncGroups() {
      surfaceGroup.style.display =
        (ui.display === 1 || ui.display === 2) ? '' : 'none';
      rayGroup.style.display = ui.display === 2 ? '' : 'none';
      ssfrGroup.style.display = ui.display === 3 ? '' : 'none';

      envGroup.style.display = (ui.display === 2 || ui.display === 3) ? '' : 'none';

      normalRow.style.display =
        (ui.display === 1 || (ui.display === 2 && ui.raySurface === 1)) ? '' : 'none';

      document.getElementById('ssfrscale').disabled = ui.display !== 3;
    }
    document.getElementById('viewmode').onchange = e => {
      ui.display = Number(e.target.value);
      syncGroups();
    };
    document.getElementById('viewmode').value = String(ui.display);
    const raySurfaceSel = document.getElementById('raysurface');
    raySurfaceSel.value = String(ui.raySurface);
    raySurfaceSel.onchange = e => { ui.raySurface = Number(e.target.value); syncGroups(); };
    const ssfrFilterSel = document.getElementById('ssfrfilter');
    ssfrFilterSel.value = String(ssfr.filter);
    ssfrFilterSel.onchange = e => { ssfr.filter = Number(e.target.value); };
    const ssfrDebugSel = document.getElementById('ssfrdebug');
    ssfrDebugSel.value = String(ssfr.debug);
    ssfrDebugSel.onchange = e => { ssfr.debug = Number(e.target.value); };
    const rayDebugSel = document.getElementById('raydebug');
    rayDebugSel.value = String(ray.debug);
    rayDebugSel.onchange = e => { ray.debug = Number(e.target.value); };

    const envInfo = document.getElementById('envinfo');
    const loadEnv = async (src) => {
      envInfo.textContent = 'loading…';
      try {
        envInfo.textContent = await env.load(src);

        if (!env.floorPlaneExplicit) { env.floorPlane = false; syncFloor(); }
      } catch (e) {
        env.clear();
        envInfo.textContent = 'cubemap: ' + (e.message || e);
        console.warn('cubemap:', e);
      }
    };
    document.getElementById('envfile').onchange = e => {
      const f = e.target.files && e.target.files[0];
      if (f) loadEnv(f);
    };
    document.getElementById('envclear').onclick = () => {
      env.clear();
      envInfo.textContent = '';
      document.getElementById('envfile').value = '';
    };
    const floorBtn = document.getElementById('floorplane');
    const syncFloor = () => {
      floorBtn.classList.toggle('on', env.floorPlane);
      ray.floorPlane = env.floorPlane;
      ssfr.floorPlane = env.floorPlane;
    };
    floorBtn.onclick = () => {
      env.floorPlane = !env.floorPlane;
      env.floorPlaneExplicit = true;
      syncFloor();
    };
    syncFloor();

    if (q.get('cubemap')) {
      loading.say('loading the environment…');
      const cap = new Promise(r => setTimeout(r, 10000));
      await Promise.race([loadEnv(q.get('cubemap')), cap]);
    }

    const uiEl = document.getElementById('ui');

    const debugPanel = document.getElementById('debugpanel');
    const showDebug = on => {
      debugPanel.hidden = !on;
      try { sessionStorage.setItem('debugopen', on ? '1' : ''); } catch {  }
    };
    let debugWanted = false;
    try { debugWanted = sessionStorage.getItem('debugopen') === '1'; } catch {  }
    showDebug(debugWanted);
    document.getElementById('debugclose').onclick = () => showDebug(false);
    {
      const gb = document.getElementById('grabmode');
      gb.classList.toggle('on', ui.grab);
      gb.onclick = () => {
        ui.grab = !ui.grab;
        gb.classList.toggle('on', ui.grab);

        if (!ui.grab) { dragBody = -1; sim.releaseBody(); }
      };
    }
    document.getElementById('nobodies').onclick = e => {
      ui.bodies = !ui.bodies;
      e.target.classList.toggle('on', ui.bodies);
      ui.params.bodies = ui.bodies ? ui.presetBodies.slice() : [];
      applyScene();
    };

    const slider = (id, get, set, fmt) => {
      const el = document.getElementById(id);
      const out = document.getElementById(id + 'v');

      el.value = String(get());
      out.textContent = fmt(get());
      el.oninput = () => { set(Number(el.value)); out.textContent = fmt(Number(el.value)); };
    };
    slider('pourspeed', () => ui.pourSpeed,
           v => { ui.pourSpeed = v; syncPour(); }, v => v.toFixed(2) + ' m/s');
    slider('pourwidth', () => ui.pourWidth,
           v => { ui.pourWidth = v; syncPour(); }, v => v.toFixed(3) + ' m');
    slider('pourheight', () => ui.pourHeight,
           v => { ui.pourHeight = v; syncPour(); }, v => v.toFixed(2));
    slider('pourtilt', () => ui.pourTilt,
           v => { ui.pourTilt = v; syncPour(); }, v => v.toFixed(0) + '°');
    slider('timescale', () => ui.timeScale, v => ui.timeScale = v, v => v.toFixed(2));
    slider('substeps', () => ui.params.substeps, v => ui.params.substeps = v, v => v);
    slider('iters', () => ui.params.iterations, v => ui.params.iterations = v, v => v);
    slider('omega', () => ui.params.omega, v => ui.params.omega = v, v => v.toFixed(2));
    slider('xsph', () => ui.params.xsphC, v => ui.params.xsphC = v, v => v.toFixed(3));
    slider('scorr', () => ui.params.sCorrK, v => ui.params.sCorrK = v, v => v.toFixed(3));
    slider('fradius', () => ui.forceRadius, v => ui.forceRadius = v, v => v.toFixed(2));
    slider('fstrength', () => ui.forceStrength, v => ui.forceStrength = v, v => v.toFixed(0));
    slider('flimit', () => ui.forceLimit, v => ui.forceLimit = v, v => v.toFixed(1));
    slider('grab', () => ui.grabStrength, v => ui.grabStrength = v, v => v.toFixed(0));

    slider('boxx', () => ui.boxScaleX, v => ui.boxScaleX = v,
           v => `${(ui.boxBaseX * v).toFixed(2)} m`);
    slider('camspeed', () => ui.camSpeed, v => ui.camSpeed = v, v => v.toFixed(2));
    slider('tension', () => ui.params.surfaceTensionK,
           v => ui.params.surfaceTensionK = v, v => v.toFixed(2));
    slider('radius', () => ui.radius, v => ui.radius = v, v => v.toFixed(2));
    slider('meshres', () => ui.meshRes, v => ui.meshRes = v, v => v);
    slider('normalsmooth', () => ui.normalSmooth,
           v => { ui.normalSmooth = v; mesh.normalSmoothPasses = v; }, v => v);
    slider('fieldsmooth', () => ui.fieldSmooth, v => ui.fieldSmooth = v, v => v);
    slider('meshiso', () => ui.meshIso, v => ui.meshIso = v, v => v.toFixed(2));
    slider('ior', () => ray.ior, v => ray.ior = v, v => v.toFixed(3));
    slider('absorption', () => ray.absorption, v => ray.absorption = v, v => v.toFixed(2));
    slider('thicksteps', () => ray.thicknessSteps, v => ray.thicknessSteps = v, v => v);
    slider('groundrefl', () => ray.groundReflection,
           v => ray.groundReflection = v, v => v.toFixed(2));
    slider('roughness', () => ray.roughness, v => ray.roughness = v, v => v.toFixed(3));
    slider('sunint', () => ray.sunIntensity, v => ray.sunIntensity = v, v => v.toFixed(1));
    slider('sunelev', () => ray.sunElevation, v => ray.sunElevation = v, v => v.toFixed(0));
    slider('sunazim', () => ray.sunAzimuth, v => ray.sunAzimuth = v, v => v.toFixed(0));
    slider('exposure', () => ray.exposure, v => ray.exposure = v, v => v.toFixed(2));
    slider('envintensity', () => env.intensity, v => env.intensity = v, v => v.toFixed(2));
    slider('envyaw', () => env.yaw * 180 / Math.PI,
           v => env.yaw = v * Math.PI / 180, v => v.toFixed(0));
    slider('ssfrradius', () => ssfr.splatRadius, v => ssfr.splatRadius = v, v => v.toFixed(2));
    slider('ssfrscale', () => ssfr.renderScale,
           v => { ssfr.renderScale = v; qualityTouched = true; },
           v => v.toFixed(2) + ' x');
    slider('anisoratio', () => mesh.anisoRatio, v => mesh.anisoRatio = v, v => v.toFixed(2));
    slider('ssfriters', () => ssfr.filterIterations,
           v => ssfr.filterIterations = v, v => v);
    slider('ssfrsigma', () => ssfr.filterSigma, v => ssfr.filterSigma = v, v => v.toFixed(2));
    slider('ssfrdelta', () => ssfr.narrowDelta, v => ssfr.narrowDelta = v, v => v.toFixed(1));
    slider('ssfrmu', () => ssfr.narrowMu, v => ssfr.narrowMu = v, v => v.toFixed(2));
    slider('ssfrthickr', () => ssfr.thicknessRadius,
           v => ssfr.thicknessRadius = v, v => v.toFixed(2));
    slider('ssfrthick', () => ssfr.thicknessScale,
           v => ssfr.thicknessScale = v, v => v.toFixed(2));
    slider('ssfrthickblur', () => ssfr.thicknessFilterSize,
           v => ssfr.thicknessFilterSize = v, v => v);
    slider('ssfrdepthcull', () => ssfr.depthCullFraction,
           v => ssfr.depthCullFraction = v, v => v.toFixed(2));
    syncGroups();

    let rotating = false, panning = false;
    let lastX = 0, lastY = 0, mouseInit = false;
    let pendingImpulse = [0, 0, 0];
    let forceOrigin = [0, 0, 0], forceDir = [0, 0, -1], forcePending = false;
    canvas.oncontextmenu = e => e.preventDefault();

    let dragBody = -1, dragDistance = 0, dragOffset = [0, 0, 0];
    canvas.onpointerdown = e => {
      if (e.button === 0 && ui.grab) {
        const r = canvas.getBoundingClientRect();
        const aspect = r.width / Math.max(1, r.height);
        const ray = screenRay(cam, (e.clientX - r.left) / r.width,
                              (e.clientY - r.top) / r.height, aspect);
        const p = pickBody(sim, ray.origin, ray.dir);
        if (p.body > 0) {
          dragBody = p.body;
          dragDistance = p.t;
          const c = sim.bodyPose[p.body - 1].centre;
          dragOffset = [p.hit[0] - c[0], p.hit[1] - c[1], p.hit[2] - c[2]];
          rotating = false;
          lastX = e.clientX; lastY = e.clientY; mouseInit = true;
          canvas.setPointerCapture(e.pointerId);
          return;
        }
      }
      if (e.button === 0) rotating = true;
      if (e.button === 2 || e.button === 1) panning = true;
      lastX = e.clientX; lastY = e.clientY; mouseInit = true;
      canvas.setPointerCapture(e.pointerId);
    };
    canvas.onpointerup = e => {
      if (e.button === 0) { dragBody = -1; sim.releaseBody(); }
      if (e.button === 0) rotating = false;
      if (e.button === 2 || e.button === 1) panning = false;
      canvas.releasePointerCapture(e.pointerId);
    };
    canvas.onpointermove = e => {
      if (!mouseInit) { lastX = e.clientX; lastY = e.clientY; mouseInit = true; return; }
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      const oldX = lastX, oldY = lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (rotating) cam.orbit(-dx * 0.008 * ui.camSpeed, dy * 0.008 * ui.camSpeed);
      if (panning) cam.pan(dx * 0.0012 * ui.camSpeed, dy * 0.0012 * ui.camSpeed);

      if (ui.force && !rotating && !panning && dragBody < 0 &&
          (dx !== 0 || dy !== 0)) {
        const r = canvas.getBoundingClientRect();
        const aspect = r.width / Math.max(1, r.height);
        const a = screenRay(cam, (oldX - r.left) / r.width, (oldY - r.top) / r.height, aspect);
        const b = screenRay(cam, (e.clientX - r.left) / r.width,
                            (e.clientY - r.top) / r.height, aspect);
        forceOrigin = b.origin;
        forceDir = b.dir;
        for (let k = 0; k < 3; k++)
          pendingImpulse[k] += ui.forceStrength * (b.dir[k] - a.dir[k]);
        forcePending = true;
      }
    };
    canvas.onwheel = e => { e.preventDefault(); cam.zoom(e.deltaY > 0 ? 1.1 : 0.9); };
    ui.consumeImpulse = () => {
      if (!forcePending) return;
      sim.applyRayImpulse(forceOrigin, forceDir, pendingImpulse,
                          ui.forceRadius, ui.forceLimit);
      pendingImpulse = [0, 0, 0];
      forcePending = false;
    };

    ui.holdFrame = () => {
      if (dragBody < 1 || ui.paused) return;
      if (dragBody > sim.bodies.length) return;
      const r = canvas.getBoundingClientRect();
      const aspect = r.width / Math.max(1, r.height);
      const ray = screenRay(cam, (lastX - r.left) / r.width,
                            (lastY - r.top) / r.height, aspect);
      const want = [ray.origin[0] + ray.dir[0] * dragDistance - dragOffset[0],
                    ray.origin[1] + ray.dir[1] * dragDistance - dragOffset[1],
                    ray.origin[2] + ray.dir[2] * dragDistance - dragOffset[2]];
      const held = sim.bodies[dragBody - 1];
      const target = dragTarget(want, held, sim.bodyPose[dragBody - 1],
                                ui.params.box);
      sim.holdBody(dragBody, target, ui.grabStrength,
                   Math.max(ui.forceLimit, 0.1));
    };
    window.__grabbing = () => dragBody;
    window.addEventListener('keydown', e => {
      if (e.code === 'Space') { document.getElementById('pause').click(); e.preventDefault(); }
      if (e.code === 'KeyH') { uiEl.classList.toggle('hidden'); }
      if (e.code === 'KeyD') { showDebug(debugPanel.hidden); }
    });

    window.__mesh = mesh;
    window.__ssfr = ssfr;

    window.__sim = sim;

    window.__readBuf = async (buf, bytes) => {
      const rb = device.createBuffer({ size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(buf, 0, rb, 0, bytes);
      device.queue.submit([e.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap(); rb.destroy();
      return Array.from(out);
    };
    window.__cam = cam;
    window.__box = ui.params.box;
    window.__ui = ui;
    window.__screenRay = (u, v, a) => screenRay(cam, u, v, a);

    window.__project = (pt) => {
      const r = canvas.getBoundingClientRect();
      const p2 = renderer.project(pt, cam, r.width / Math.max(1, r.height));
      return p2 ? { x: r.left + p2[0] * r.width, y: r.top + p2[1] * r.height } : null;
    };

    loading.say('building the scene…');
    applyScene();
    resize();

    if (q.get('verify') === '1') {
      sim.step(1 / 60);
      const dev2 = device;
      const read = async (buf, bytes) => {
        const rb = dev2.createBuffer({ size: bytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const e = dev2.createCommandEncoder();
        e.copyBufferToBuffer(buf, 0, rb, 0, bytes);
        dev2.queue.submit([e.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        const out = rb.getMappedRange().slice(0);
        rb.unmap(); rb.destroy();
        return out;
      };

      const nCells = sim.nCells, n = sim.n;
      const dbg = {};
      const cs = new Uint32Array(await read(sim.buf.cellStart, (nCells + 1) * 4));
      const pred = new Float32Array(await read(sim.buf[sim.parity ? 'predB' : 'predA'], n * 16));
      const dens = new Float32Array(await read(sim.buf.density, n * 4));

      let bad = 0, prev = 0;
      for (let i = 0; i <= nCells; i++) { if (cs[i] < prev) bad++; prev = cs[i]; }
      dbg.scanMonotonic = bad === 0;
      dbg.scanTotal = cs[nCells];
      dbg.n = n;

      const h = sim.h, h2 = h * h, coef = 315 / (64 * Math.PI * Math.pow(h, 9));
      const gd = sim.gridDim;
      const cellIdx = (x, y, z) => {
        const c = [Math.min(gd[0] - 1, Math.max(0, Math.floor(x / h))),
                   Math.min(gd[1] - 1, Math.max(0, Math.floor(y / h))),
                   Math.min(gd[2] - 1, Math.max(0, Math.floor(z / h)))];
        return (c[2] * gd[1] + c[1]) * gd[0] + c[0];
      };
      const buckets = new Map();
      for (let i = 0; i < n; i++) {
        const k = cellIdx(pred[i * 4], pred[i * 4 + 1], pred[i * 4 + 2]);
        let a = buckets.get(k); if (!a) { a = []; buckets.set(k, a); }
        a.push(i);
      }
      let worst = 0, worstI = -1, nbMax = 0;
      for (let s = 0; s < 200; s++) {
        const i = Math.floor(s * n / 200);
        const px = pred[i * 4], py = pred[i * 4 + 1], pz = pred[i * 4 + 2];
        let rho = sim.scene.mass * coef * h2 * h2 * h2, nb = 0;
        const c = [Math.floor(px / h), Math.floor(py / h), Math.floor(pz / h)];
        for (let dz = -1; dz <= 1; dz++)
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const k = (Math.min(gd[2] - 1, Math.max(0, c[2] + dz)) * gd[1] +
                         Math.min(gd[1] - 1, Math.max(0, c[1] + dy))) * gd[0] +
                         Math.min(gd[0] - 1, Math.max(0, c[0] + dx));
              const a = buckets.get(k); if (!a) continue;
              for (const j of a) {
                if (j === i) continue;
                const r2 = (px - pred[j * 4]) ** 2 + (py - pred[j * 4 + 1]) ** 2 +
                           (pz - pred[j * 4 + 2]) ** 2;
                if (r2 >= h2) continue;
                const t = h2 - r2;
                rho += sim.scene.mass * coef * t * t * t;
                nb++;
              }
            }
        nbMax = Math.max(nbMax, nb);
        const rel = Math.abs(dens[i] - rho) / Math.max(1, rho);
        if (rel > worst) { worst = rel; worstI = i; }
      }

      let zeros = 0, lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
      for (let i = 0; i < n; i++) {
        const x = pred[i * 4], y = pred[i * 4 + 1], z = pred[i * 4 + 2];
        if (Math.abs(x) + Math.abs(y) + Math.abs(z) < 1e-6) zeros++;
        lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
        lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
        lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z);
      }
      const lam = new Float32Array(await read(sim.buf.lambda, n * 4));
      let lmin = 1e30, lmax = -1e30, dmin = 1e30, dmax = -1e30, nan = 0;
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(lam[i]) || !Number.isFinite(dens[i])) nan++;
        lmin = Math.min(lmin, lam[i]); lmax = Math.max(lmax, lam[i]);
        dmin = Math.min(dmin, dens[i]); dmax = Math.max(dmax, dens[i]);
      }
      dbg.lambda = `${lmin.toExponential(2)}..${lmax.toExponential(2)}`;
      dbg.density = `${dmin.toFixed(1)}..${dmax.toFixed(1)}`;
      dbg.nonFinite = nan;
      {
        const sp = sim.scene.pos;
        let a = [1e9, 1e9, 1e9], b2 = [-1e9, -1e9, -1e9];
        for (let i = 0; i < sim.scene.n; i++)
          for (let k = 0; k < 3; k++) {
            a[k] = Math.min(a[k], sp[i * 4 + k]); b2[k] = Math.max(b2[k], sp[i * 4 + k]);
          }
        dbg.initialBox = a.map((v, k) => `${v.toFixed(3)}..${b2[k].toFixed(3)}`).join(' ');
      }
      dbg.predZeros = zeros;
      dbg.predBox = lo.map((v, k) => `${v.toFixed(3)}..${hi[k].toFixed(3)}`).join(' ');
      dbg.worstDensityRelError = Number(worst.toFixed(4));
      dbg.worstAt = worstI;
      dbg.gpuDensity = Number(dens[worstI].toFixed(1));
      dbg.cpuNeighboursMax = nbMax;
      window.__result = dbg;
      window.__done = true;
      console.info('verify', JSON.stringify(dbg));
      setPaused(true);
    }

    let last = performance.now();
    let fpsAcc = 0, fpsN = 0, shown = 0;
    let frames = 0;
    function frame(now) {
      resize();

      const dt = Math.max(0, Math.min((now - last) / 1000, 0.05));
      last = now;
      fpsAcc += dt; fpsN++;
      if (fpsAcc > 0.25) { shown = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }

      ssfr.ior = ray.ior;
      ssfr.absorption = ray.absorption;
      ssfr.transmit = ray.transmit;
      ssfr.roughness = ray.roughness;
      ssfr.sunIntensity = ray.sunIntensity;
      ssfr.sunElevation = ray.sunElevation;
      ssfr.sunAzimuth = ray.sunAzimuth;
      ssfr.exposure = ray.exposure;
      ssfr.groundReflection = ray.groundReflection;

      {
        const R = mesh.anisoRadiusScale * sim.h;
        const full = 4.18879 * Math.pow(R / Math.max(ui.params.spacing, 1e-6), 3);
        ssfr.depthCullNeighbours = ssfr.depthCullFraction > 0
          ? Math.floor(ssfr.depthCullFraction * full) : 0;
      }
      ui.consumeImpulse();
      ui.holdFrame();

      if (!ui.paused) {
        const want = ui.boxBaseX * ui.boxScaleX;
        const gap = want - ui.params.box[0];

        const step = Math.sign(gap) * Math.min(Math.abs(gap), 2 * dt);
        if (Math.abs(step) > 1e-6) {
          sim.resizeBox([ui.params.box[0] + step, ui.params.box[1], ui.params.box[2]]);
          renderer.setBox(ui.params.box);

          sceneLine();
        }
      }

      if (!ui.paused) {
        if (ui.pouring && ui.pourLeft > 0) {
          const sub = 1 / 60 / Math.max(1, ui.params.substeps);
          let want = dt * ui.timeScale;
          let poured = 0;
          for (let guard = 0; want > 1e-6 && guard < 64; ++guard) {
            const slice = Math.min(want, sub);
            sim.step(slice);
            poured += pourStep(sim, ui, sim.lastAdvanced || 0);
            want -= slice;
          }
          if (poured > 0) { syncPour(); sceneLine(); }
        } else {
          sim.step(dt * ui.timeScale);
        }
      }
      renderer.draw(sim, cam, {
        radius: ui.radius * ui.params.spacing,
        speedMax: ui.speedMax,
        display: ui.display,
        mesh: mesh,
        solids: solids,
        ray: ray,
        ssfr: ssfr,
        raySurface: ui.raySurface,
        meshRes: ui.meshRes,
        meshIso: ui.meshIso,
        fieldSmooth: ui.fieldSmooth,
      });

      loading.done();
      if (ui.display !== 0 && mesh.vertDim) {
        const info = document.getElementById('meshinfo');
        const tris = mesh.lastTriangles;
        const over = tris > mesh.vertBudget;
        info.textContent =
          `field ${mesh.vertDim.join('x')}, voxel ${(mesh.voxel * 1000).toFixed(1)} mm\n` +
          `mesh tris ${(tris * 1e-6).toFixed(2)}M / ${(mesh.vertBudget * 1e-6).toFixed(2)}M` +
          (over ? ' OVER - the mesh is cut open, lower field res' : '');
        info.style.color = over ? '#ff5a3c' : '';
        info.style.whiteSpace = 'pre';
      }
      const s = sim.stats;
      window.__lastStats = s;
      panel.stats.textContent =
        `${shown.toFixed(1)} fps   t = ${sim.simTime.toFixed(2)} s\n` +
        `density avg ${s.avgRho.toFixed(3)}  max ${s.maxRho.toFixed(3)} (x rest)\n` +
        `max speed ${s.maxSpeed.toFixed(2)} m/s   KE ${(s.ke ?? 0).toFixed(1)} J\n` +
        `${ui.params.substeps} substeps x ${ui.params.iterations} iters`;

      if (panel.timing) {
        panel.timing.textContent = renderTimer.on
          ? `sim    ${simTimer.frameMs.toFixed(2)} ms  ${simTimer.line()}\n` +
            `render ${renderTimer.frameMs.toFixed(2)} ms  ${renderTimer.line()}`
          : 'GPU timing: this browser has no timestamp-query';
      }
      frames++;

      if (sim.simTime > ui.reportAt && !window.__done) {
        window.__result = {
          adapter: `${info.vendor} ${info.architecture}`,
          particles: sim.n, fluid: sim.scene.nFluid, solid: sim.scene.nBody,
          boundary: sim.nBoundary, bodies: sim.nBodies,
          fps: Number(shown.toFixed(1)),
          avgRho: Number(s.avgRho.toFixed(4)),
          maxRho: Number(s.maxRho.toFixed(3)),
          maxSpeed: Number(s.maxSpeed.toFixed(3)),
          simTime: Number(sim.simTime.toFixed(2)),
          gpuError: window.__gpuError || null,

          timing: renderTimer.on ? {
            frames: renderTimer.frames,
            simMs: Number(simTimer.avgFrame().toFixed(3)),
            simPasses: simTimer.table(),
            renderMs: Number(renderTimer.avgFrame().toFixed(3)),
            renderPasses: renderTimer.table(),
          } : null,
        };
        if (renderTimer.on) {
          console.info(`SIM PASSES (avg, ms): total ${simTimer.avgFrame().toFixed(2)}  ` +
                       simTimer.line(true));
          console.info(`RENDER PASSES (avg, ms): total ${renderTimer.avgFrame().toFixed(2)}  ` +
                       renderTimer.line(true));
        }
        window.__done = true;
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
}

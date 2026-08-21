

const KEYS = {
  substeps: 'substeps', iterations: 'iters', cfm: 'cfm', scorrk: 'scorr',
  scorrdq: 'scorrdq', xsph: 'xsph', tension: 'tension', gravity: 'gravity',
  omega: 'omega', particles: 'particles', spacing: 'spacing',
  meshres: 'meshres', meshiso: 'meshiso', fieldsmooth: 'fieldsmooth',
  normalsmooth: 'normalsmooth', raysurface: 'raysurface',
  anisoratio: 'anisoratio', anisolambda: 'anisolambda', anisonbrs: 'anisonbrs',
  anisolonely: 'anisolonely', anisoradius: 'anisoradius',
  anisostretch: 'anisostretch', anisoks: 'anisoks',
  ior: 'ior', absorption: 'absorption', thicksteps: 'thicksteps',
  groundreflection: 'groundrefl', roughness: 'roughness',
  sunintensity: 'sunint', sunelev: 'sunelev', sunazim: 'sunazim',
  exposure: 'exposure',
  ssfrradius: 'ssfrradius', ssfrfilter: 'ssfrfilter', ssfriters: 'ssfriters',
  ssfrsigma: 'ssfrsigma', ssfrdelta: 'ssfrdelta', ssfrmu: 'ssfrmu',
  ssfrrange: 'ssfrrange', ssfrcleanup: 'ssfrcleanup', ssfrthick: 'ssfrthick',
  ssfrthickr: 'ssfrthickr', ssfrthickblur: 'ssfrthickblur',
  ssfrdepthcull: 'ssfrdepthcull', ssfrscale: 'ssfrscale',
  renderradius: 'radius', timescale: 'timescale', colorspeedmax: 'speedmax',
  hover: 'force', hoverradius: 'fradius', hoverstrength: 'fstrength',
  hoverlimit: 'flimit', grabstrength: 'grabstrength',

  pourspeed: 'pourspeed', pourwidth: 'pourwidth', pourheight: 'pourheight',
  pourtilt: 'pourtilt',
  floorplane: 'floorplane', cubemapintensity: 'cubemapintensity',
  cubemapyaw: 'cubemapyaw',

  box: 'box', transmit: 'transmit', camera: 'camera',

  bodysize: 'bodysize',
};

export const UNSUPPORTED = [
  'showkernels', 'kernelscale',
  'soraverage',
  'fieldfilter',
  'ssfrstretch', 'ssfrstretchmax', 'ssfrstretchthin', 'ssfrstretchreach',
  'ssfrhalfthick',
  'weburl', 'webhighperf',
  'webpresets',
];

export function parseIni(text) {
  const kv = {};
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line.startsWith(';')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    kv[line.slice(0, eq).trim().toLowerCase()] = line.slice(eq + 1).trim();
  }
  return kv;
}

const VIEW = { 0: 'particles', 1: 'mesh', 2: 'mesh', 3: 'ray', 4: 'ssfr' };

export function iniToQuery(text, opts = {}) {
  const kv = parseIni(text);
  const sp = new URLSearchParams();
  const applied = [], skipped = [];
  for (const [k, v] of Object.entries(kv)) {
    const q = KEYS[k];
    if (q) { sp.set(q, v); applied.push(k); continue; }
    if (k === 'display') {
      const view = VIEW[String(parseInt(v, 10))];
      if (view) { sp.set('view', view); applied.push(k); } else skipped.push(k);
      continue;
    }
    if (k === 'cubemap') continue;
    if (k === 'body') {
      if (v.trim()) sp.set('body', v.trim()); else sp.set('bodies', '0');
      applied.push(k);
      continue;
    }
    skipped.push(k);
  }

  let missingPanorama = null;
  if (kv.cubemap) {
    const name = kv.cubemap.replace(/^"|"$/g, '').split(/[\\/]/).pop();
    if (name) {
      sp.set('cubemap', 'env/' + name);
      applied.push('cubemap');
      missingPanorama = name;
    }
  }

  for (const [k, v] of Object.entries(opts.keep || {}))
    if (v != null && !sp.has(k)) sp.set(k, v);
  return { query: sp.toString(), applied, skipped, missingPanorama };
}

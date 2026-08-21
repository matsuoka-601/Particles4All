

const kPi = Math.PI;

function decodeHDR(bytes) {
  const u8 = new Uint8Array(bytes);
  let p = 0;
  const line = () => {
    let s = '';
    while (p < u8.length && u8[p] !== 10) s += String.fromCharCode(u8[p++]);
    p++;
    return s;
  };
  const magic = line();
  if (!magic.startsWith('#?')) throw new Error('not a Radiance .hdr file');
  let l;
  while ((l = line()) !== '') {
    if (l.startsWith('FORMAT=') && !l.includes('32-bit_rle_rgbe'))
      throw new Error('unsupported .hdr format: ' + l);
  }
  const dim = line().trim().split(/\s+/);

  if (dim.length !== 4 || dim[0] !== '-Y' || dim[2] !== '+X')
    throw new Error('unsupported .hdr resolution line: ' + dim.join(' '));
  const h = parseInt(dim[1], 10), w = parseInt(dim[3], 10);
  const out = new Float32Array(w * h * 3);
  const rgbe = new Uint8Array(w * 4);
  const put = (y) => {
    for (let x = 0; x < w; x++) {
      const e = rgbe[x * 4 + 3];
      const f = e ? Math.pow(2, e - 136) : 0;
      const o = (y * w + x) * 3;
      out[o + 0] = rgbe[x * 4 + 0] * f;
      out[o + 1] = rgbe[x * 4 + 1] * f;
      out[o + 2] = rgbe[x * 4 + 2] * f;
    }
  };
  for (let y = 0; y < h; y++) {
    if (p + 4 > u8.length) throw new Error('.hdr ended early');
    const r = u8[p], g = u8[p + 1], b = u8[p + 2], e = u8[p + 3];
    if (r === 2 && g === 2 && ((b << 8) | e) === w && w >= 8 && w < 32768) {
      p += 4;
      for (let c = 0; c < 4; c++) {
        let x = 0;
        while (x < w) {
          let n = u8[p++];
          if (n > 128) {
            n -= 128;
            const v = u8[p++];
            while (n-- > 0) rgbe[(x++) * 4 + c] = v;
          } else {
            while (n-- > 0) rgbe[(x++) * 4 + c] = u8[p++];
          }
        }
      }
    } else {
      let x = 0, shift = 0;
      while (x < w) {
        const rr = u8[p], gg = u8[p + 1], bb = u8[p + 2], ee = u8[p + 3];
        p += 4;
        if (rr === 1 && gg === 1 && bb === 1) {
          const n = ee << (shift * 8);
          const prev = (x - 1) * 4;
          for (let k = 0; k < n; k++, x++)
            for (let c = 0; c < 4; c++) rgbe[x * 4 + c] = rgbe[prev + c];
          shift++;
        } else {
          rgbe[x * 4 + 0] = rr; rgbe[x * 4 + 1] = gg;
          rgbe[x * 4 + 2] = bb; rgbe[x * 4 + 3] = ee;
          x++; shift = 0;
        }
      }
    }
    put(y);
  }
  return { w, h, data: out };
}

async function decodeImage(blob) {
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const out = new Float32Array(bmp.width * bmp.height * 3);
  for (let i = 0, n = bmp.width * bmp.height; i < n; i++) {
    out[i * 3 + 0] = px[i * 4 + 0] / 255;
    out[i * 3 + 1] = px[i * 4 + 1] / 255;
    out[i * 3 + 2] = px[i * 4 + 2] / 255;
  }
  bmp.close();
  return { w: bmp.width, h: bmp.height, data: out };
}

function sampleEquirect(src, w, h, dx, dy, dz, out, o) {
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  dx /= len; dy /= len; dz /= len;
  const u = 0.5 + Math.atan2(dx, dz) / (2 * kPi);
  const v = Math.acos(Math.min(1, Math.max(-1, dy))) / kPi;
  const fx = u * w - 0.5, fy = v * h - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const tx = fx - x0, ty = fy - y0;
  const at = (x, y, c) => {
    x = ((x % w) + w) % w;
    y = Math.min(h - 1, Math.max(0, y));
    return src[(y * w + x) * 3 + c];
  };
  for (let c = 0; c < 3; c++) {
    const a = at(x0, y0, c) * (1 - tx) + at(x0 + 1, y0, c) * tx;
    const b = at(x0, y0 + 1, c) * (1 - tx) + at(x0 + 1, y0 + 1, c) * tx;
    out[o + c] = a * (1 - ty) + b * ty;
  }
}

function faceDir(face, s, t, d) {
  switch (face) {
    case 0: d[0] =  1;  d[1] = -t; d[2] = -s; break;
    case 1: d[0] = -1;  d[1] = -t; d[2] =  s; break;
    case 2: d[0] =  s;  d[1] =  1; d[2] =  t; break;
    case 3: d[0] =  s;  d[1] = -1; d[2] = -t; break;
    case 4: d[0] =  s;  d[1] = -t; d[2] =  1; break;
    default: d[0] = -s; d[1] = -t; d[2] = -1; break;
  }
}

function toHalf(f32, n) {
  const out = new Uint16Array(n * 4);
  const buf = new DataView(new ArrayBuffer(4));
  const half = (v) => {
    buf.setFloat32(0, v);
    const x = buf.getUint32(0);
    const s = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 112;
    let m = x & 0x7fffff;
    if (e <= 0) {
      if (e < -10) return s;
      m = (m | 0x800000) >>> (1 - e);
      return s | (m + 0x1000 >>> 13);
    }
    if (e >= 0x1f) return s | 0x7c00;
    return s | (e << 10) | ((m + 0x1000) >>> 13);
  };
  for (let i = 0; i < n; i++) {
    out[i * 4 + 0] = half(f32[i * 3 + 0]);
    out[i * 4 + 1] = half(f32[i * 3 + 1]);
    out[i * 4 + 2] = half(f32[i * 3 + 2]);
    out[i * 4 + 3] = half(1);
  }
  return out;
}

export class Environment {
  constructor(device) {
    this.dev = device;
    this.intensity = 1.0;
    this.yaw = 0.0;
    this.floorPlane = true;

    this.floorPlaneExplicit = false;
    this.has = false;
    this.status = '';
    this.gen = 0;

    this.texture = device.createTexture({
      size: [1, 1, 6], format: 'rgba16float', label: 'envCubeEmpty',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.view = this.texture.createView({ dimension: 'cube' });
    this.sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  }

  async load(src) {
    let img, name;
    if (src instanceof Blob) {
      name = src.name || 'file';
      const buf = await src.arrayBuffer();
      img = /\.hdr$/i.test(name) ? decodeHDR(buf) : await decodeImage(src);
    } else {
      name = String(src);
      const res = await fetch(name);
      if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`);
      if (/\.hdr$/i.test(name)) img = decodeHDR(await res.arrayBuffer());
      else img = await decodeImage(await res.blob());
    }
    const { w, h, data } = img;
    if (w < 2 * h - 2 || w > 2 * h + 2)
      throw new Error(`${name} is ${w}x${h} - an equirectangular panorama is 2:1`);

    const size = Math.min(2048, Math.max(16, Math.floor(w / 4)));
    const levels = Math.floor(Math.log2(size)) + 1;
    this.texture?.destroy();
    this.texture = this.dev.createTexture({
      size: [size, size, 6], format: 'rgba16float', mipLevelCount: levels,
      label: 'envCube',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const face = new Float32Array(size * size * 3);
    const d = [0, 0, 0];
    for (let f = 0; f < 6; f++) {
      for (let y = 0; y < size; y++) {
        const t = 2 * (y + 0.5) / size - 1;
        for (let x = 0; x < size; x++) {
          const s = 2 * (x + 0.5) / size - 1;
          faceDir(f, s, t, d);
          sampleEquirect(data, w, h, d[0], d[1], d[2], face, (y * size + x) * 3);
        }
      }

      let lvl = face, lw = size;
      for (let m = 0; m < levels; m++) {
        this.dev.queue.writeTexture(
          { texture: this.texture, mipLevel: m, origin: [0, 0, f] },
          toHalf(lvl, lw * lw), { bytesPerRow: lw * 8, rowsPerImage: lw },
          [lw, lw, 1]);
        if (m + 1 === levels) break;
        const nw = Math.max(1, lw >> 1);
        const next = new Float32Array(nw * nw * 3);
        for (let y = 0; y < nw; y++)
          for (let x = 0; x < nw; x++)
            for (let c = 0; c < 3; c++) {
              const x0 = Math.min(lw - 1, x * 2), x1 = Math.min(lw - 1, x * 2 + 1);
              const y0 = Math.min(lw - 1, y * 2), y1 = Math.min(lw - 1, y * 2 + 1);
              next[(y * nw + x) * 3 + c] = 0.25 * (
                lvl[(y0 * lw + x0) * 3 + c] + lvl[(y0 * lw + x1) * 3 + c] +
                lvl[(y1 * lw + x0) * 3 + c] + lvl[(y1 * lw + x1) * 3 + c]);
            }
        lvl = next; lw = nw;
      }
    }
    this.view = this.texture.createView({ dimension: 'cube' });
    this.has = true;
    this.gen++;
    this.status = `equirectangular ${w}x${h} from ${name}`;
    return this.status;
  }

  clear() {
    this.has = false;
    this.status = '';
    this.gen++;
  }
}

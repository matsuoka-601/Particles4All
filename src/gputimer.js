

export class GpuTimer {
  constructor(device, slots = 32, label = 'timer') {
    this.dev = device;
    this.on = !!device.features?.has('timestamp-query');
    this.slots = slots;
    this.entries = [];
    this.used = 0;
    this.last = {};
    this.sum = {};
    this.order = [];
    this.frameMs = 0;
    this.sumFrameMs = 0;
    this.frames = 0;
    if (!this.on) return;
    const bytes = slots * 2 * 8;
    this.qs = device.createQuerySet({ type: 'timestamp', count: slots * 2, label });
    this.resolveBuf = device.createBuffer({ size: bytes, label: label + 'Resolve',
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });

    this.read = [0, 1, 2].map(i => device.createBuffer({ size: bytes,
      label: `${label}Read${i}`,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    this.busy = [false, false, false];
    this.ring = 0;
  }

  begin() { this.entries = []; this.used = 0; }

  mark(name) {
    if (!this.on || this.used >= this.slots) return undefined;
    const slot = this.used++;
    this.entries.push({ name, slot });
    return { timestampWrites: { querySet: this.qs,
      beginningOfPassWriteIndex: slot * 2, endOfPassWriteIndex: slot * 2 + 1 } };
  }

  gap(name) { if (this.on) this.entries.push({ name, gap: true }); }

  resolve(enc) {
    if (!this.on || this.used === 0) return;
    const idx = this.ring;
    if (this.busy[idx]) return;
    const count = this.used * 2;
    enc.resolveQuerySet(this.qs, 0, count, this.resolveBuf, 0);
    enc.copyBufferToBuffer(this.resolveBuf, 0, this.read[idx], 0, count * 8);
    this.busy[idx] = true;
    this.ring = (this.ring + 1) % this.read.length;
    this.pending = { idx, count, entries: this.entries };
  }

  poll() {
    const p = this.pending;
    if (!p) return;
    this.pending = null;
    const buf = this.read[p.idx];
    const bytes = p.count * 8;
    buf.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
      const t = new BigUint64Array(buf.getMappedRange(0, bytes).slice(0));
      buf.unmap();
      this.busy[p.idx] = false;
      this.take(p.entries, t);
    }, () => { this.busy[p.idx] = false; });
  }

  take(entries, t) {
    const ms = {};
    const order = [];
    let first = 0n, lastEnd = 0n, prev = null, gapName = null;
    for (const e of entries) {
      if (!(e.name in ms)) { ms[e.name] = 0; order.push(e.name); }
      if (e.gap) { gapName = e.name; continue; }
      const a = t[e.slot * 2], b = t[e.slot * 2 + 1];
      if (a === 0n || b <= a) continue;
      if (first === 0n) first = a;
      if (prev !== null && a > lastEnd) {
        ms[gapName || prev] += Number(a - lastEnd) * 1e-6;
      }
      ms[e.name] += Number(b - a) * 1e-6;
      gapName = null;
      prev = e.name;
      lastEnd = b;
    }
    const total = first === 0n ? 0 : Number(lastEnd - first) * 1e-6;
    this.last = ms;
    this.order = order;
    this.frameMs = total;
    for (const k of order) this.sum[k] = (this.sum[k] || 0) + ms[k];
    this.sumFrameMs += total;
    this.frames++;
  }

  reset() { this.sum = {}; this.sumFrameMs = 0; this.frames = 0; }

  avg(name) { return this.frames ? this.sum[name] / this.frames : 0; }
  avgFrame() { return this.frames ? this.sumFrameMs / this.frames : 0; }

  line(useAvg = false) {
    if (!this.on) return '';
    return this.order.map(k =>
      `${k} ${(useAvg ? this.avg(k) : (this.last[k] || 0)).toFixed(2)}`).join('  ');
  }

  table() {
    const o = {};
    for (const k of this.order) o[k] = Number(this.avg(k).toFixed(3));
    return o;
  }
}

# Environment panoramas

`?cubemap=env/<file>` loads one of these instead of the procedural sky. One
equirectangular `.hdr` (Radiance RGBE) or a directory of six faces; the page
resamples it into a cubemap and builds the mips itself.

## What is here

- **`quarry_cloudy_1k.hdr`** — [Quarry Cloudy](https://polyhaven.com/a/quarry_cloudy)
  from Poly Haven, 1k. Photographed by Dimitrios Savva, processed by Jarod
  Guest. Licensed [CC0](https://polyhaven.com/license): public domain, free to
  redistribute and to use commercially. Not our work, and not re-licensed —
  it is included so the link in the README works from a fresh clone.

Any other Poly Haven HDRI drops in beside it and works the same way. A missing
file is not fatal: the page logs the 404 and falls back to the procedural sky.

# Particles4All

Position Based Fluids in WebGPU: water and rigid bodies solved in the same
constraint loop, with the surface rendered in screen space.

<img width="2554" height="1276" alt="スクリーンショット 2026-08-20 234309" src="https://github.com/user-attachments/assets/80da8b4a-2e36-4abf-993a-ab50736070a3" />

## Run it

WebGPU needs a secure context, so open it over http rather than from `file://`:

```
python -m http.server 8080
```

then <http://localhost:8080>. Chrome or Edge 113+, Safari 18+, or Firefox with
WebGPU enabled. No build step, no dependencies — it is ES modules and WGSL.

`Scene` picks the size, `quality` is the fraction of the window the per-pixel
passes run at, `box size` moves a wall while the water is in it, and
`Pour water` runs a hose from the near wall. Left-drag rotates, right or middle
drag pans, the wheel zooms, moving the pointer over the water pushes it, space
pauses, **H** hides the controls and **D** opens the debug window.

Panorama: [Quarry Cloudy](https://polyhaven.com/a/quarry_cloudy) by
[Poly Haven](https://polyhaven.com), CC0.

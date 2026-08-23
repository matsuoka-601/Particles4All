# Particles4All

Position Based Fluids in WebGPU: water and rigid bodies solved in the same
constraint loop, with the surface rendered in screen space.

<img width="2554" height="1276" alt="スクリーンショット 2026-08-20 234309" src="https://github.com/user-attachments/assets/80da8b4a-2e36-4abf-993a-ab50736070a3" />

The features of this repo are like below:
- **Treat both fluid and rigid bodies as sets of particles and solves them in one unified solver** presented in [Unified Particle Physics for Real-Time Applications](https://matthias-research.github.io/pages/publications/flex.pdf).
    - Buoyancy is naturally realized within this unified solver.
- **Reconstruct a smooth fluid surface using anisotropic kernels** presented in [Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels](https://cs.nyu.edu/~exact/doc/anisotropic.pdf).
    - Reconstructed fluid surface is further **smoothed in screen space using a narrow-range filter** presented in [A Narrow-Range Filter for Screen-Space Fluid Rendering](https://ttnghia.github.io/pdf/NarrowRangeFilter.pdf).
- **Realistic surface tension** is realized based on the method presented in  [Versatile surface tension and adhesion for SPH fluids](https://dl.acm.org/doi/10.1145/2508363.2508395).

<img width="800" height="482" alt="レコーディング 2026-08-22 140438" src="https://github.com/user-attachments/assets/ac4e999f-245a-49df-afd7-9222457e04f9" />

## Simulation
### Unified particle physics
(TODO: write)
### Surface tension
(TODO: write)
## Rendering
### Screen-space fluid rendering using narrow-range filter
(TODO: write)
### Anisotropic kernels for smoothing the fluid surface
(TODO: write)
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

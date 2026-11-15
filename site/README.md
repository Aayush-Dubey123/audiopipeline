# AudioPipeline docs site

Static showcase site for AudioPipeline, built with [Astro Starlight](https://starlight.astro.build).

| Command | Action |
|---|---|
| `npm install` | Install dependencies |
| `npm run dev` | Local dev server at `http://localhost:4321` |
| `npm run build` | Build the static site into `dist/` |
| `npm run preview` | Serve the built site locally |

## Content

- Pages: `src/content/docs/` (MDX). Sidebar order: `astro.config.mjs`.
- Custom components: `src/components/` (`AudioCompare`, `PipelineFlow`, `StatGrid`, `BarCompare`, `FixCard`).
- Theme: `src/styles/theme.css`.
- Audio samples: `public/audio/*.mp3`; waveform data and metrics: `src/data/showcase.json`.
  All samples use the model's built-in demo voice (no real person's voice).

## Deploy (free)

**Cloudflare Pages:** connect the repo, set the root directory to `site`, the build command to `npm run build` and the output directory to `dist`.

**GitHub Pages:** set `site` (and `base` for a project site) in `astro.config.mjs`, then deploy `dist/` with the official
[Astro GitHub Pages action](https://docs.astro.build/en/guides/deploy/github/).

Astro 7 recommends Node.js 22.22 or newer.

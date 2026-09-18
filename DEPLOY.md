# Publishing the maps

The project is published as one static city map. It contains the research
write-up, the building-level map, and the interactive time-of-day/day-of-year
shadow projection. The map is an HTML file plus a `tiles/` directory it
fetches from.

## GitHub Pages (free, and what the links in the write-up point at)

1. Create an empty repository on github.com — public, no README, no
   `.gitignore` (this repo has one).

2. Push:

       git remote add origin https://github.com/<you>/madrid-solar.git
       git branch -M main
       git push -u origin main

3. In the repository, **Settings → Pages**. Under *Build and deployment*,
   set **Source: Deploy from a branch**, **Branch: `main`**, **Folder:
   `/docs`**, and Save.

   Pages will not serve `web/` directly — it only offers the repository root
   or `/docs`. So publish by copying the built pages into `docs/`:

       python publish_docs.py

   then commit and push `docs/`.

4. After a minute the site is at

       https://<you>.github.io/madrid-solar/

## Why the city map cannot be opened from `file://`

The per-panel layer is fetched per tile as you pan, so the browser applies
its usual rules about local file access and the fetches fail silently — you
get the buildings and no panels. Any HTTP server fixes it, including:

    python -m http.server -d docs 8000

The final page needs HTTP because the panel layer is fetched on demand.

## Other free hosts

Netlify Drop (drag the `docs/` folder onto https://app.netlify.com/drop),
Cloudflare Pages and Vercel all work the same way and serve the tiles
correctly. Nothing here needs a build step, a framework or a server.

## Size

    docs/index.html          the single city map and write-up
    docs/tiles/              per-panel detail, fetched on demand

Comfortably inside the 1 GB soft limit for a GitHub Pages site.

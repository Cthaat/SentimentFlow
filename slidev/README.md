# Welcome to [Slidev](https://github.com/slidevjs/slidev)!

To start the slide show:

- `yarn install`
- `yarn run dev`
- visit <http://localhost:3030>

Docker Compose also builds and serves this deck:

- from repo root: `docker compose up slides`
- visit <http://localhost:3031>

The interactive demo can call the FastAPI backend for live text prediction. In
Compose, the `slides` service starts `backend` as a dependency and Nginx proxies
same-origin `/api/` requests to `backend:8846`.

Edit the [slides.md](./slides.md) to see the changes.

Learn more about Slidev at the [documentation](https://sli.dev/).

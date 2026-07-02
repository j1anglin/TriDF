# TriDF Demo Page

This repository contains the static demo page for **TriDF**, a benchmark for interpretable
DeepFake detection across image, video, and audio. It includes a leaderboard, qualitative
examples, dataset details, and a citation section. All content is pure HTML/CSS/vanilla JS and
deployable on GitHub Pages.

Paper: [arXiv:2512.10652](https://arxiv.org/abs/2512.10652)

## Local Preview

From the repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/` in your browser.

## GitHub Pages Deployment

1. Go to **Settings → Pages**.
2. Select **Deploy from a branch**.
3. Choose branch `gh-pages` and folder `/ (root)`.
4. Save and wait for the GitHub Pages URL to go live.

The site works for both project pages (`https://username.github.io/repo/`) and user pages
(`https://username.github.io/`). All asset paths are relative and JSON is loaded via
`document.baseURI`.

## Updating Leaderboard or Examples

Edit the JSON files in `data/`:

- `data/leaderboard.json`
- `data/examples.json`

The frontend automatically re-renders the leaderboard and cards on page load.

## License and Acknowledgements

Adapted from [Nerfies](https://nerfies.github.io/). This website is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
See `LICENSE` for details.

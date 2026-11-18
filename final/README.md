# EVST Topic 11: final submission

- `EVST_Topic11_Paper.pdf`: the paper (source in `paper/`).
- `EVST_Topic11_Final.pptx`: the 25-minute deck, 16 slides, with speaker notes for four speakers.
- `explainer/EVST_Topic11_Explained.md`: a ground-up guide to the project, models and results.

Paper, deck and explainer cover the same hypotheses: H1 to H8 and S0.

## Rebuild

Run from the repository root with `.venv/Scripts/python`:
1. `final/scripts/block_bootstrap.py`: spatial-block confidence intervals (about 15 minutes).
2. `final/scripts/make_tables.py`: paper tables and `paper/numbers.tex`.
3. `final/scripts/figures.py`: paper figures. Run again with `SLIDES=1` and the arguments `missing sflip socc` for the slide versions.
4. In `final/paper/`: `latexmk -pdf main.tex`.
5. In `final/slides/`: `node deck.js`, then `add_motion.py deck_raw.pptx deck_motion.pptx`. Both need the mid-evaluation deck builder (`presentation/build`), which is not in this repository.

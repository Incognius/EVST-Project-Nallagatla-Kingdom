# EVST Topic 11: final submission

- `EVST_Topic11_Paper.pdf`: the paper (source in `paper/`).
- `EVST_Topic11_Final.pptx`: the 25-minute deck, 17 slides, with speaker notes for four speakers.
- `explainer/EVST_Topic11_Explained.md`: a ground-up guide to the project, models and results.

Paper, deck and explainer cover the same hypotheses: H1 to H8 and S0. The deck and explainer also include the Europe CNN test (E20) and the survey-size result (E21), which are not yet in the paper.

## Rebuild

Run from the repository root with `.venv/Scripts/python`:
1. `final/scripts/block_bootstrap.py`: spatial-block confidence intervals (about 15 minutes).
2. `final/scripts/make_tables.py`: paper tables and `paper/numbers.tex`.
3. `final/scripts/survey_size.py`: the survey-size subsampling (E21, under a minute). Then `final/scripts/figures.py` for the paper figures; run it again with `SLIDES=1` and the arguments `bias missing sflip socc cvsize glc` for the slide versions. The Europe CNN numbers (E20) come from `results/glc24/`, copied from the teammate branch `phase-d-robustness`.
4. In `final/paper/`: `latexmk -pdf main.tex`.
5. In `final/slides/`: `node deck.js`, then `add_motion.py deck_raw.pptx deck_motion.pptx`. Both need the mid-evaluation deck builder (`presentation/build`), which is not in this repository.

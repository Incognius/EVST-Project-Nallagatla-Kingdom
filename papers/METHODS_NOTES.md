# Methods notes: what each paper actually did, and what we reuse

Status per paper: **full text read** = methods checked against the full text in this repo;
**pending** = full text not yet read. Each note lists the data, the model, the evaluation, then what
we copy and what we test that the paper did not. These are our summaries for planning. Write the
report's review text from your own reading.

---

## Ploton et al. 2020, *Nat. Commun.* 11:4540 — full text read
- **Data.** Forest above-ground biomass (AGB) in central Africa: about 1 km reference pixels from forest
  inventories, with remote-sensing and environmental covariates.
- **Model.** Random forest (RF_RSE).
- **Evaluation.** Three designs on the same model:
  - random 10-fold CV: R² 0.53, RMSPE 56.5 Mg/ha;
  - spatial 44-fold CV with k-means clusters: R² 0.14, RMSPE 77.5 Mg/ha, close to a null model;
  - buffered leave-one-out (B-LOO) with exclusion radii 0–150 km: R² falls from about 0.50 at 0 km to
    about 0.15, and is near zero beyond about 100 km.
- **Mechanism.** Random splits put test pixels inside the autocorrelation range of training pixels, so
  the CV scores interpolation, not prediction into new areas.
- **What we reuse.** Random vs spatial-block CV (`validate/blocks.py`).
- **What we add.** Ploton's problem is autocorrelation. Ours is *observer bias in the test presences
  themselves*. E02 shows spatial blocking does not remove it: blocked presence-vs-random-background AUC
  still prefers the uncorrected model.

## Deneu et al. 2021, *PLoS Comput. Biol.* 17:e1008856 — full text read
- **Data.** France, 4,520 plant species, presence-only occurrences, with location uncertainty from
  metres to 10 km. 33 rasters (climate, soil, elevation, land cover); land cover is one-hot encoded,
  giving 77 channels.
- **Model.** A CNN on 64×64-pixel environmental tensors around each occurrence, compared with RF and
  other point models. It is a multi-class softmax: P(species | an observation was made here).
- **Bias handling (corrects our lit-review draft).** The authors argue this conditional formulation is
  less sensitive to observation bias, since it needs no absences or pseudo-absences. Implicitly every
  other species' observations act as background, the same logic as Phillips's target-group
  background. So Deneu does not ignore bias.
- **Evaluation.** A random 90/10 occurrence split. Main metric: mean top-k accuracy per species.
  AUC and TSS are reported with caveats about pseudo-absences.
- **What we reuse.** The patch-CNN idea: `models/batched.py`, 9×9 km patches at 1 km, scaled to a
  6 GB GPU.
- **What we add.** The test split is random, so spatial skill and bias leakage are untested. We put the
  CNN in a binary presence/background setting under a measured bias (E03) and compare its leakage
  with lower-capacity models.

## Elith et al. 2020, *Biodiversity Informatics* 15:69-80 (disdat) — full text read
- **Data.** NCEAS data: 6 regions (AWT, CAN, NSW, NZ, SA, SWI) and 226 anonymised species. Presence-only
  training records, 10,000 random background points per region, and independent presence/absence
  evaluation sites, released on OSF.
- **Key statements.**
  - A model trained and evaluated on biased presence-only data can appear to perform well while not
    representing the true distribution.
  - The original NCEAS work (Elith et al. 2006) found some regions "clearly hampered by bias", which
    led directly to Phillips et al. 2009.
- **What we reuse.** The whole dataset (E02).
- **Consequence for the review.** Elith 2006 did not ignore bias; it diagnosed it. Reword the draft
  sentence accordingly.

## Zizka et al. 2021, *Ecography* 44:25-32 (sampbias) — full text read
- **Model.** Records per grid cell are Poisson: S_i ~ Poi(λ_i), with λ_i = q·exp(−Σ_j w_j X_i(j)).
  X_i(j) is the straight-line distance from cell i to the nearest feature of bias factor j.
  A weight w ≈ 0 means uniform sampling.
- **Priors.** q ~ Γ(1, 0.01). The weights are hierarchical: w ~ Γ(1, b) with b ~ Γ(1, 0.001), so
  they shrink toward 0 when the data show no bias.
- **Inference.** MCMC.
- **Scale.** Meant for regional work at about 100 m – 10 km resolution, because it assumes
  homogeneous habitat. The study area can be a polygon.
- **What we reuse.** `bias/sampbias.py` matches these priors exactly; it was initially mis-specified
  (flat Exp(1) prior on max-rescaled distances) and was corrected after reading the full text. It is
  cross-checked with the R package on an identical grid.

## Hughes et al. 2021, *Ecography* 44:1259-1269 — full text read
- **Data.** About 704M GBIF and 38M OBIS records, synonym-filtered.
- **Road statistics.** Records are binned at 0–1, 1–2.5, 2.5–5 and >5 km from roads. 41–65% of
  non-marine records lie within 1 km of a road, and at least 80% within 2.5 km for every taxon.
- **Coverage.** On a 5 km grid, under 7% of the Earth's surface is sampled (11% of land). A 10 km grid
  inflates coverage up to threefold. Excluding birds (87% of records), land coverage falls to 7%.
- **What we reuse.** The exact bins and the 5 km coverage measure (E01).
- **What we add.** The share of *land* near roads as a null. In a densely roaded region, "most records
  are near roads" can be true even under uniform sampling.

## Inman et al. 2021, *Ecosphere* 12:e03422 — full text read
- **Design.** 100 virtual species, from generalists to specialists.
- **Bias.** Geographic and environmental, each at 3 intensities.
- **Corrections.** Geographic thinning (G-Filter), environmental thinning (E-Filter), and
  FactorBiasOut (background weighted toward sampled areas, close in spirit to target-group background).
- **Findings.**
  - FactorBiasOut most often best recovered the true distribution.
  - No method reliably recovered response curves or the true covariates.
  - Narrow-niche species were hardest, and corrections sometimes made them worse.
- **What we reuse.** The 3-intensity design. E03 compares random, TGB (≈ FactorBiasOut), geographic
  thinning (G-Filter) and bias-covariate conditioning.
- **What we add.** Model capacity (GLM → BRT/RF → MLP → patch CNN) as a factor, and a leakage index.

## Fithian et al. 2015, *MEE* 6:424-438 — summary and design read
- **Data.** 36 eucalypt species in south-eastern Australia, presence-only plus presence/absence
  surveys. The presence-only records are biased toward the coast and Sydney. **These are not the
  disdat NSW data**: an earlier plan line assumed they were, and that was wrong.
- **Model.** A joint point-process likelihood over PO and PA data, with one sampling-bias surface
  shared by all species.
- **Finding.** Pooling helps most when a species has few PA data. With PO-only data for one species
  plus PO+PA for others sharing the bias, its range can be estimated without bias.
- **Phase B.** `multispeciesPP` on the disdat NSW groups (which have both PO and PA).

## Oliver et al. 2021, *PLoS Biol.* 19:e3001336 — full text read
- **What it does.** National indicators of data coverage and sampling effectiveness, 1950–2019.
- **What we reuse.** The effectiveness idea, adapted to one region (E01): new species × 10 km cell
  combinations per 1,000 records, per year and per source.

## Meyer et al. 2015, *Nat. Commun.* 6:8221 — full text read
- **What it does.** Builds "digital accessible information" from 157M point records and range maps for
  21,170 vertebrates. Drivers of coverage: distance to researchers, research funding, and participation
  in data sharing.
- **What we reuse.** Justification for WorldPop (observer proximity) in the effort model, alongside roads.

## Fourcade et al. 2014, *PLoS ONE* 9:e97122 — full text read
- **What it does.** A virtual-species comparison of MaxEnt bias corrections, with no consensus winner.
- **Role.** Context for Inman 2021 and for E03's corrections table.

---

## Pending
| Paper | What we reproduce | Status |
|---|---|---|
| Elith et al. 2006 *Ecography* | 16-method benchmark (E02 baseline) | PDF in repo, notes pending |
| Valavi et al. 2019 *MEE* (blockCV) | Range-based block size. Our variogram ranges did NOT match blockCV's (see the E02 sensitivity run) | PDF in repo, notes pending |
| Johnston et al. 2021 *DDI* | eBird effort covariates (phase B, needs the EBD) | PDF in repo, notes pending |
| Phillips et al. 2009 *Ecol. Appl.* | Target-group background on the same NCEAS data (E02 reproduces the gain) | need PDF |
| Valavi et al. 2022 *Ecol. Monogr.* | Published per-model AUCs on disdat, to validate our E02 baseline | need PDF |
| Beck 2014, Isaac 2020, Hortal 2015 | Framing only | need PDF |

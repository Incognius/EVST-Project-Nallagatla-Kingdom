# Phase B validation: independent truth, a second continent, and prior art

EVST Topic 11 · 17 November 2026 · follows `reports/phase_b_bayesian.md`

Every earlier result was scored against citizen-science records, which carry the same biases we are
studying. This round asks three questions:
1. Do our findings hold against a **systematic survey** that was designed to be unbiased?
2. Do they hold on a **second, independent dataset** from another continent?
3. Has **someone already done this**, and how large are our effects in practical terms?

## Data added

| Dataset | What it is | Why it matters |
|---|---|---|
| **Kerala Bird Atlas 2015–2020** (Dryad [10.5061/dryad.zpc866t8g](https://doi.org/10.5061/dryad.zpc866t8g)) | 6.6 km grid over all of Kerala. Every cell has up to 32 timed 15-minute complete checklists, run by trained volunteers in fixed wet- and dry-season rounds | Systematic, effort-standardised presence/absence for birds over most of the southern Western Ghats. It is the closest thing to ground truth for "what lives here that iNaturalist has not recorded" |
| **GeoLifeCLEF 2024** (Europe) | 5.08 M presence-only records and 88,987 presence/absence plots | A second continent and a second taxon (plants). Its presence-only records lack observer IDs, which tests how fragile the visit definition is |

The atlas zip holds `kba_data.rds`, `kba_names.rds` and `kba_species_details.rds`. These were exported to
CSV with R and harmonised to our iNaturalist names through the atlas's own name table. 844 atlas cells fall
inside our study region, and 196 bird species have enough data to model.

## 1. How much does citizen science miss? (E09)

- **435 of 844** atlas cells (52%) have **no iNaturalist bird record at all** up to 2020.
- In the cells iNaturalist *did* visit, it recorded only **12% of the species the atlas found**. On average
  **88%** were missed. Across all cells the figure is 94%.
- The atlas averages 62 species per cell, **4.2 of them Western Ghats endemics** that iNaturalist has never
  logged in that cell.

So the gap the topic asks about is not small. It is most of the avifauna, in most of the landscape.

## 2. Can AI identify the missing species? (E09b), the headline result

We take every (cell, species) pair that iNaturalist **never recorded** by 2020: **170,312 pairs**. The
atlas found the species in **27.4%** of them. We then ask each method to rank those pairs.

| Method | Mean per-species AUC (196 species) | Atlas hit rate in the method's top 10% |
|---|---|---|
| Random guess / range-wide prevalence | 0.500 | 27.4% (base rate) |
| Nearest existing record (simple distance) | 0.659 | 41.2% |
| GLM, target-group background | 0.725 | 45.3% |
| BRT, random background (uncorrected) | 0.736 | 46.9% |
| **BRT, target-group background (bias-corrected)** | **0.758** | **48.4%** |

- Bias-corrected BRT beats the distance baseline for **89%** of species: +0.098 AUC, Wilcoxon p = 1.4 × 10⁻²⁹.
- It beats the uncorrected BRT for **76%** of species: +0.022, p = 1.2 × 10⁻¹³.
- **Western Ghats endemics (17 species):** bias-corrected BRT scores **0.834** against 0.740 for the distance
  baseline (p = 1.5 × 10⁻⁵). The gain from bias correction is twice as large for endemics as for
  widespread species (+0.040 vs +0.020).

**In plain terms:** if a surveyor visits the pairs the corrected model ranks highest, nearly half the
"missing" species turn up. Picking at random finds about a quarter. **This is a direct, independently
validated "yes" to the topic question**, and correcting for sampling bias is what makes the AI better than
the naive model.

### Why this reverses the mid-evaluation null
At the mid-evaluation, bias-corrected richness maps did *not* predict where **later iNaturalist** records
appeared. The atlas shows why. Later citizen-science records are themselves biased, because they follow
observers, not species (E08). A test that uses biased records as truth cannot reward a model for finding
what biased observers never visit. Scoring against a systematic survey removes that problem, and the
corrected model wins.

## 3. Random cross-validation picks the wrong model, again (H3)

For each species we compared what 5-fold random CV says with what the atlas says:

| | Random CV | Atlas truth |
|---|---|---|
| Mean AUC, uncorrected BRT | 0.851 | 0.742 |
| Mean AUC, corrected BRT (TGB) | 0.752 | 0.763 |
| Species where the corrected model "wins" | **1.0%** | **75.5%** |

Random CV overstates the uncorrected model by **+0.109 AUC**. It is almost exactly right for the corrected
model (−0.011). It chooses the wrong model for three species in four.

On disdat (six regions, mid-evaluation) the agreement was 38%. Here it is 25.5%. Two continents, two
independent truth sets: the pattern replicates.

## 4. Does the Bayesian discovery model survive tougher checks?

### 4a. Spatial-block bootstrap (Western Ghats, 50 km blocks)
The earlier confidence intervals resampled cells, which ignores spatial autocorrelation. Resampling 50 km
blocks instead:

| Origin | 2016 | 2018 | 2020 | 2022 |
|---|---|---|---|---|
| Discovery model vs effort regression (nats/cell) | +0.022 [0.000, 0.043] | +0.054 [0.030, 0.078] | +0.039 [0.007, 0.069] | +0.020 [0.003, 0.036] |
| Intercepts-only discovery model | +0.019 [0.005, 0.034] | +0.056 [0.035, 0.075] | +0.047 [0.024, 0.069] | +0.024 [0.008, 0.038] |

All four origins still exclude zero, though 2016 only just does. By taxon (2018), odonates, birds,
butterflies and reptiles stay significant. Amphibians and mammals do not. Plants are negative
(−0.041 [−0.083, 0.000]).

**Where the gain comes from.** The gain sits in the poorly sampled cells, which is where the topic says it
should matter:

| Origin | No prior visits | 1–5 prior visits | >5 prior visits |
|---|---|---|---|
| 2018 | +0.040 [0.011, 0.069] | +0.050 [0.014, 0.085] | +0.084 [0.004, 0.161] |
| 2020 | +0.075 [0.014, 0.147] | +0.045 [0.003, 0.084] | −0.003 n.s. |
| 2022 | +0.026 [0.004, 0.049] | +0.055 [0.028, 0.085] | −0.022 n.s. |

### 4b. Second continent (E10, GeoLifeCLEF Europe): **the discovery forecast fails**

| Origin | 2018 | 2019 | 2020 |
|---|---|---|---|
| Discovery model vs effort regression | −0.129 [−0.162, −0.096] | −0.194 [−0.229, −0.154] | −0.169 [−0.206, −0.123] |

GeoLifeCLEF's presence-only records carry no observer ID, so a "visit" had to be defined as
*publisher × day*. One publisher-day can be thousands of records from many people. Detection per "visit"
then becomes nearly certain, the saturation curve collapses (P₀ ≈ 1), and the model loses its advantage.
**The model's edge depends on a meaningful unit of effort.** iNaturalist provides one (observer-day).
Aggregated portal data do not. We report this as a boundary condition, not as noise.

### 4c. Does the Bayesian "undetected probability" rank missed species better than visit counts?

| Truth | Bayesian P₀ (full) | Bayesian P₀ (intercepts) | 1 / (1 + visits) | Chao coverage deficit | Chao2 undetected |
|---|---|---|---|---|---|
| Atlas fraction missed, Kerala (409 visited cells) | 0.832 | 0.847 | **0.847** | −0.219 | −0.838 |
| PA-plot fraction missed, Europe (6,151 cells) | 0.638 | – | **0.638** | 0.218 | – |

(Spearman ρ, with block-bootstrap 95% CIs in `results/E09_kba_missed_species.csv` and `results/E10_glc_discovery.csv`.)

**Honest reading.** For ranking which cells are most incompletely sampled, the Bayesian P₀ is no better than
"fewest visits". The classical Chao indicators do far worse: Chao2 points the **wrong way** (ρ = −0.84),
because it can only estimate undetected species where many visits already exist. The Bayesian model's
absolute numbers are also not trustworthy. It predicts P₀ ≈ 0.99–1.0 where 0.88 is observed, and its pool
sizes are far larger than the real Kerala avifauna. **Pool size and detectability are not separately
identified from presence-only records.** The model's value is in forecasting *how many new records the
next visits will add* (§4a), not in estimating true richness.

## 5. Has this been done before? (novelty check)

| Our component | Closest prior work | Verdict |
|---|---|---|
| Beta-binomial incidence model for undetected species | Chiu 2022 *MEE* (beta-binomial lower-bound richness estimator from singletons, doubletons and tripletons) and Chiu 2023 *MEE* (truncated beta-binomial mixture) ([DOAJ record](https://doaj.org/article/fd488256d86f4bf6bc8d561ed720b413), [NTU record](https://scholars.lib.ntu.edu.tw/handle/123456789/631760)); Colwell, Mao & Chang 2004 binomial-mixture extrapolation ([PDF](https://www.esapubs.org/archive/mono/M084/003/appendix-A.pdf)) | **Not novel as a richness estimator.** Our likelihood is close to theirs. They estimate one community at a time by moments. We use a spatially hierarchical model fitted jointly across thousands of cells and forecast new records forward in time. We found no published version of that, but we should call it an *extension* of Chiu's model |
| Multispecies detection modelling | Kéry & Royle hierarchical multispecies occupancy (2008 onward) | Those models need repeat-visit detection data. Ours runs on presence-only citizen-science incidence |
| Forecasting where sampling yields new species | Callaghan et al. 2022 *Ecosphere* (completeness vs effort); Mondain-Monval et al. 2024 (adaptive sampling, simulated); an eDNA study predicting new occurrences from added sampling | Closest in spirit, but none validates an effort-conditioned discovery forecast forward in time on real citizen-science data at landscape scale |
| Random CV fails to reward bias correction | Baker et al. 2024 *Divers. Distrib.* ([link](https://onlinelibrary.wiley.com/doi/full/10.1111/ddi.13802)); sample-bias-correction assessment ([arXiv 2103.07107](https://arxiv.org/abs/2103.07107)); environmental-bias correction and transferability, *Ecography* ([link](https://nsojournals.onlinelibrary.wiley.com/doi/10.1002/ecog.08002)); presence-absence AUC inflation ([PubMed 22624221](https://pubmed.ncbi.nlm.nih.gov/22624221/)) | **Known phenomenon.** Those studies mostly use virtual species. Our contribution is a *real-data* replication with independent survey truth on two systems (disdat; Kerala atlas), with an effect size: 1% vs 75.5% |
| Validating citizen-science SDMs against a systematic atlas | eBird vs atlas SDMs (*J. Biogeogr.* 10.1111/jbi.14382); a UK atlas reference study (*Ecol. Model.* S0304380019304351); Jha, Nameer & Praveen 2021 used the same Kerala atlas for 17 endemics ([Dryad](https://doi.org/10.5061/dryad.ncjsxksvp)) | Atlas validation exists. Scoring **only the pairs citizen science never recorded** ("does the model find what was missed?") against an atlas appears new |
| Western Ghats context | Kerala Bird Atlas paper: high-elevation southern Ghats are the largest under-sampled blocks ([ResearchGate](https://www.researchgate.net/publication/371856566_Kerala_Bird_Atlas_2015-20_features_outcomes_and_implications_of_a_citizen-science_project)); IUCN maps overestimate Western Ghats endemic ranges ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0006320716310588)); eBird under-samples Indomalayan forest ([La Sorte 2024](https://onlinelibrary.wiley.com/doi/10.1111/ddi.13863)); iNaturalist users stay near home ([CSTP 2025](https://theoryandpractice.citizenscienceassociation.org/articles/10.5334/cstp.868)) | Consistent with our E01 bias anatomy and E08 |

**Claims we can make:**
1. Validated against a systematic atlas: in a biodiversity hotspot, bias-corrected SDMs rank species that
   iNaturalist never recorded in a cell well above distance and uncorrected baselines, and endemics gain
   most.
2. Random CV inverts the model choice on real data in two independent systems.
3. An effort-conditioned hierarchical extension of the Chiu beta-binomial model forecasts new
   species-in-cell records better than regression and Chao extrapolation. This holds only where effort is
   measured per observer.

**Claims we should drop:**
- That the Bayesian model estimates true undetected richness. It does not (§4c).
- That beta-binomial discovery modelling is new.
- That habitat covariates drive the forecast gain. The ablation says no.

## 6. How meaningful are the effects?

- **+0.054 nats/cell** (2018) means the discovery model gives the realised outcome in each cell about
  **5.5% more probability** than the regression does. Over 2,646 cells that is **≈143 nats**: an aggregate
  likelihood ratio of e¹⁴³ in favour of the discovery model. The effect is small per cell but consistent
  across origins and taxa. It is a forecast-quality gain, not a dramatic one.
- **+0.098 AUC over nearest-record, and 48% vs 27% hit rate.** For field planning, the corrected model
  turns about one productive pair in four into about one in two. That is the most *useful* number we have.
- **+0.022 AUC from bias correction** is modest on average. It is larger for endemics, and it is the
  difference random CV gets backwards.

## 7. Updated answer to the topic question

> *Can AI identify biodiversity patterns missing from conventional sampling?*

**Yes, if both the AI and the test account for bias.** Against a systematic atlas, a bias-corrected
model finds species that citizen science never recorded, at almost twice the base rate. Western Ghats
endemics show the largest gain. Two conditions matter:
1. **Correction:** use target-group background. Uncorrected models learn observer behaviour.
2. **Honest evaluation:** random CV and later citizen-science records both reward the wrong model. Only
   independent survey data shows the gain.

A Bayesian sampling model adds a second, complementary tool. It forecasts *how many* new records the next
visits will produce, and it helps most in the least-sampled cells. It does not estimate true missing
richness, and it needs observer-level effort data.

## Limitations
- The atlas covers birds only, and Kerala only. The northern Ghats have no systematic truth here.
- Atlas truth is itself imperfect. Thirty-two 15-minute lists per cell miss nocturnal and secretive species.
- Kerala SDMs were trained on iNaturalist only (2020 cut-off), with no eBird. eBird would shrink the gap.
- The Europe test (E10) used publisher-day visits. A fair re-test needs observer-level data such as GBIF
  `recordedBy`.

## Reproduce
```bash
python experiments/E05_discovery_full.py
python experiments/E09_kba_validation.py
python experiments/E09b_kba_missed_species.py
python experiments/E10_glc_discovery.py
```
The atlas zip goes in `data/raw/kba/`, exported to CSV with `KBA.script.R`-style `readRDS` + `write.csv`.

const path = require("path");
const fs = require("fs");
const BUILD = ["../../presentation/build", "../../../presentation/build"].map(p => path.resolve(__dirname, p)).find(p => fs.existsSync(p));
const pptxgen = require(path.join(BUILD, "node_modules/pptxgenjs"));
const SKILL = process.env.PPTX_SKILL || "C:/Users/ponna/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/9ee3c4f5-8eed-4703-8676-0a85a1365a7b/3eba1376-adc6-498c-9a28-ecdc7d727ec5/skills/pptx";
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");
const SZ = JSON.parse(fs.readFileSync(path.join(__dirname, "sizes.json"), "utf8"));
const A = (f) => path.join(BUILD, "assets", f);
const F = (f) => path.join(__dirname, "figs", f);
const AR = (f) => SZ[f][0] / SZ[f][1];

const INK = "23302A", FOREST = "1F3D2B", DEEP = "24452F", MOSS = "4F6B35", LEAF = "8BAA4F";
const SAGE = "C9D6B0", PAPER = "F1F0E8", CARD = "F7F6EF", ORANGE = "D2601A", APRICOT = "F0A25C";
const STONE = "7E7C72", RUST = "8E3510", BLUE = "3B6EA5";
const HEAD = "Bodoni MT", BODY = "Gill Sans MT", LABEL = "Bahnschrift";
const THEME = {
  name: "Paper Atlas", headFontFace: HEAD, bodyFontFace: BODY,
  colors: { dk1: INK, lt1: PAPER, dk2: FOREST, lt2: "E2E8D6", accent1: ORANGE, accent2: MOSS, accent3: LEAF,
            accent4: APRICOT, accent5: DEEP, accent6: STONE, hlink: ORANGE, folHlink: STONE },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Ponnambalam V, Nandini C, Mohit Nallagatla, Manali Gupta";
pres.title = "Where the roads end: EVST Topic 11 final evaluation";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
const footer = (dark) => ({ text: { text: "EVST · TOPIC 11 · FINAL EVALUATION", options: {
  x: 0.75, y: 7.08, w: 5, h: 0.25, fontFace: LABEL, fontSize: 9, color: dark ? SAGE : STONE, charSpacing: 3, margin: 0 } } });
const num = (c) => ({ x: 12.35, y: 7.02, w: 0.5, h: 0.3, fontFace: LABEL, fontSize: 10, color: c, align: "right" });
pres.defineSlideMaster({ title: "PAPER", background: { path: A("bg_paper.jpg") }, objects: [footer(false)], slideNumber: num(STONE) });
pres.defineSlideMaster({ title: "SAGE", background: { path: A("bg_sage.jpg") }, objects: [footer(false)], slideNumber: num(STONE) });
pres.defineSlideMaster({ title: "FOREST", background: { path: A("bg_forest.jpg") }, objects: [], slideNumber: num(SAGE) });

let n = 0;
const nm = (eff, d) => `a${String(++n).padStart(2, "0")}_${eff}_${d}`;
const shadow = (o = {}) => ({ type: "outer", blur: o.blur ?? 10, offset: o.offset ?? 4, angle: o.angle ?? 45,
  color: "142016", opacity: o.opacity ?? 0.28 });
function newSlide(master, section) { n = 0; return pres.addSlide({ masterName: master, sectionTitle: section }); }
function heading(s, eyebrow, title, dark = false) {
  s.addText(eyebrow, { x: 0.75, y: 0.36, w: 10, h: 0.3, fontFace: LABEL, fontSize: 11, color: dark ? APRICOT : ORANGE,
    charSpacing: 4, margin: 0, isTextBox: true, objectName: nm("fade", "eyebrow") });
  s.addText(title, { x: 0.75, y: 0.68, w: 11.3, h: 0.75, fontFace: HEAD, fontSize: 31, color: dark ? PAPER : FOREST,
    margin: 0, valign: "top", isTextBox: true, objectName: nm("up", "title") });
}
function leaf(s, file, key, x, y, w) { s.addImage({ path: A(file), x, y, w, h: w / AR(file), objectName: `!!${key}` }); }
const SW = 13.333, SH = 7.5;
const leafTR = (s, w, dx = 0, dy = 0) => leaf(s, "leaf_branch_top.png", "leafT", SW - 0.8 * w + dx, -0.22 * (w / AR("leaf_branch_top.png")) + dy, w);
const leafBR = (s, w, dx = 0, dy = 0) => leaf(s, "leaf_bottom.png", "leafB", SW - 0.84 * w + dx, SH - 0.7 * (w / AR("leaf_bottom.png")) + dy, w);
const leafR = (s, w, y) => leaf(s, "leaf_pine_mid.png", "leafP", SW - 0.7 * w, y, w);

function card(s, x, y, w, h, desc, eff = "fade") {
  s.addShape(pres.ShapeType.rect, { x: x + 0.08, y: y + 0.08, w, h, fill: { color: SAGE }, line: { type: "none" },
    objectName: nm(eff, desc + "_back") });
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: CARD }, line: { type: "none" },
    shadow: shadow({ blur: 12, offset: 3 }), objectName: nm(eff, desc) });
}
function fig(s, file, x, y, opt = {}) {
  let w = opt.w, h = opt.h;
  if (w && !h) h = w / AR(file);
  if (h && !w) w = h * AR(file);
  const pad = opt.pad ?? 0.12;
  card(s, x, y, w + 2 * pad, h + 2 * pad, "figcard", "fade");
  s.addImage({ path: F(file), x: x + pad, y: y + pad, w, h, objectName: nm(opt.eff ?? "up", "fig") });
  return { x, y, w: w + 2 * pad, h: h + 2 * pad };
}
function tag(s, x, y, w, h, title, rows, rot = -1.2) {
  const runs = [{ text: title, options: { fontFace: LABEL, fontSize: 10, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } }];
  rows.forEach(([k, v], i) => {
    runs.push({ text: k + "  ", options: { fontFace: LABEL, fontSize: 10, color: INK, bold: true } });
    runs.push({ text: v, options: { fontFace: BODY, fontSize: 11, color: INK, breakLine: i < rows.length - 1, paraSpaceAfter: 3 } });
  });
  s.addText(runs, { shape: pres.ShapeType.rect, x, y, w, h, fill: { color: CARD }, line: { color: INK, width: 0.75 },
    rotate: rot, margin: [7, 9, 7, 9], valign: "top", shadow: shadow({ blur: 10, offset: 4 }), objectName: nm("up", "fieldtag") });
}
function bullets(s, x, y, w, h, items, size = 14, color = INK) {
  const runs = [];
  items.forEach((p, i) => {
    const parts = Array.isArray(p) ? p : [{ text: p }];
    parts.forEach((r, j) => runs.push({ text: r.text, options: { ...(r.options || {}),
      bullet: j === 0 ? { code: "25AA", indent: 14 } : undefined, breakLine: j === parts.length - 1 && i < items.length - 1,
      paraSpaceAfter: 7 } }));
  });
  s.addText(runs, { x, y, w, h, fontFace: BODY, fontSize: size, color, margin: 0, valign: "top", isTextBox: true,
    objectName: nm("fade", "bullets") });
}
const B = (t, c = FOREST) => ({ text: t, options: { bold: true, color: c } });
const T = (t) => ({ text: t });
const LINE = { type: "solid", color: "B9C2A8", pt: 0.75 }, NOL = { type: "none" };
function table(s, x, y, w, colW, header, rows, opt = {}) {
  const fsz = opt.size ?? 11;
  const hdr = header.map((t) => ({ text: t, options: { fontFace: LABEL, fontSize: fsz - 1, bold: true, color: STONE,
    charSpacing: 1, border: [NOL, NOL, LINE, NOL] } }));
  const body = rows.map((r, ri) => r.map((c, ci) => {
    const o = typeof c === "string" ? { text: c } : c;
    return { text: o.text, options: { fontFace: o.font ?? BODY, fontSize: o.size ?? fsz, color: o.color ?? INK,
      bold: o.bold ?? (ci === 0 && opt.boldFirst), border: [NOL, NOL, LINE, NOL], fill: o.fill ? { color: o.fill } : undefined } };
  }));
  s.addTable([hdr].concat(body), { x, y, w, colW, valign: "middle", margin: [3, 5, 3, 5],
    rowH: opt.rowH, objectName: nm("fade", "table") });
}

pres.addSection({ title: "Opening" });
let s = newSlide("FOREST", "Opening");
s.addShape(pres.ShapeType.rect, { x: 6.2, y: 0, w: 7.5 * AR("terrain_title.png"), h: 7.5, fill: { color: FOREST },
  line: { type: "none" }, shadow: shadow({ blur: 24, offset: 8, angle: 180, opacity: 0.55 }), objectName: nm("fade", "terrain_shadow") });
s.addImage({ path: A("terrain_title.png"), x: 6.2, y: 0, w: 7.5 * AR("terrain_title.png"), h: 7.5, objectName: nm("zoom", "terrain") });
s.addText("EVST · TOPIC 11 · FINAL EVALUATION", { x: 0.75, y: 1.2, w: 5.3, h: 0.3, fontFace: LABEL, fontSize: 12,
  color: APRICOT, charSpacing: 4, margin: 0, isTextBox: true, objectName: nm("fade", "eyebrow") });
s.addText("Where the\nroads end", { x: 0.75, y: 1.6, w: 5.3, h: 2.1, fontFace: HEAD, fontSize: 60, color: PAPER,
  margin: 0, valign: "top", isTextBox: true, objectName: nm("up", "title") });
s.addText("Can AI identify biodiversity patterns missing from conventional sampling? Bias-corrected models tested against independent bird surveys in Kerala and Taiwan.",
  { x: 0.75, y: 3.8, w: 5.1, h: 1.4, fontFace: BODY, fontSize: 17, color: SAGE, margin: 0, valign: "top",
    isTextBox: true, objectName: nm("fade", "subtitle") });
s.addText([{ text: "Ponnambalam V · Nandini C · Mohit Nallagatla · Manali Gupta", options: { breakLine: true } },
  { text: "November 2026", options: { fontFace: LABEL, fontSize: 11, color: SAGE, charSpacing: 2 } }],
  { x: 0.75, y: 5.55, w: 5.3, h: 0.8, fontFace: BODY, fontSize: 14, color: PAPER, margin: 0, valign: "top",
    isTextBox: true, objectName: nm("fade", "team") });
leafTR(s, 4.2, 0.1, 0);
leafBR(s, 5.4, 0.2, 0.15);
s.addNotes("SPEAKER 1 (about 30 s). Biodiversity maps are built from records that people chose to collect, and people go where roads go. Our topic asks whether AI-predicted maps can show what those records miss once that bias is accounted for. At the mid-evaluation the answer looked like 'not yet'. Since then we added two independent bird surveys as ground truth, and the answer changed. This talk covers the hypotheses, the data, the experiments, and what held up.");

s = newSlide("PAPER", "Opening");
heading(s, "THE QUESTION", "One topic question, broken into testable claims");
card(s, 0.75, 1.6, 4.6, 5.15, "qcard");
s.addText([
  { text: "TOPIC STATEMENT", options: { fontFace: LABEL, fontSize: 10, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Do environmentally predicted biodiversity patterns reveal potentially important areas that are poorly represented in conventional observations? The analysis explicitly accounts for spatial sampling bias before ecological interpretation.",
    options: { fontFace: BODY, fontSize: 13, italic: true, color: INK, breakLine: true, paraSpaceAfter: 12 } },
  { text: "THE HIDDEN PREREQUISITE", options: { fontFace: LABEL, fontSize: 10, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 6 } },
  { text: "A model trained on biased records can learn where observers go and present it as where species live, and still score well. Before trusting a predicted gap, we have to show the method is not reproducing observer bias, and that our test would notice if it were.",
    options: { fontFace: BODY, fontSize: 13, color: INK } }],
  { x: 1.0, y: 1.8, w: 4.15, h: 4.8, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "qtext") });
table(s, 5.75, 1.6, 7.05, [0.55, 4.6, 1.9], ["", "Hypothesis", "Experiments"], [
  [{ text: "H1", color: ORANGE, bold: true }, "Records are biased toward roads, and the bias is measurable", "E01"],
  [{ text: "H2", color: ORANGE, bold: true }, "Flexible models absorb more bias than a linear model", "E03"],
  [{ text: "H3", color: ORANGE, bold: true }, "Random cross-validation misses the damage and picks the wrong model", "E02, E03, E09b, E17"],
  [{ text: "H4", color: ORANGE, bold: true }, "Corrections can be ranked where the truth is known", "E02, E03"],
  [{ text: "H5", color: FOREST, bold: true }, { text: "Corrected models find real, unrecorded occurrences (the topic question)", bold: true }, "E04, E09b, E17"],
  [{ text: "H6", color: STONE, bold: true }, "A Bayesian model of sampling forecasts where new records appear", "E05"],
  [{ text: "H7", color: STONE, bold: true }, "Later citizen-science records and a survey disagree on the same pairs", "E15"],
  [{ text: "H8", color: STONE, bold: true }, "Endemic and montane species gain most from the correction", "E15, E17"],
  [{ text: "S0", color: STONE, bold: true }, "The survey truth survives a model of missed detections", "E16, E17b"],
], { size: 12, rowH: 0.5 });
s.addText("H1 to H4: can the method be trusted?   H5: the answer.   H6 onward: why it works, and how robust it is.",
  { x: 5.75, y: 6.62, w: 7.05, h: 0.35, fontFace: BODY, fontSize: 12, italic: true, color: STONE, margin: 0, isTextBox: true,
    objectName: nm("fade", "legend") });
s.addNotes("SPEAKER 1 (1.5 min). We split the topic into a chain. H1 to H4 ask whether the method can be trusted: is the bias measurable, do flexible AI models absorb it, does the usual check notice, and can corrections be ranked. H5 is the topic question itself. After the mid-evaluation we added hypotheses that explain the results and test how robust they are: a Bayesian model of sampling, the role of the truth source, which species gain, and whether the survey truth itself can be trusted.");

s = newSlide("PAPER", "Opening");
heading(s, "LITERATURE", "What was known, what was missing, what we added");
table(s, 0.75, 1.55, 11.85, [2.05, 5.0, 4.8], ["Theme", "What the literature established", "What we added"], [
  [{ text: "Bias exists", bold: true }, "Records cluster near roads and towns (Hughes 2021, Meyer 2015); volume outpaces new knowledge (Oliver 2021)", "Measured in the Western Ghats with the land-area null that the road statistic lacks; per-taxon sampbias weights checked against the R package"],
  [{ text: "Corrections", bold: true }, "Target-group background (Phillips 2009); pooling with surveys (Fithian 2015); comparisons mostly on virtual species (Inman 2021)", "Ranked corrections on real survey truth in two tropical/subtropical bird systems"],
  [{ text: "AI models", bold: true }, "Flexible models win benchmarks (Elith 2006, Valavi 2022); CNNs (Deneu 2021), SINR (Cole 2023), DeepMaxent (2025)", "Tested SINR- and DeepMaxent-style losses against tropical survey truth; proposed an effort-aware likelihood"],
  [{ text: "Validation", bold: true }, "Spatial blocking removes autocorrelation optimism (Ploton 2020, Valavi 2019); internal checks cannot judge corrections, shown mostly in simulation (El-Gabbas 2018, Baker 2024)", "Real-data demonstration on identical cell-species pairs, in two systems: later citizen-science records reverse the survey's verdict"],
  [{ text: "Detection", bold: true }, "Occupancy models (MacKenzie 2002); beta-binomial richness (Chiu 2022)", "Occupancy audit of the survey truth itself; spatial beta-binomial forecast of new records"],
], { size: 12, rowH: 0.85 });
s.addNotes("SPEAKER 1 (1.5 min). Our literature review was organised by theme. Each row says what was already established and what we added. The key gap: almost everything about whether bias corrections work, and whether internal checks can tell, comes from simulations with virtual species, because real data rarely include an unbiased test set. Baker and colleagues in 2024 and El-Gabbas and Dormann in 2018 argued that internal evaluation cannot judge a correction, but showed it mainly in simulation. We bring two independent systematic bird surveys, which lets us test these claims on real data.");

pres.addSection({ title: "Data and methods" });
s = newSlide("PAPER", "Data and methods");
heading(s, "DATA", "Five test beds, five ways to know the truth");
fig(s, "F1_study_systems.png", 0.7, 1.5, { h: 5.25 });
table(s, 8.35, 1.5, 4.5, [1.35, 1.6, 1.55], ["Test bed", "Training data", "Truth"], [
  [{ text: "T1 Virtual species", bold: true }, "30 species, real Ghats layers, real road bias at 3 strengths", "known by construction"],
  [{ text: "T2 disdat", bold: true }, "226 species, 6 regions, 52,605 records", "53,044 survey sites"],
  [{ text: "T3 Western Ghats", bold: true }, "493,227 iNaturalist records, 10,219 species", "records after the training year"],
  [{ text: "T4 Kerala Bird Atlas", bold: true }, "iNaturalist birds to 2020", "844 cells, timed 15-min lists, 2015-20"],
  [{ text: "T5 Taiwan BBS", bold: true }, "67,819 iNaturalist bird records to 2022", "420 sites, ~77 point counts each"],
], { size: 10, rowH: 0.62 });
tag(s, 8.6, 5.45, 4.15, 1.3, "FIELD NOTE · WHY TWO SURVEYS", [["", "Phases B-D: the atlas gives Indian ground truth; Taiwan is the replication. The US survey was behind a bot check, eBird access was not available."]], -1);
s.addNotes("SPEAKER 1 (1.5 min). Every claim needs a source of truth. T1, virtual species: we invent them on real Western Ghats layers, so the truth is exact. T2, disdat: the benchmark behind Elith 2006 and Phillips 2009, with independent absence surveys at 53,000 sites. T3, the Western Ghats itself: half a million iNaturalist records of ten thousand species. The maps are the real data. Panel b is the Kerala Bird Atlas: 844 cells of 6.6 km, each surveyed with up to 32 timed checklists by trained volunteers. Colour is the share of atlas species iNaturalist never recorded in that cell: nearly everything is dark, and the white dots are the 435 cells with no iNaturalist bird visit at all. Panel c is Taiwan, our replication: iNaturalist records in orange, 420 Breeding Bird Survey sites in blue.");

s = newSlide("PAPER", "Data and methods");
heading(s, "INTEGRATION", "How the components fit together");
const cols = [
  ["SOURCES", ["iNaturalist open data (India, Taiwan)", "WorldClim 2.1: 19 climate layers + elevation", "ESA WorldCover 10 m: 9 land-cover fractions", "WorldPop 2020 · OpenStreetMap roads, towns, reserves", "Surveys: Kerala Bird Atlas · Taiwan BBS · disdat"]],
  ["PROCESSING", ["Cleaning: species rank, \u2264 5 km uncertainty, hotspot and centroid filters", "1 km grid: 29 habitat + 5 access layers per cell", "Visits = observer \u00d7 day", "Names joined to atlas table / GBIF backbone", "Survey cells linked to 1 km pixels"]],
  ["MODELS", ["GLM, MaxEnt, boosted trees, random forest", "MLP, patch CNN, multi-species neural losses (GPU)", "Effort-aware occupancy likelihood", "Bayesian discovery model (beta-binomial)", "Occupancy model of the surveys (checks the truth)"]],
  ["EVALUATION", ["Random and spatial-block CV", "Later citizen-science records", "Missing-pair protocol vs survey", "Same-pair truth-source test", "Spatial-block bootstrap, Wilcoxon, pre-registered predictions"]],
];
cols.forEach(([title, items], i) => {
  const x = 0.75 + i * 3.08, w = 2.78;
  card(s, x, 1.65, w, 3.7, "pipe" + i, "left");
  s.addText(title, { x: x + 0.15, y: 1.75, w: w - 0.3, h: 0.35, fontFace: LABEL, fontSize: 11, bold: true, color: i === 3 ? BLUE : ORANGE,
    charSpacing: 3, margin: 0, isTextBox: true, objectName: nm("left", "pipetitle") });
  s.addText(items.map((t, j) => ({ text: t, options: { bullet: { code: "25AA", indent: 12 }, breakLine: j < items.length - 1, paraSpaceAfter: 6 } })),
    { x: x + 0.15, y: 2.15, w: w - 0.3, h: 3.1, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0, valign: "top",
      isTextBox: true, objectName: nm("left", "pipeitems") });
  if (i < 3) s.addShape(pres.ShapeType.rightArrow, { x: x + w + 0.02, y: 3.3, w: 0.28, h: 0.4, fill: { color: ORANGE },
    line: { type: "none" }, objectName: nm("fade", "arrow") });
});
const tiles = [["tile_rain.png", "RAINFALL"], ["tile_tree.png", "TREE COVER"], ["tile_people.png", "PEOPLE"], ["tile_roads.png", "ROAD DISTANCE"]];
tiles.forEach(([f, lab], i) => {
  const h = 1.3, w = h * AR(f), x = 0.95 + i * 1.0;
  s.addImage({ path: A(f), x, y: 5.6, w, h, rotate: [-2, 1.5, -1.5, 2][i], objectName: nm("up", "tile") });
  s.addText(lab, { x: x - 0.25, y: 6.92, w: w + 0.5, h: 0.2, fontFace: LABEL, fontSize: 7.5, color: STONE, align: "center",
    margin: 0, isTextBox: true, objectName: nm("fade", "tilelab") });
});
s.addText([B("Habitat vs access. "), T("Each cell carries what species respond to (climate, land cover) and what observers respond to (people, roads). The real layers are shown at left. "),
  B("One codebase, two systems: "), T("a system switch builds identical inputs for Kerala and Taiwan, so every experiment runs unchanged on both.")],
  { x: 5.3, y: 5.6, w: 7.5, h: 1.3, fontFace: BODY, fontSize: 13, color: INK, margin: 0, valign: "top", isTextBox: true,
    objectName: nm("fade", "pipenote") });
s.addNotes("SPEAKER 1 (1.5 min). This is the system we built, in Python with one R cross-check. Sources on the left: iNaturalist for records, WorldClim, ESA WorldCover, WorldPop and OpenStreetMap for covariates, and the three survey datasets for truth. Processing puts everything on a 1 km grid with 29 habitat layers and 5 access layers per cell, defines a visit as one observer on one day, and harmonises species names. Models range from a GLM to boosted trees to neural networks trained for all species at once on the GPU, plus the Bayesian and occupancy models. Evaluation is the core: we judge the same models with four different tests. A single switch builds identical inputs for Kerala and Taiwan, so the replication used exactly the same code.");

s = newSlide("PAPER", "Data and methods");
heading(s, "H1 · EXPLORATORY ANALYSIS", "The records follow the roads, and more records add less");
fig(s, "F2_bias_anatomy.png", 0.7, 1.5, { w: 8.4 });
bullets(s, 0.85, 4.55, 8.2, 2.3, [
  [B("Intensity, not proximity. "), T("99.4% of records lie within 2.5 km of a road, but so does 94.9% of the land. The useful measure divides share of records by share of land: 3.4\u00d7 within 100 m, 0.1\u00d7 beyond 2 km.")],
  [B("Coverage. "), T("Only 14.4% of 1 km cells hold any record; 1.4% for amphibians.")],
  [B("Diminishing returns. "), T("Records per year rose 61-fold since 2008; new species-by-cell finds per 1,000 records fell 56%.")],
], 13);
table(s, 9.45, 1.55, 3.4, [1.45, 0.65, 0.65, 0.65], ["sampbias decay /km", "road", "major", "PA"], [
  ["all records", "0.47", "0.17", "0.03"], ["birds", "0.63", "0.20", "0.02"], ["butterflies", "0.42", "0.16", "0.04"],
  ["amphibians", "0.26", { text: "0.35", bold: true, color: ORANGE }, "0.05"], ["mammals", "0.67", "0.01", { text: "0.08", bold: true, color: ORANGE }],
  ["reptiles", "0.25", "0.15", "0.05"],
], { size: 10.5, rowH: 0.36 });
s.addText("Bias is taxon-specific: amphibian records follow major roads, mammal records follow protected areas (PA). All six weights fall inside the R package's 95% intervals.",
  { x: 9.45, y: 4.35, w: 3.4, h: 1.2, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "sbnote") });
tag(s, 9.7, 5.75, 3.05, 1.0, "FIELD NOTE", [["DATA", "493,227 records · OSM roads at 100 m"]]);
s.addNotes("SPEAKER 2 (1.5 min). Exploratory analysis of the training data, which is H1. The often-quoted statistic, records within 2.5 km of a road, is uninformative here because 95% of the Western Ghats is within 2.5 km of a road. So we divided the share of records by the share of land in each band: land within 100 m of a road is recorded 3.4 times more than its area predicts, land beyond 2 km at a tenth. Recording volume grew 61-fold since 2008, but each record adds less new knowledge, a 56% drop. The table on the right is our Python port of the sampbias model, a per-kilometre decay of record density with distance. It shows bias differs by taxon: amphibian records follow major roads, mammal records follow protected areas. We validated the port against the original R package's MCMC.");

s = newSlide("PAPER", "Data and methods");
heading(s, "METHODS", "Models, the correction, and how we score them");
card(s, 0.75, 1.55, 5.6, 5.25, "mcard");
s.addText([
  { text: "PRESENCE-ONLY MODELS", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "A model contrasts pixels where a species was recorded (1) with background pixels (0). Capacity ladder: GLM \u2192 MaxEnt \u2192 boosted trees \u2192 random forest \u2192 MLP \u2192 patch CNN.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "THE CORRECTION", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Random background ", options: { bold: true } }, { text: "compares roadside records with random land, so \u201cnear a road\u201d looks like habitat. ", options: {} },
  { text: "Target-group background ", options: { bold: true, color: BLUE } }, { text: "draws the background from where anyone recorded a related species, so both sides share the observer bias.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "EFFORT-AWARE LIKELIHOOD (OURS)", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "P(recorded) = \u03c8 \u00d7 [1 \u2212 (1 \u2212 p)^U]", options: { fontFace: "Cambria Math", fontSize: 15, color: FOREST, breakLine: true, paraSpaceAfter: 3 } },
  { text: "\u03c8 occupancy, p detection per visit, U visits. A cell visited once and not recorded is weak evidence of absence; an unvisited cell is no evidence." }],
  { x: 1.0, y: 1.7, w: 5.15, h: 5.0, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "mtext") });
card(s, 6.7, 1.55, 6.1, 5.25, "ecard");
s.addText("MISSING-PAIR PROTOCOL", { x: 6.95, y: 1.7, w: 5.6, h: 0.3, fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE,
  charSpacing: 3, margin: 0, isTextBox: true, objectName: nm("fade", "eptitle") });
const grid = [[1, 0, 0, 1, 0, 0, 2, 0], [0, 2, 1, 0, 0, 2, 0, 0], [0, 0, 0, 2, 1, 0, 0, 2], [2, 0, 0, 0, 0, 1, 2, 0], [0, 1, 2, 0, 0, 0, 0, 0]];
grid.forEach((row, r) => row.forEach((v, c) => {
  const x = 7.0 + c * 0.36, y = 2.15 + r * 0.36;
  s.addShape(pres.ShapeType.rect, { x, y, w: 0.31, h: 0.31, fill: { color: v === 1 ? ORANGE : "E8E6DD" },
    line: { color: v === 2 ? BLUE : "D5D2C6", width: v === 2 ? 2 : 0.5 }, objectName: nm("fade", "cell") });
  if (v === 2) s.addShape(pres.ShapeType.ellipse, { x: x + 0.09, y: y + 0.09, w: 0.13, h: 0.13, fill: { color: BLUE }, line: { type: "none" }, objectName: nm("fade", "dot") });
}));
s.addText("species \u2192", { x: 7.0, y: 4.0, w: 2.9, h: 0.25, fontFace: LABEL, fontSize: 9, color: STONE, margin: 0, isTextBox: true, objectName: nm("fade", "axl") });
s.addText("cells \u2193", { x: 6.95, y: 1.95, w: 1, h: 0.2, fontFace: LABEL, fontSize: 9, color: STONE, margin: 0, isTextBox: true, objectName: nm("fade", "axl2") });
s.addText([{ text: "\u25A0 ", options: { color: ORANGE } }, T("recorded by citizen science (excluded)"), { text: "", options: { breakLine: true } },
  { text: "\u25A0 ", options: { color: "C9C6BA" } }, T("never recorded: the pairs we score"), { text: "", options: { breakLine: true } },
  { text: "\u25CF ", options: { color: BLUE } }, T("the survey found the species there")],
  { x: 10.05, y: 2.15, w: 2.65, h: 1.75, fontFace: BODY, fontSize: 11, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "gridlegend") });
s.addText([T("Among the places citizen science never recorded a species, does the model rank the places where the survey found it above the places where it did not? Scored as per-species AUC."),
  { text: "", options: { breakLine: true } }, { text: "", options: { breakLine: true } },
  { text: "SAME-PAIR TEST  ", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3 } },
  T("Train up to year T. Judge the same unrecorded pairs twice: by the survey, and by whether citizen science recorded the species there later.")],
  { x: 6.95, y: 4.35, w: 5.65, h: 2.35, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "eptext") });
s.addNotes("SPEAKER 2 (1.5 min). Left: the models and the correction. A presence-only model compares places where a species was recorded with background places. With a random background, roadside records are compared with random land, so 'near a road' looks like habitat. Target-group background, from Phillips 2009, takes the background from places where people recorded related species, so both sides of the comparison share the same observer bias. For the neural models we also wrote an effort-aware likelihood: the chance of a record is occupancy times the chance of detecting the species in U visits. Right: our main evaluation. The grid is cells by species. Orange pairs were recorded by citizen science; we throw those away. We score models only on the grey pairs, never recorded, against the survey (blue dots). That directly asks: does the model find what citizen science missed? The same-pair test judges identical pairs with two different truths.");

pres.addSection({ title: "Results" });
s = newSlide("PAPER", "Results");
heading(s, "H2-H4 · KNOWN TRUTH", "With known truth: more bias, worse maps, better CV scores");
fig(s, "F3_virtual_species.png", 0.7, 1.5, { w: 11.9 });
bullets(s, 0.85, 5.35, 11.6, 1.6, [
  [B("a  "), T("As observer bias grows, true AUC falls (0.79 \u2192 0.71) while random-CV AUC rises (0.74 \u2192 0.92): a better CV score means more bias absorbed.")],
  [B("b  "), T("Every flexible model leaks more bias into its map than the GLM (p \u2264 0.003). Boosted trees leak most, not the CNN, so the problem is flexibility, not depth.")],
  [B("c  "), T("Target-group background is the truly best correction for 77-90% of species; thinning barely helps. Random CV instead picks bias-covariate conditioning for 83-97%, the method that models the bias rather than removing it.")],
], 13);
s.addNotes("SPEAKER 2 (1.5 min). Virtual species are the controlled experiment. We invented 30 species on the real Western Ghats layers and recorded them through the real road bias at three strengths. Panel a is the key pattern: as bias increases, the true accuracy of uncorrected models falls but their cross-validation accuracy rises, because the held-out records share the bias. Panel b: bias leakage, the share of the predicted map explained by access layers beyond true suitability. All flexible models leak more than a GLM, and boosted trees leak most. Panel c: among corrections, target-group background is best against the truth. Random CV picks a different correction, bias-covariate conditioning, because that model uses accessibility directly and the biased test records reward it. So H2, H3 and H4 hold with known truth.");

s = newSlide("PAPER", "Results");
heading(s, "H3 · THREE REAL SYSTEMS", "On real data, random cross-validation rejects the correction");
fig(s, "F4_cv_inversion.png", 0.7, 1.5, { w: 11.2 });
table(s, 0.75, 5.7, 7.4, [2.8, 1.5, 1.55, 1.55], ["System", "Random CV prefers fix", "Survey prefers fix", "CV optimism, naive"], [
  ["disdat, 896 species-model fits", "4.5%", "62.0%", "+0.02 to +0.33"],
  ["Kerala, 196 species", "1.0%", "75.5%", "+0.109"],
  ["Taiwan, 131 species", "0.8%", "82.4%", "+0.073"],
], { size: 11, rowH: 0.3 });
bullets(s, 8.5, 5.65, 4.3, 1.4, [
  [B("Blocking does not fix it. "), T("Bigger spatial blocks shrink the optimism (+0.076 \u2192 +0.018) but agreement with truth stays at 43-44%.")],
  [B("One internal check works: "), T("test against held-out target-group sites (62% agreement).")],
], 12);
s.addNotes("SPEAKER 2 (1.5 min). Now real data with independent truth, in three systems. Panel a: in disdat, the surveys say the correction helps in 62% of fits; random CV says so in 4.5%. In Kerala it is 1% versus 75.5%, and in Taiwan 0.8% versus 82.4%. Panels b and c show every species: nearly all fall left of zero on the x axis, so CV says the correction hurts, while most are above zero, so the survey says it helps. Random CV overstates the uncorrected model by about 0.1 AUC but is nearly unbiased for the corrected one. Spatial blocking, the standard remedy, shrinks the optimism but does not change which model gets picked, because the held-out records still carry the observer bias. The only internal check that tracked the truth scored held-out records against held-out target-group sites.");

s = newSlide("PAPER", "Results");
heading(s, "H5 · THE TOPIC QUESTION", "Corrected models find the birds citizen science missed");
fig(s, "F5_core.png", 0.7, 1.5, { w: 7.6 });
table(s, 8.65, 1.55, 4.2, [2.0, 1.1, 1.1], ["Unrecorded pairs", "Kerala", "Taiwan"], [
  ["pairs scored", "170,286", "58,551"],
  ["survey presence (base rate)", "27.5%", "17.0%"],
  [{ text: "BRT target-group, AUC", bold: true }, { text: "0.758", bold: true }, { text: "0.820", bold: true }],
  ["gain vs random background", "+0.022", "+0.024"],
  ["  95% spatial-block CI", "[0.015, 0.028]", "[0.015, 0.035]"],
  ["gain vs nearest record", "+0.098", "+0.101"],
  [{ text: "hit rate in top 10%", bold: true }, { text: "48.4%", bold: true }, { text: "40.2%", bold: true }],
  ["  random background", "47.0%", "36.5%"],
  ["  nearest record", "41.3%", "29.3%"],
], { size: 11, rowH: 0.36 });
bullets(s, 0.85, 4.5, 7.5, 2.0, [
  [B("The gap is large. "), T("52% of atlas cells have no iNaturalist bird record; where iNaturalist did visit, it recorded 12% of the atlas species.")],
  [B("Best in both systems: "), T("boosted trees with target-group background; the effort-aware network ties it in Taiwan.")],
  [B("Endemics gain most "), T("in Kerala: 0.834 vs 0.740 for nearest record.")],
], 12.5);
s.addNotes("SPEAKER 3 (1.5 min). This is the direct answer to the topic question. In Kerala there are 170 thousand cell-species pairs that iNaturalist never recorded, and the atlas found the species in 27.5% of them. Bias-corrected boosted trees rank those pairs at AUC 0.758 in Kerala and 0.820 in Taiwan, beating the uncorrected model in about four of five species and the nearest-record baseline by about 0.1. The confidence intervals resample whole spatial blocks of survey cells, so they account for spatial autocorrelation. In practical terms: if a surveyor visits the 10% of unrecorded pairs the model ranks highest, the species is there 48% of the time in Kerala, against 27.5% at random; 40% against 17% in Taiwan.");

s = newSlide("PAPER", "Results");
heading(s, "H7 · WHY THE MID-EVALUATION SAID \u201cNOT YET\u201d", "Later citizen-science records reverse the survey\u2019s verdict");
card(s, 0.75, 1.55, 3.7, 5.25, "midcard");
s.addText([
  { text: "AT THE MID-EVALUATION", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 5 } },
  { text: "Correction moved predicted hotspots from built-up, heavily recorded land into forest (92% vs 78% tree cover; 127 vs 321 records per cell).", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "But 2019-2026 records did not appear there beyond effort: \u03b2 = +0.005 per SD (95% CI \u00b1 0.03).", options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: "THE TEST", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 5 } },
  { text: "Same models, same pairs, two truths. The survey rewards the corrected model every time; later citizen-science records penalise it every time: 2 cut-offs per system, 2 model families.", options: {} }],
  { x: 0.95, y: 1.7, w: 3.3, h: 5.0, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "midtext") });
fig(s, "S_flip.png", 4.75, 1.55, { w: 8.0 });
s.addText([B("Why: "), T("later records come from the same observers, so they reward the model that learned where observers go. Forward-in-time validation with citizen-science data, often recommended as the fix for optimistic CV, gives the wrong answer.")],
  { x: 4.85, y: 5.45, w: 7.9, h: 1.3, fontFace: BODY, fontSize: 13, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "flipnote") });
s.addNotes("SPEAKER 3 (1.5 min). At the mid-evaluation, the corrected maps did not predict where new iNaturalist records appeared, and we reported that as an honest null. The survey data explain it. We trained on records up to a cut-off year and took the pairs unrecorded at that point, in cells that iNaturalist visited again. Each pair then has two labels: did the survey find the species, and did iNaturalist later record it. Blue bars are the gain from the correction under survey truth, orange under later iNaturalist. Same models, same pairs, opposite verdicts, at two cut-offs in Kerala and two in Taiwan, for boosted trees and for the neural model. The error bars resample spatial blocks; the two truths never overlap. Later records are made by the same observers, so they reward the model that learned observer behaviour.");

s = newSlide("PAPER", "Results");
heading(s, "ROBUSTNESS · IS THE TRUTH TRUE?", "An occupancy model of the survey strengthens the result");
fig(s, "S_occupancy.png", 0.7, 1.5, { w: 8.0 });
card(s, 9.15, 1.55, 3.65, 5.25, "occcard");
s.addText([
  { text: "THE CONCERN", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "A survey \u201cabsence\u201d may be a missed bird. We fitted an occupancy model per species and season, with the atlas lists as repeat visits.", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "CHECKED FIRST", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "On simulated histories it recovers occupancy to 0.011 and calibrates false absences (0.071 vs 0.073).", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "FINDINGS", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Per-list detection is only 0.08, so a third of atlas absences are doubtful. Dropping them doubles the gain (+0.022 \u2192 +0.043). Model rankings never change (Kendall \u03c4 = 1).", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "Side finding: the \u201cdetection\u201d learned from iNaturalist runs opposite to real detectability (Spearman \u22120.54).", options: { italic: true } }],
  { x: 9.35, y: 1.7, w: 3.3, h: 5.0, fontFace: BODY, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "occtext") });
bullets(s, 0.85, 5.3, 7.9, 1.5, [
  [B("Taiwan is cleaner: "), T("about 77 point counts per site, so only 6.5% of its absences are doubtful.")],
  [B("Why the gains rise: "), T("missed presences sit where species are likely to occur, so treating the atlas as perfect was penalising the good models.")],
], 12.5);
s.addNotes("SPEAKER 3 (1.5 min). A reviewer would ask: are the atlas absences real? Each atlas cell was surveyed with many lists, so we can fit a standard occupancy model that separates 'not there' from 'there but missed'. Before trusting it we simulated data with known values on the real list structure and checked it recovers them. The answer is that the atlas is noisier than we assumed: a single list detects a typical species only 8% of the time, and a third of atlas absences have more than a 10% chance of being missed birds. But every result survives, and the Kerala gains get larger: dropping doubtful absences doubles the gain from the correction. Model rankings are identical under every truth variant, including wet against dry season. A side result: the per-visit detection a model learns from iNaturalist is negatively correlated with real detectability, so it measures what people photograph, not what birds can be detected.");

s = newSlide("PAPER", "Results");
heading(s, "H8 · WHO GAINS?", "Endemics gain in both systems; only elevation replicates");
fig(s, "F7_endemic_elevation.png", 0.7, 1.5, { h: 5.3 });
card(s, 9.4, 1.55, 3.4, 5.25, "endcard");
s.addText([
  { text: "KERALA", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Endemics gain twice as much (+0.040 vs +0.020). Once exposure, the share of a species' range iNaturalist never visited, enters the model, the endemic effect vanishes (+0.020 \u2192 +0.005, p = 0.52).", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "TAIWAN", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Taiwan-concentrated species also gain more (+0.033, permutation p = 0.0005), yet they are less exposed (0.34 vs 0.47), so that explanation is Kerala-specific.", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "WHAT REPLICATES", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Higher-elevation ranges gain more: +0.031 per km in Kerala, +0.022 to +0.031 in Taiwan (multiple regression).", options: {} }],
  { x: 9.6, y: 1.7, w: 3.05, h: 5.0, fontFace: BODY, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "endtext") });
s.addNotes("SPEAKER 3 (1.5 min). Which species benefit most from the correction? Each point is a species; orange are endemics. In Kerala, endemics gain twice as much, and the reason looked clear: their ranges sit in cells iNaturalist never visits. Once that 'exposure' is in the regression, the endemic effect disappears. This matches Botella's theory that target-group background helps species whose ranges run against sampling effort. We predicted Taiwan would show the same. It did not: Taiwan's concentrated species gain more but are less exposed. So we present the exposure mechanism as specific to Kerala. What does replicate is elevation: species of higher, montane ranges gain more in both systems. Our reading is that mountains are where a random-background model most often confuses 'rarely visited' with 'unsuitable'.");

s = newSlide("PAPER", "Results");
heading(s, "H6 · A BAYESIAN MODEL OF SAMPLING", "Modelling the observers forecasts discovery better than habitat");
fig(s, "F8_discovery.png", 0.7, 1.5, { w: 8.6 });
card(s, 9.75, 1.55, 3.05, 5.25, "bcard");
s.addText([
  { text: "THE MODEL", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Each 5 km cell has a Poisson pool of species; each is detected per visit with p ~ Beta(a, b). Still undetected after U visits:", options: { breakLine: true, paraSpaceAfter: 3 } },
  { text: "P\u2080(U) = B(a, b+U) / B(a, b)", options: { fontFace: "Cambria Math", fontSize: 13.5, color: FOREST, breakLine: true, paraSpaceAfter: 3 } },
  { text: "New finds in m more visits ~ Poisson(\u039b[P\u2080(U) \u2212 P\u2080(U+m)]). Extends Chiu 2022 to a spatial forecast.", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "LIMITS", options: { fontFace: LABEL, fontSize: 10.5, bold: true, color: ORANGE, charSpacing: 3, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Fails in Europe (GeoLifeCLEF), where records lack observer IDs. Does not estimate true missing richness.", options: {} }],
  { x: 9.95, y: 1.7, w: 2.7, h: 5.0, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "btext") });
bullets(s, 0.85, 5.35, 8.6, 1.5, [
  [B("Wins at every origin "), T("(+0.020 to +0.054 nats per cell); classical Shen-Chao extrapolation scores \u22124 to \u22125.")],
  [B("The gain comes from the detection-and-saturation structure: "), T("the intercepts-only version matches the full model. Habitat covariates help only in ranking cells.")],
], 12.5);
s.addNotes("SPEAKER 4 (1.5 min). The richness maps did not predict new records, so we asked whether a model of the sampling process would. The discovery model treats each cell as a pool of species, each with a per-visit detection probability drawn from a beta distribution. That gives a closed form for how many new species further visits should find. It beats the effort regression at every forecast origin, with spatial-block intervals above zero, and classical Chao-type extrapolation fails badly. The ablation is the interesting part: removing habitat covariates does not hurt. Knowing how much a cell has already been searched predicts discovery; predicted habitat adds little. The map on the right is the deliverable: expected new records from the next ten visits per cell. Two limits: it needs observer-level effort, so it failed in Europe where visits could only be defined by publisher and day, and it should not be read as an estimate of true missing richness.");

pres.addSection({ title: "Close" });
s = newSlide("SAGE", "Close");
heading(s, "WHERE WE STAND", "What held up");
const V = (t, c) => ({ text: t, font: HEAD, size: 14, color: c });
table(s, 0.75, 1.5, 11.9, [0.85, 5.1, 2.25, 3.7], ["", "Hypothesis", "Verdict", "Key evidence"], [
  [{ text: "H1", bold: true, color: ORANGE }, "Records are measurably biased", V("Supported", FOREST), "3.4\u00d7 roadside intensity; sampbias matches R"],
  [{ text: "H2", bold: true, color: ORANGE }, "Flexible models absorb more bias", V("Supported", FOREST), "all flexible > GLM; trees leak most"],
  [{ text: "H3", bold: true, color: ORANGE }, "Random CV picks the wrong model", V("Supported, 3 systems", FOREST), "1% / 0.8% vs 75.5% / 82.4%"],
  [{ text: "H4", bold: true, color: ORANGE }, "Corrections can be ranked", V("Supported", FOREST), "target-group background best"],
  [{ text: "H5", bold: true, color: ORANGE }, { text: "Corrected models find unrecorded occurrences", bold: true }, V("Yes, vs surveys", FOREST), "AUC 0.758 / 0.820; hit rate \u00d71.8 / \u00d72.4"],
  [{ text: "H6", bold: true, color: STONE }, "Bayesian sampling model forecasts discovery", V("Supported*", FOREST), "* only with observer-level effort"],
  [{ text: "H7", bold: true, color: STONE }, "Truth source decides the verdict", V("Supported, 2 systems", FOREST), "opposite signs on identical pairs"],
  [{ text: "H8", bold: true, color: STONE }, "Endemic and montane species gain most", V("Supported, 2 systems", FOREST), "elevation replicates; exposure is Kerala-only"],
  [{ text: "S0", bold: true, color: STONE }, "Survey truth survives missed detections", V("Supported", FOREST), "gains double once doubtful absences go"],
], { size: 12.5, rowH: 0.5 });
leafBR(s, 3.0, 0.4, 0.4);
s.addNotes("SPEAKER 4 (1.5 min). The scorecard. H1 to H4, whether the method can be trusted, all hold, and H3 now holds on real data in three systems. H5, the topic question, is supported against both surveys. Of the later hypotheses, the Bayesian model works when effort is measured per observer, and the truth-source result replicates. Endemic and montane species gain most in both systems, although the exposure explanation holds only in Kerala. And the survey truth survives an explicit model of missed detections, which makes the gains larger.");

s = newSlide("FOREST", "Close");
s.addImage({ path: A("canopy.png"), x: 8.6, y: -0.75, w: 9.0 * AR("canopy.png"), h: 9.0, objectName: nm("zoom", "canopy") });
s.addText("CONCLUSIONS", { x: 0.75, y: 0.55, w: 5, h: 0.3, fontFace: LABEL, fontSize: 12, color: APRICOT, charSpacing: 4,
  margin: 0, isTextBox: true, objectName: nm("fade", "eyebrow") });
s.addText("Yes, if both the model and the test account for bias", { x: 0.75, y: 0.9, w: 7.6, h: 1.2, fontFace: HEAD, fontSize: 34,
  color: PAPER, margin: 0, valign: "top", isTextBox: true, objectName: nm("up", "title") });
const concl = [
  ["The answer", "Bias-corrected models find species citizen science never recorded, in two independent survey systems, with endemic and montane species gaining most."],
  ["What we add to the literature", "A missing-pair protocol; real-data proof, in two systems, that random CV and later citizen-science records reward observer bias; an occupancy audit of the truth; a discovery forecast built on observer effort."],
  ["Practical advice", "Use target-group background; never validate a correction with random CV or later records; get independent surveys, and model missed detections before trusting their absences."],
  ["Limits", "Birds only; Indian truth covers Kerala only; no eBird access; the Taiwan truth unit is a 2 km neighbourhood; one closely related study not yet read in full."],
];
s.addText(concl.flatMap(([k, v], i) => [{ text: k, options: { fontFace: LABEL, fontSize: 12, bold: true, color: APRICOT, charSpacing: 2, breakLine: true } },
  { text: v, options: { fontFace: BODY, fontSize: 14, color: PAPER, breakLine: i < concl.length - 1, paraSpaceAfter: 11 } }]),
  { x: 0.75, y: 2.2, w: 7.5, h: 4.3, margin: 0, valign: "top", isTextBox: true, objectName: nm("fade", "concl") });
s.addText("Thank you \u00b7 questions welcome", { x: 0.75, y: 6.55, w: 6, h: 0.45, fontFace: HEAD, fontSize: 21, italic: true,
  color: APRICOT, margin: 0, isTextBox: true, objectName: nm("fade", "thanks") });
s.addNotes("SPEAKER 4 (1 min). The answer to the topic question is yes, with a condition: both the model and its evaluation have to account for observer bias. Corrected models find species citizen science missed in two independent systems. What we add: a protocol that scores only the missed pairs, the first real-data demonstration on identical pairs that the common validation shortcuts reward observer bias, an audit of the survey truth, and a forecast of where new records will appear. The practical advice follows directly. The limits are real: birds only, Kerala only for India, and no eBird. Thank you.");

pres.writeFile({ fileName: path.join(__dirname, "deck_raw.pptx") }).then(async () => {
  await applyTheme(path.join(__dirname, "deck_raw.pptx"), THEME);
  console.log("written deck_raw.pptx");
});

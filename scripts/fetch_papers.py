import json, urllib.request, urllib.parse, pathlib, sys
PAPERS = {
 "Meyer2015": "10.1038/ncomms9221", "Hughes2021": "10.1111/ecog.05926", "Oliver2021": "10.1371/journal.pbio.3001336",
 "Zizka2021": "10.1111/ecog.05102", "Beck2014": "10.1016/j.ecoinf.2013.11.002", "Phillips2009": "10.1890/07-2153.1",
 "Fithian2015": "10.1111/2041-210X.12242", "Isaac2020": "10.1016/j.tree.2019.08.006", "Inman2021": "10.1002/ecs2.3422",
 "Elith2006": "10.1111/j.2006.0906-7590.04596.x", "Deneu2021": "10.1371/journal.pcbi.1008856",
 "Johnston2021": "10.1111/ddi.13271", "Ploton2020": "10.1038/s41467-020-18321-y", "Valavi2019": "10.1111/2041-210X.13107",
 "Hortal2015": "10.1146/annurev-ecolsys-112414-054400", "Valavi2022": "10.1002/ecm.1486",
 "Elith2020_disdat": "10.17161/bi.v15i2.13384", "Kramer-Schadt2013": "10.1111/ddi.12096", "Fourcade2014": "10.1371/journal.pone.0097122",
}
out = pathlib.Path("papers"); out.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; EVST course project)"}
for k, doi in PAPERS.items():
    f = out / f"{k}.pdf"
    if f.exists() and f.stat().st_size > 50_000: print("have", k); continue
    try:
        meta = json.load(urllib.request.urlopen(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email=evst.project@example.org", timeout=30))
        locs = [l for l in meta.get("oa_locations") or [] if l.get("url_for_pdf")]
        ok = False
        for l in locs:
            try:
                data = urllib.request.urlopen(urllib.request.Request(l["url_for_pdf"], headers=UA), timeout=60).read()
                if data[:4] == b"%PDF":
                    f.write_bytes(data); ok = True; print("ok", k, len(data)//1024, "KB", l["url_for_pdf"][:80]); break
            except Exception as e:
                pass
        if not ok: print("MISS", k, "oa" if meta.get("is_oa") else "closed", [l["url_for_pdf"][:70] for l in locs][:3])
    except Exception as e:
        print("ERR", k, e)

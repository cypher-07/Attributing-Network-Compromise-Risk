"""Fetch EPSS scores for every CVE in the catalog.

    python src/epss_fetch.py

EPSS (Exploit Prediction Scoring System, FIRST) gives P(exploited in the wild
within 30 days). Unlike CVSS it IS a probability, which is the whole point.
Scores are refreshed daily -- record the date you pulled them and ship the
cached file with your code so results stay reproducible.

No silent fallbacks. A batch that fails after retries aborts the whole run and
nothing is written: a half-fetched file would make unfetched CVEs look like
low-risk ones. CVEs that FIRST answers for but does not score (rejected,
reserved, too new) are listed explicitly under "_unscored" so scoring.py can
tell "FIRST has no score" apart from "we never asked".
"""
import json, sys, time, urllib.request, datetime
from paths import CATALOG, EPSS, rel

API = "https://api.first.org/data/v1/epss"
BATCH = 100


def fetch(cves):
    out, model, score_date = {}, set(), set()
    for i in range(0, len(cves), BATCH):
        chunk = cves[i:i + BATCH]
        url = f"{API}?cve={','.join(chunk)}&pretty=false"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.load(r)
                break
            except Exception as e:
                if attempt == 2:
                    sys.exit(f"batch {i//BATCH} failed after 3 attempts: {e}\n"
                             "nothing written -- rerun when the API is reachable")
                time.sleep(2 ** attempt)
        if data.get("status-code", 200) != 200:
            sys.exit(f"batch {i//BATCH}: API returned {data.get('status-code')} "
                     f"{data.get('status')} -- nothing written")
        for row in data.get("data", []):
            out[row["cve"]] = float(row["epss"])
            if "date" in row:
                score_date.add(row["date"])
        if "model_version" in data:
            model.add(data["model_version"])
        print(f"  {min(i+BATCH, len(cves))}/{len(cves)}", file=sys.stderr)
        time.sleep(0.5)
    return out, sorted(model), sorted(score_date)


if __name__ == "__main__":
    cat = json.load(open(CATALOG))
    cves = sorted({v["cve"] for e in cat.values() for v in e["vulnerabilities"]})
    print(f"{len(cves)} unique CVEs", file=sys.stderr)
    scores, model, score_date = fetch(cves)
    unscored = [c for c in cves if c not in scores]
    print(f"scored {len(scores)}, unscored by FIRST {len(unscored)}", file=sys.stderr)
    out = EPSS
    json.dump({"_fetched": datetime.date.today().isoformat(),
               "_score_date": score_date,
               "_model_version": model,
               "_n_requested": len(cves),
               "_unscored": unscored,
               **scores},
              open(out, "w"), indent=1)
    print(f"wrote {rel(out)}", file=sys.stderr)

"""Compare every cached EPSS score with FIRST's live API. Never writes the cache.

    python src/verify_epss.py          (needs internet; about 1 minute)

EPSS is refreshed daily, so small differences are normal when the live date is
later than the cached date. Large differences or missing CVEs are the thing to
look for. Writes results/epss_verification.csv and prints a summary.
"""
import csv, json, os, sys, time, urllib.request
from paths import EPSS, RESULTS

API = os.environ.get("EPSS_API", "https://api.first.org/data/v1/epss")
BATCH = 100


def fetch_live(cves):
    live, dates = {}, set()
    for i in range(0, len(cves), BATCH):
        chunk = cves[i:i + BATCH]
        url = f"{API}?cve={','.join(chunk)}&limit={BATCH}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.load(r)
                break
            except Exception as e:
                if attempt == 2:
                    sys.exit(f"batch {i // BATCH} failed after 3 attempts: {e}\n(nothing written)")
                time.sleep(2 ** attempt)
        for row in data.get("data", []):
            live[row["cve"]] = float(row["epss"]); dates.add(row.get("date"))
    return live, sorted(d for d in dates if d)


def compare(cached, live, tol=0.01):
    rows = []
    for c in sorted(cached):
        lv = live.get(c)
        rows.append({"cve": c, "cached": cached[c], "live": lv,
                     "abs_diff": None if lv is None else abs(lv - cached[c]),
                     "status": "missing live" if lv is None else ("differs" if abs(lv - cached[c]) > tol else "ok")})
    return rows


if __name__ == "__main__":
    raw = json.load(open(EPSS))
    cached = {k: v for k, v in raw.items() if not k.startswith("_")}
    live, dates = fetch_live(sorted(cached))
    rows = compare(cached, live)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "epss_verification.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["status"] != "ok"]
    diffs = [r["abs_diff"] for r in rows if r["abs_diff"] is not None]
    print(f"cached snapshot: {raw.get('_score_date')}   live dates: {dates}")
    print(f"{len(rows)} CVEs compared: {len(rows) - len(bad)} within 0.01, "
          f"{sum(r['status'] == 'differs' for r in rows)} differ, "
          f"{sum(r['status'] == 'missing live' for r in rows)} missing live; "
          f"max abs diff {max(diffs):.4f}" if diffs else "no live scores returned")
    for r in sorted(bad, key=lambda r: -(r['abs_diff'] or 9))[:10]:
        print("  ", r["cve"], r["cached"], r["live"], r["status"])

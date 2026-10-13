"""Per-node compromise susceptibility, driven by EPSS.

    p_j = P_FLOOR + (P_CEIL - P_FLOOR) * max(EPSS over node's attack CVEs)

attack CVEs = NETWORK attack vector AND a foothold (rule 3 below).

No severity tiers, no hand-chosen constants beyond the floor and ceiling, each
of which is defended separately below.

Why EPSS rather than CVSS
-------------------------
CVSS was never built as a probability. FIRST's own figures: a CVSS 7+
prioritisation strategy runs ~57% effort, 82% coverage, 4% efficiency, while an
EPSS 0.1 strategy runs ~2.7% effort, 63% coverage, 65% efficiency. 4% efficiency
means ~96% of CVSS-driven remediation targets vulnerabilities nobody exploits.
Of 139,473 CVEs with CVSS 3.x scores, ~2.7% showed observed exploitation in
October 2023.  [https://www.first.org/epss/model]

EPSS is P(exploited in the wild within 30 days) -- a probability by
construction, which is exactly the quantity this simulator needs.
  Jacobs, Romanosky, Suciu, Edwards, Sarabi (2023), arXiv:2302.14172
  Jacobs et al. (2019), arXiv:1908.04856

Scoring rules
-------------
1. MAX over the node's CVEs, never count or sum. The attacker needs one working
   exploit. Counting ranks vsftpd 2.3.4 (3 CVEs, one of them a shipped root
   backdoor) as the safest software in this catalog -- an inversion.
2. NETWORK attack vector only. LOCAL and ADJACENT_NETWORK CVEs cannot drive
   remote lateral movement.
3. Foothold CVEs only (decided 2026-10-12). Compromise here means the attacker
   controls the node, so a CVE counts only if it gives access, judged from
   CVSS impact as NVD records it:
     v3.x  confidentiality HIGH and integrity HIGH
     v2    confidentiality and integrity both PARTIAL or COMPLETE, or an
           obtain*Privilege flag set
   This drops what MAX over EPSS used to pick and which gives no foothold:
   Heartbleed (memory disclosure, C:H/I:N), HTTP/2 Rapid Reset (DoS, A:H),
   OpenSSH user enumeration (C:L). v2's PARTIAL is coarser than v3's HIGH --
   state that asymmetry in threats to validity. A package with no foothold
   CVE is rated at P_FLOOR, like a fully patched node (nginx 1.26.2 and
   MySQL 8.0.40 in the current catalog): nothing in NVD lets an attacker in.
4. P_FLOOR applies to every node including fully patched ones. Justification is
   credential reuse rather than zero-days: pass-the-hash and stolen admin
   credentials are indifferent to patch level, so p = 0 is indefensible.
5. P_CEIL < 1. Exploits fail -- wrong minor version, service down, EDR.

Known modelling choice to state in the paper: EPSS is a 30-day in-the-wild
probability, while p_j is a per-attempt success rate. We use EPSS as a
calibrated relative exploitability measure mapped linearly onto [P_FLOOR,
P_CEIL], not as a literal per-attempt probability.
"""
import json

P_FLOOR = 0.05     # changed from 0.03 on 2026-10-12
P_CEIL = 0.95      # changed from 0.90 on 2026-10-12

NETWORK_ONLY = True
FOOTHOLD_ONLY = True


def foothold(v):
    """True if the CVE gives the attacker access to the node (rule 3)."""
    if v["cvss_version"].startswith("3"):
        return v["conf"] == "HIGH" and v["integ"] == "HIGH"
    return (v["conf"] in ("PARTIAL", "COMPLETE") and v["integ"] in ("PARTIAL", "COMPLETE"))         or bool(v.get("obtain_priv"))


def network_cves(entry):
    if not NETWORK_ONLY:
        return entry["vulnerabilities"]
    return [v for v in entry["vulnerabilities"] if v["attack_vector"] == "NETWORK"]


def attack_cves(entry):
    """The CVEs that rate a node: NETWORK vector and, by default, foothold."""
    vs = network_cves(entry)
    return [v for v in vs if foothold(v)] if FOOTHOLD_ONLY else vs


def _shape(e, transform):
    """Map raw EPSS onto [0,1]. 'linear' is primary; the others are for the
    robustness check, since EPSS is heavily right-skewed in the general CVE
    population (less so in a catalog of famous vulnerable versions)."""
    if transform == "linear":
        return e
    if transform == "sqrt":
        return e ** 0.5
    if transform == "rank":          # caller substitutes a percentile
        return e
    raise ValueError(transform)


def susceptibility(entry, epss, floor=P_FLOOR, ceil=P_CEIL, transform="linear",
                   unscored=frozenset()):
    """entry: catalog record, or None for a fully-patched node.
    epss: {cve_id: float}. unscored: CVEs FIRST was asked about and returned no
    score for -- these are excluded from the max. A CVE in neither is an error,
    never a silent 0.0. Returns p in [floor, ceil]."""
    if entry is None:
        return floor
    vs = attack_cves(entry)
    if not vs:           # no foothold CVE in NVD: rated like a patched node
        return floor
    unknown = [v["cve"] for v in vs if v["cve"] not in epss and v["cve"] not in unscored]
    if unknown:
        raise KeyError(f"{len(unknown)} CVEs absent from EPSS file, e.g. {unknown[:3]} "
                       "-- re-run epss_fetch.py")
    scored = [epss[v["cve"]] for v in vs if v["cve"] in epss]
    if not scored:
        raise ValueError(f"no NETWORK CVE of {entry.get('cpe', '?')} has an EPSS "
                         "score -- cannot rate it, and will not guess")
    best = max(scored)
    return float(min(max(floor + (ceil - floor) * _shape(best, transform), floor), ceil))


def coverage_report(catalog, epss):
    """How much of the catalog EPSS actually covers. A software whose BEST CVE
    is unscored will be silently under-rated, so this must be checked."""
    rows = []
    for sw, entry in sorted(catalog.items()):
        vs = attack_cves(entry)
        ids = [v["cve"] for v in vs]
        scored = [c for c in ids if c in epss]
        best_id = max(ids, key=lambda c: epss.get(c, 0.0)) if ids else None
        rows.append({
            "software": sw,
            "n_network": len(ids),
            "n_scored": len(scored),
            "pct_scored": 100.0 * len(scored) / len(ids) if ids else 0.0,
            "best_cve": best_id,
            "best_epss": epss.get(best_id, 0.0) if best_id else 0.0,
        })
    return rows


def build_table(catalog_path, epss_path, floor=P_FLOOR, ceil=P_CEIL,
                transform="linear"):
    catalog = json.load(open(catalog_path))
    raw = json.load(open(epss_path))
    epss = {k: v for k, v in raw.items() if not k.startswith("_")}
    unscored = frozenset(raw.get("_unscored", ()))
    out = {}
    for sw, e in catalog.items():
        try:
            out[sw] = round(susceptibility(e, epss, floor, ceil, transform, unscored), 4)
        except (KeyError, ValueError) as err:
            raise SystemExit(f"{sw}: {err}")
    out["Fully patched (no known CVEs)"] = floor
    return out, epss, catalog


if __name__ == "__main__":
    from paths import CATALOG, EPSS
    tbl, epss, cat = build_table(CATALOG, EPSS)

    print(f"{'software':30s} {'atkCVEs':>8s} {'scored':>7s} {'best CVE':>16s} "
          f"{'EPSS':>7s} {'p':>7s}")
    for r in sorted(coverage_report(cat, epss), key=lambda r: -r["best_epss"]):
        print(f"{r['software']:30s} {r['n_network']:8d} {r['pct_scored']:6.0f}% "
              f"{str(r['best_cve']):>16s} {r['best_epss']:7.4f} {tbl[r['software']]:7.3f}")
    print(f"{'Fully patched':30s} {0:8d} {'   n/a':>7s} {'-':>16s} {0.0:7.4f} "
          f"{tbl['Fully patched (no known CVEs)']:7.3f}")

    vals = sorted(tbl.values())
    print(f"\np range {vals[0]:.3f} .. {vals[-1]:.3f}   spread {vals[-1]-vals[0]:.3f}")
    if vals[-1] - vals[0] < 0.1:
        print("WARNING: almost no spread -- configuration will carry no severity")
        print("         signal. Widen the catalog with well-patched versions.")

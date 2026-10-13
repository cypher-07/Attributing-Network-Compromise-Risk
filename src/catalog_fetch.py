"""Fetch CVE lists from NVD for all 24 catalog packages: the original 12
famous-vulnerable versions and 12 well-patched versions of the same products.

    python src/catalog_fetch.py

Both halves of the catalog pass through the same NVD query and the same
filter (since 2026-10-12; before that the original 12 kept the teammate's
lists, vulnerability_catalog_live.json, which were not filtered and are now
deleted -- refetching changed only 4 CVEs and no max).

Why the patched versions: every package in the original catalog is a famous catastrophic version, so
under EPSS they all land at p ~ 0.87-0.90 and configuration carries placement
but no severity. These are later, patched releases of the SAME twelve products,
so a config can differ in how bad a node is, not only in where vulnerable nodes
sit.

Entries use the original schema (cve, cvss, cvss_version, attack_vector,
exploitability) plus the impact fields scoring.py needs to tell a foothold
from a nuisance: conf / integ / avail impact as NVD writes them (v3: HIGH /
LOW / NONE, v2: COMPLETE / PARTIAL / NONE), v2's obtain*Privilege flags, and
the CWE ids. CVSS metric preference: v3.1, then v3.0, then v2 -- NVD primary
source where available. Rejected CVEs are skipped.

Unbounded matches are dropped, for all 24 packages. NVD's cpeName search also returns CVEs whose
configuration names the product with NO version or version range ("all
versions") -- mostly 2009-2019 entries. For a release from 2024 those match
by construction, not by evidence, and they set the max EPSS (Apache 2.4.62 was
rated by CVE-2009-0796). A CVE is kept only if some vulnerable cpeMatch for
this vendor:product carries an explicit version or a version bound.

Version choice: each release is after the fix for its product's well-known
recent CVE -- Tomcat 9.0.108 (> CVE-2025-24813, fixed 9.0.99), Exim 4.98.2
(> CVE-2025-26794, fixed 4.98.1), OpenSSH 9.9p1 (> CVE-2024-6387 regreSSHion;
NVD bounds it "< 9.8p1", which plain "9.8" falls under), vsftpd 3.0.3 (newest
in the NVD CPE dictionary).

No silent fallbacks: any request that fails after retries aborts the run and
nothing is written. NVD allows 5 requests / 30 s without an API key, hence the
sleep.
"""
import json, sys, time, urllib.request, urllib.parse
from paths import CATALOG, rel

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE = 2000

ORIGINAL = {
    "Apache HTTP Server 2.4.49": "cpe:2.3:a:apache:http_server:2.4.49",
    "OpenSSH 7.2":               "cpe:2.3:a:openbsd:openssh:7.2",
    "OpenSSL 1.0.1":             "cpe:2.3:a:openssl:openssl:1.0.1",
    "Samba 4.5.0":               "cpe:2.3:a:samba:samba:4.5.0",
    "nginx 1.4.0":               "cpe:2.3:a:f5:nginx:1.4.0",
    "MySQL 5.5.20":              "cpe:2.3:a:oracle:mysql:5.5.20",
    "Apache Struts 2.3.31":      "cpe:2.3:a:apache:struts:2.3.31",
    "Drupal 7.57":               "cpe:2.3:a:drupal:drupal:7.57",
    "Exim 4.87":                 "cpe:2.3:a:exim:exim:4.87",
    "vsftpd 2.3.4":              "cpe:2.3:a:vsftpd_project:vsftpd:2.3.4",
    "Apache Tomcat 8.5.0":       "cpe:2.3:a:apache:tomcat:8.5.0",
    "ProFTPD 1.3.5":             "cpe:2.3:a:proftpd:proftpd:1.3.5",
}

PATCHED = {
    "Apache HTTP Server 2.4.62": "cpe:2.3:a:apache:http_server:2.4.62",
    "OpenSSH 9.9p1":             "cpe:2.3:a:openbsd:openssh:9.9:p1",
    "OpenSSL 3.0.15":            "cpe:2.3:a:openssl:openssl:3.0.15",
    "Samba 4.20.5":              "cpe:2.3:a:samba:samba:4.20.5",
    "nginx 1.26.2":              "cpe:2.3:a:f5:nginx:1.26.2",
    "MySQL 8.0.40":              "cpe:2.3:a:oracle:mysql:8.0.40",
    "Apache Struts 6.4.0":       "cpe:2.3:a:apache:struts:6.4.0",
    "Drupal 10.3.6":             "cpe:2.3:a:drupal:drupal:10.3.6",
    "Exim 4.98.2":               "cpe:2.3:a:exim:exim:4.98.2",
    "vsftpd 3.0.3":              "cpe:2.3:a:vsftpd_project:vsftpd:3.0.3",
    "Apache Tomcat 9.0.108":     "cpe:2.3:a:apache:tomcat:9.0.108",
    "ProFTPD 1.3.9a":            "cpe:2.3:a:proftpd:proftpd:1.3.9a",
}


def get(params):
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                sys.exit(f"NVD request failed after 4 attempts: {url}\n{e}\n"
                         "nothing written")
            time.sleep(10 * (attempt + 1))


def metric(cve):
    m = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        rows = m.get(key)
        if not rows:
            continue
        row = next((r for r in rows if r.get("type") == "Primary"), rows[0])
        d = row["cvssData"]
        out = {
            "cvss": d["baseScore"],
            "cvss_version": d["version"],
            "attack_vector": d.get("attackVector") or d.get("accessVector"),
            "exploitability": row.get("exploitabilityScore"),
            "conf": d.get("confidentialityImpact"),
            "integ": d.get("integrityImpact"),
            "avail": d.get("availabilityImpact"),
        }
        if key == "cvssMetricV2":
            out["obtain_priv"] = any(row.get(k) for k in (
                "obtainAllPrivilege", "obtainUserPrivilege", "obtainOtherPrivilege"))
        return out
    return None


def bounded(cve, cpe):
    """True if NVD names this vendor:product with an explicit version or a
    version bound in some vulnerable match -- not just "all versions"."""
    vendor_product = ":".join(cpe.split(":")[3:5])
    for conf in cve.get("configurations", []):
        for node in conf.get("nodes", []):
            for m in node.get("cpeMatch", []):
                parts = m["criteria"].split(":")
                if not m.get("vulnerable") or ":".join(parts[3:5]) != vendor_product:
                    continue
                if parts[5] not in ("*", "-") or any(
                        k in m for k in ("versionStartIncluding", "versionStartExcluding",
                                         "versionEndIncluding", "versionEndExcluding")):
                    return True
    return False


def fetch(cpe):
    parts = cpe.split(":")
    name = ":".join(parts + ["*"] * (13 - len(parts)))
    vulns, start, unscored, unbounded = [], 0, [], []
    while True:
        d = get({"cpeName": name, "startIndex": start, "resultsPerPage": PAGE})
        for v in d["vulnerabilities"]:
            c = v["cve"]
            if c.get("vulnStatus") == "Rejected":
                continue
            if not bounded(c, cpe):
                unbounded.append(c["id"])
                continue
            m = metric(c)
            if m is None:
                unscored.append(c["id"])
                continue
            cwe = sorted({x["value"] for w in c.get("weaknesses", [])
                          for x in w.get("description", []) if x["value"].startswith("CWE-")})
            vulns.append({"cve": c["id"], **m, "cwe": cwe})
        start += d["resultsPerPage"]
        time.sleep(6.5)
        if start >= d["totalResults"]:
            break
    if d["totalResults"] == 0:
        sys.exit(f"{cpe}: NVD returned no CVEs -- check the CPE name. "
                 "nothing written")
    return vulns, unscored, unbounded


if __name__ == "__main__":
    out = CATALOG
    merged = {}
    for sw, cpe in {**ORIGINAL, **PATCHED}.items():
        vulns, unscored, unbounded = fetch(cpe)
        if not vulns:
            sys.exit(f"{sw}: no version-bounded CVEs left -- pick another "
                     "version. nothing written")
        net = sum(v["attack_vector"] == "NETWORK" for v in vulns)
        print(f"  {sw:28s} {len(vulns):4d} CVEs ({net} NETWORK), "
              f"{len(unbounded)} unbounded dropped"
              + (f", {len(unscored)} without CVSS skipped" if unscored else ""),
              file=sys.stderr)
        merged[sw] = {"cpe": cpe, "vulnerabilities": vulns}
    json.dump(merged, open(out, "w"), indent=1)
    print(f"wrote {rel(out)}: {len(ORIGINAL)} original + {len(PATCHED)} patched, "
          "all re-fetched from NVD", file=sys.stderr)

"""
Download core openFDA datasets into data/openfda/<domain>/<endpoint>/
- Skips files already downloaded (resume-safe)
- Shows progress per file and overall
"""

import urllib.request
import urllib.error
import json
import os
import sys
import time
import ssl

# Bypass SSL verification (handles corporate proxies with self-signed certs)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MANIFEST_URL = "https://api.fda.gov/download.json"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "openfda")

CORE_DATASETS = {
    "drug":         ["label", "ndc", "drugsfda", "enforcement", "shortages"],
    "device":       ["510k", "pma", "classification", "enforcement", "registrationlisting"],
    "transparency": ["crl"],
    "other":        ["substance", "unii"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.1f} MB"


def download_file(url: str, dest: str) -> float:
    """Download url → dest, return elapsed seconds. Skips if already exists."""
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"    [SKIP] already exists ({fmt_mb(size/1024/1024)}): {os.path.basename(dest)}")
        return 0.0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=120, context=SSL_CTX) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 256  # 256 KB
            with open(tmp, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = downloaded / total * 100
                        bar = "#" * int(pct / 5)
                        sys.stdout.write(
                            f"\r    [{bar:<20}] {pct:5.1f}%  {fmt_mb(downloaded/1024/1024)}/{fmt_mb(total/1024/1024)}"
                        )
                        sys.stdout.flush()
        print()  # newline after progress bar
        os.rename(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e

    return time.time() - start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Fetching openFDA download manifest...")
    with urllib.request.urlopen(MANIFEST_URL, context=SSL_CTX) as r:
        manifest = json.loads(r.read())
    results = manifest["results"]

    # Collect all files to download
    plan = []  # (domain, endpoint, part_index, url, size_mb)
    for domain, endpoints in CORE_DATASETS.items():
        for endpoint in endpoints:
            info = results.get(domain, {}).get(endpoint)
            if not info:
                print(f"  WARNING: {domain}/{endpoint} not found in manifest, skipping.")
                continue
            parts = info.get("partitions", [])
            for i, part in enumerate(parts):
                plan.append((domain, endpoint, i + 1, len(parts), part["file"], float(part.get("size_mb", 0))))

    total_files = len(plan)
    total_mb = sum(p[5] for p in plan)
    print(f"\nPlan: {total_files} files, {fmt_mb(total_mb)} total (compressed)\n")
    print("=" * 60)

    overall_start = time.time()
    completed_mb = 0.0
    skipped = 0
    downloaded = 0

    for idx, (domain, endpoint, part_num, total_parts, url, size_mb) in enumerate(plan, 1):
        filename = url.split("/")[-1]
        dest = os.path.join(OUT_DIR, domain, endpoint, filename)
        label = f"{domain}/{endpoint}" + (f" (part {part_num}/{total_parts})" if total_parts > 1 else "")

        print(f"[{idx}/{total_files}] {label}  ({fmt_mb(size_mb)})")
        print(f"    URL: {url}")

        already_exists = os.path.exists(dest)
        elapsed = download_file(url, dest)

        if already_exists:
            skipped += 1
        else:
            downloaded += 1
            print(f"    Saved → {dest}  ({elapsed:.1f}s)")

        completed_mb += size_mb
        elapsed_total = time.time() - overall_start
        rate = completed_mb / elapsed_total if elapsed_total > 0 else 0
        remaining = (total_mb - completed_mb) / rate if rate > 0 else 0
        print(f"    Overall: {completed_mb:.0f}/{total_mb:.0f} MB  |  ETA: {remaining/60:.1f} min\n")

    print("=" * 60)
    print(f"Done. {downloaded} downloaded, {skipped} skipped (already existed).")
    print(f"Files saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()

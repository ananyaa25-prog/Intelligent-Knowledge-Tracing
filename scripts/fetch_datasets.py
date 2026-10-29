"""
scripts/fetch_datasets.py
-------------------------
Tries to obtain the external knowledge-tracing datasets used in v3 Stage 7.

For each dataset it first looks in data/external/<folder>/ for files the user placed
there. Otherwise, if a PUBLIC direct download (no login, no licence click-through)
is known, it downloads it with connect/read timeouts and a size cap into its own new
folder (data/external/<folder>/), writing to a .part file first. Nothing is executed
or imported from downloaded files; they are only read as data by the adapters.
If the network is blocked, a login or licence acceptance is needed, or the download
fails, the script does NOT fail: it records the dataset in docs/DATASETS_TODO.md
(name, where to get it, expected folder and file names) and continues.

Usage:
  python scripts/fetch_datasets.py                 # check + download public ones
  python scripts/fetch_datasets.py --no-download   # only check local files, write TODO
  python scripts/fetch_datasets.py --max-mb 800 --only eedi
Exit code is always 0 unless the arguments are wrong.
"""

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
TODO = ROOT / "docs" / "DATASETS_TODO.md"
STATUS = ROOT / "reports" / "v3" / "datasets_status.json"

# Only URLs that need no login. Everything else is listed for manual download.
DATASETS = [
    {"key": "eedi", "name": "Eedi / NeurIPS 2020 Education Challenge (tasks 1-2)",
     "folder": "eedi_neurips2020", "expect": ["data.zip  (or the extracted data/train_data/train_task_1_2.csv, "
                                              "data/metadata/question_metadata_task_1_2.csv, "
                                              "data/metadata/student_metadata_task_1_2.csv, data/metadata/subject_metadata.csv)"],
     "url": "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip", "file": "data.zip",
     "where": "https://eedi.com/projects/neurips-education-challenge (public archive linked from the challenge page)",
     "section": "primary + secondary (UK learners; age from date of birth)",
     "licence": "check the licence on the challenge page before redistributing; academic, non-commercial use only"},
    {"key": "xes3g5m", "name": "XES3G5M (TAL Education, primary-school maths, China)", "folder": "xes3g5m",
     "expect": ["kc_level/train_valid_sequences.csv", "metadata/questions.json (from the official archive)"],
     "url": None, "where": "https://github.com/ai4ed/XES3G5M (Google Drive link; downloading means accepting the licence)",
     "section": "primary", "licence": "licence acceptance required -> manual"},
    {"key": "junyi", "name": "Junyi Academy learning-activity dataset (2018-2019)", "folder": "junyi",
     "expect": ["Log_Problem.csv", "Info_Content.csv", "Info_UserData.csv"], "url": None,
     "where": "https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy (Kaggle login)",
     "section": "primary + secondary (grade in Info_UserData / Info_Content)", "licence": "Kaggle login -> manual"},
    {"key": "kddcup2010", "name": "KDD Cup 2010 (Algebra I 2005-2006 / Bridge to Algebra 2006-2007)",
     "folder": "kddcup2010", "expect": ["algebra_2005_2006_train.txt", "bridge_to_algebra_2006_2007_train.txt"],
     "url": None, "where": "https://pslcdatashop.web.cmu.edu/KDDCup/ (registration required)",
     "section": "secondary", "licence": "registration -> manual"},
    {"key": "statics2011", "name": "Statics2011 (OLI Engineering Statics, Fall 2011)", "folder": "statics2011",
     "expect": ["STATICS_train.csv", "STATICS_test.csv", "STATICS_qid_sid_sname", "STATICS_sid_sname"],
     "url": None,
     "mirror": "https://raw.githubusercontent.com/jennyzhang0215/DKVMN/master/data/STATICS/",
     "where": "public mirror in the DKVMN authors' repository (github.com/jennyzhang0215/DKVMN, data/STATICS); "
              "original: https://pslcdatashop.web.cmu.edu/ dataset 507 (login)",
     "section": "undergraduate (engineering statics, not maths)",
     "licence": "mirror has no licence file; original DataShop terms apply - academic use, check before redistributing"},
    {"key": "assist2012", "name": "ASSISTments 2012-2013 (school data with affect)", "folder": "assistments2012",
     "expect": ["2012-2013-data-with-predictions-4-final.csv"], "url": None,
     "where": "https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect (manual)",
     "section": "secondary", "licence": "terms on the ASSISTments data page"},
    {"key": "assist2015", "name": "ASSISTments 2015 skill builder", "folder": "assistments2015",
     "expect": ["2015_100_skill_builders_main_problems.csv"], "url": None,
     "where": "https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data (manual)",
     "section": "secondary", "licence": "terms on the ASSISTments data page"},
]


def _local_files(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".part") and p.name != ".gitkeep"]


def _download(url: str, dest: Path, max_mb: int, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": "kt-v3-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        size = int(r.headers.get("Content-Length") or 0)
        if "html" in ctype.lower():
            raise RuntimeError(f"got an HTML page ({ctype}); a login or click-through is probably required")
        if size and size > max_mb * 1024 * 1024:
            raise RuntimeError(f"file is {size / 1e6:.0f} MB, above --max-mb {max_mb}")
        part = dest.with_suffix(dest.suffix + ".part")
        got = 0
        t0 = time.time()
        with open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got > max_mb * 1024 * 1024:
                    raise RuntimeError(f"download exceeded --max-mb {max_mb}")
                if got % (50 << 20) < (1 << 20):
                    print(f"    {got / 1e6:.0f} MB, {got / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s", flush=True)
        if size and got != size:
            raise RuntimeError(f"incomplete download ({got} of {size} bytes)")
        os.replace(part, dest)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError(f"corrupt zip member {bad}")
    return got


def write_todo(results):
    L = ["# External datasets: status and manual steps (v3 Stage 7)", "",
         "Generated by `python scripts/fetch_datasets.py`. A section counts as \"validated on real data\" only if a "
         "real dataset for that level was loaded (see reports/v3/datasets.md). Never replace a missing dataset with "
         "invented data.", "",
         "| Dataset | Status | Where to get it | Put it in | Expected files | Would serve section | Licence / access |",
         "|---|---|---|---|---|---|---|"]
    for d, st in results:
        L.append(f"| {d['name']} | {st} | {d['where']} | data/external/{d['folder']}/ | {'; '.join(d['expect'])} | "
                 f"{d['section']} | {d['licence']} |")
    L += ["", "After placing files, re-run `python scripts/fetch_datasets.py --no-download` and "
          "`python scripts/run_external_benchmark.py`.", ""]
    TODO.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-mb", type=int, default=800)
    ap.add_argument("--timeout", type=int, default=30, help="connect/read timeout in seconds")
    args = ap.parse_args()
    socket.setdefaulttimeout(args.timeout)
    results, status = [], {}
    for d in DATASETS:
        folder = EXT / d["folder"]
        files = _local_files(folder)
        if files:
            st = f"available locally ({len(files)} file(s))"
        elif d.get("mirror") and not args.no_download and (not args.only or args.only == d["key"]):
            print(f"[fetch] {d['name']}: downloading {len(d['expect'])} files from {d['mirror']}", flush=True)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                n = sum(_download(d["mirror"] + fn, folder / fn, args.max_mb, args.timeout) for fn in d["expect"])
                st = f"downloaded ({n / 1e6:.1f} MB) from the public mirror"
            except (urllib.error.URLError, socket.timeout, OSError, RuntimeError) as e:
                st = f"missing: download failed ({type(e).__name__}: {str(e)[:120]})"
        elif args.no_download or (args.only and args.only != d["key"]) or not d["url"]:
            st = "missing: manual download needed" if not d["url"] else "missing: public download not attempted"
        else:
            print(f"[fetch] {d['name']}: downloading {d['url']}", flush=True)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                n = _download(d["url"], folder / d["file"], args.max_mb, args.timeout)
                st = f"downloaded ({n / 1e6:.0f} MB) from the public archive"
            except (urllib.error.URLError, socket.timeout, OSError, RuntimeError, zipfile.BadZipFile) as e:
                st = f"missing: download failed ({type(e).__name__}: {str(e)[:120]})"
        print(f"[fetch] {d['key']}: {st}", flush=True)
        results.append((d, st))
        status[d["key"]] = {"status": st, "folder": f"data/external/{d['folder']}"}
    write_todo(results)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("[fetch] docs/DATASETS_TODO.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

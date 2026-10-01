#!/usr/bin/env python3
"""Read-only status of a GA / walk-forward job — no server, no writes.

    python scripts/ga_job_status.py                    # newest GA job
    python scripts/ga_job_status.py --job-id ga_0a442907
    python scripts/ga_job_status.py --stale-minutes 15 --tail 5

Prints the progress JSON, wall-clock elapsed time, the worker processes
(CPU seconds + RSS, matched by command line), the job log's mtime / size /
line count and the last log lines.

Exit codes
    0  job running and its progress file is fresh
    1  no job file found
    2  progress file stale (mtime older than --stale-minutes, default 5)
    3  no worker process matches the job (only when it can be determined)

Process stats need no third-party package: ``psutil`` is used when it is
installed (it is NOT in ``requirements.txt``), otherwise the Windows
``Get-CimInstance Win32_Process`` inventory is queried through ``subprocess``
and finally ``tasklist`` is used (that last one cannot see command lines, so it
is reported as an approximate match only).  This tool only ever reads files and
process metadata.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_JOBS_DIR = PROJECT_ROOT / "data" / "ga_jobs"

#: Exit codes (documented in the module docstring).
RC_OK, RC_NO_JOB, RC_STALE, RC_NOT_RUNNING = 0, 1, 2, 3

_CIM_PS = (
    "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | "
    "Select-Object ProcessId,ParentProcessId,WorkingSetSize,UserModeTime,"
    "KernelModeTime,CommandLine | ConvertTo-Json -Compress"
)


# ── job / file resolution ───────────────────────────────────────────────────

def newest_job_file(jobs_dir: Path) -> Path | None:
    """Newest ``ga_*.json`` job file (by mtime), or None."""
    try:
        candidates = [p for p in jobs_dir.glob("ga_*.json") if p.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_job_file(args) -> Path | None:
    if args.job_file:
        return Path(args.job_file)
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else DEFAULT_JOBS_DIR
    if args.job_id:
        jid = args.job_id
        if not jid.endswith(".json"):
            jid = f"{jid if jid.startswith('ga_') else 'ga_' + jid}.json"
        return jobs_dir / jid
    return newest_job_file(jobs_dir)


def read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def file_stats(path: Path) -> dict:
    """mtime / size / line count (all read-only, missing file tolerated)."""
    out = {"path": str(path), "exists": path.exists()}
    if not out["exists"]:
        return out
    st = path.stat()
    out["mtime"] = st.st_mtime
    out["mtime_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
    out["size"] = st.st_size
    try:
        with open(path, "rb") as f:
            out["lines"] = sum(1 for _ in f)
    except Exception:
        out["lines"] = None
    return out


def tail_lines(path: Path, n: int) -> list:
    if n <= 0 or not path.exists():
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception:
        return []
    return lines[-n:]


# ── process inventory (matched by command line) ─────────────────────────────

def _cim_processes() -> list:
    if os.name != "nt":
        return []
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _CIM_PS],
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    data = json.loads(proc.stdout)
    if isinstance(data, dict):
        data = [data]
    rows = []
    for item in data:
        cpu_100ns = int(item.get("UserModeTime") or 0) + int(item.get("KernelModeTime") or 0)
        rows.append({
            "pid": int(item.get("ProcessId") or 0),
            "ppid": int(item.get("ParentProcessId") or 0),
            "cpu_seconds": round(cpu_100ns / 1e7, 2),
            "rss_mb": round(int(item.get("WorkingSetSize") or 0) / 1048576.0, 1),
            "cmdline": item.get("CommandLine") or "",
        })
    return rows


def _tasklist_processes() -> list:
    """Last resort: CPU time / RSS for python.exe, but NO command line / parent."""
    if os.name != "nt":
        return []
    proc = subprocess.run(
        ["tasklist", "/FO", "CSV", "/V", "/FI", "IMAGENAME eq python.exe"],
        capture_output=True, text=True, timeout=30)
    rows = []
    for row in csv.DictReader(io.StringIO(proc.stdout)):
        cpu = (row.get("CPU Time") or "0:00:00").strip()
        try:
            h, m, s = [float(x) for x in cpu.split(":")]
            cpu_s = h * 3600 + m * 60 + s
        except Exception:
            cpu_s = None
        try:
            rss_kb = float((row.get("Mem Usage") or "0 K").replace(",", "").split()[0])
        except Exception:
            rss_kb = 0.0
        rows.append({"pid": int(row.get("PID") or 0), "ppid": 0,
                     "cpu_seconds": cpu_s, "rss_mb": round(rss_kb / 1024.0, 1),
                     "cmdline": ""})
    return rows


def _match_job(rows: list, job_file: Path) -> tuple[list, int]:
    """Direct command-line matches plus their descendants (the pool workers).

    The pool's worker processes are spawned with a ``multiprocessing.spawn``
    command line that does NOT mention the job file, so a command-line-only
    match reports 1 process and hides the CPU the job is actually burning.  A
    process whose parent (transitively) is a direct match is therefore reported
    as a ``child`` of the job.
    """
    name = job_file.name
    direct = {r["pid"] for r in rows if name in (r["cmdline"] or "")}
    keep = set(direct)
    changed = True
    while changed:  # transitive descendants
        changed = False
        for r in rows:
            if r["pid"] not in keep and r["ppid"] and r["ppid"] in keep:
                keep.add(r["pid"])
                changed = True
    out = []
    for r in rows:
        if r["pid"] not in keep:
            continue
        item = dict(r)
        item["match"] = "worker" if r["pid"] in direct else "child"
        out.append(item)
    return sorted(out, key=lambda r: r["pid"]), len(direct)


def process_stats(job_file: Path) -> dict:
    """Worker processes of *job_file*: pid / cpu_seconds / rss_mb / cmdline.

    ``source`` says how they were matched, ``note`` says what could not be
    determined (never silently): ``cmdline-exact`` (psutil/CIM) is a real match,
    ``image-name-approx`` (tasklist) may include unrelated python processes.
    """
    try:
        import psutil  # not a declared dependency — used only when present
    except Exception:
        psutil = None

    if psutil is not None:
        rows = []
        for p in psutil.process_iter(["pid", "name", "cmdline",
                                      "memory_info", "cpu_times"]):
            try:
                ct = p.info["cpu_times"]
                rows.append({
                    "pid": p.info["pid"], "ppid": p.ppid(),
                    "cpu_seconds": round((ct.user or 0) + (ct.system or 0), 2),
                    "rss_mb": round((p.info["memory_info"].rss or 0) / 1048576.0, 1),
                    "cmdline": " ".join(p.info["cmdline"] or []),
                })
            except Exception:
                continue
        procs, direct = _match_job(rows, job_file)
        return {"source": "psutil:cmdline-exact", "matched": True, "note": "",
                "direct_matches": direct, "processes": procs}

    note = ""
    try:
        procs, direct = _match_job(_cim_processes(), job_file)
        return {"source": "cim:cmdline-exact", "matched": True, "note": note,
                "direct_matches": direct, "processes": procs}
    except Exception as e:
        note = f"command-line inventory unavailable ({type(e).__name__}: {e})"

    try:
        rows = _tasklist_processes()
        return {"source": "tasklist:image-name-approx", "matched": False,
                "note": note + "; tasklist cannot see command lines, so these "
                        "python.exe rows are NOT verified to be this job's workers",
                "direct_matches": None,
                "processes": sorted(rows, key=lambda r: r["pid"])}
    except Exception as e:
        return {"source": "none", "matched": False,
                "note": note + f"; tasklist failed too ({type(e).__name__}: {e})",
                "direct_matches": None, "processes": []}


# ── reporting ───────────────────────────────────────────────────────────────

def collect(args) -> dict:
    job_file = resolve_job_file(args)
    if job_file is None or not job_file.exists():
        return {"job_file": str(job_file) if job_file else None, "found": False}

    progress_file = Path(args.progress_file) if args.progress_file else Path(str(job_file) + ".progress")
    log_file = Path(args.log_file) if args.log_file else Path(str(job_file) + ".log")
    result_file = Path(str(job_file) + ".result")
    job = read_json(job_file) or {}
    progress = read_json(progress_file)

    now = time.time()
    pstat = file_stats(progress_file)
    age = (now - pstat["mtime"]) if pstat.get("exists") else None
    stale = (age is None) or (age > args.stale_minutes * 60)

    started_ts = None
    if isinstance(progress, dict) and progress.get("started_at"):
        try:
            started_ts = time.mktime(time.strptime(progress["started_at"], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            started_ts = None
    if started_ts is None:
        started_ts = job_file.stat().st_mtime
    elapsed_s = round(now - started_ts, 1)

    procs = {"source": "skipped", "matched": None, "note": "--no-process-check",
             "processes": []} if args.no_process_check else process_stats(job_file)
    running = None
    if procs["matched"] is True:
        running = bool(procs["processes"])
    elif procs["source"].startswith("tasklist"):
        running = None  # cannot be determined without command lines

    log = file_stats(log_file)
    log["tail"] = tail_lines(log_file, args.tail)

    return {
        "found": True,
        "job_file": str(job_file),
        "job": job,
        "progress_file": str(progress_file),
        "progress": progress,
        "progress_age_s": round(age, 1) if age is not None else None,
        "stale": stale,
        "stale_minutes": args.stale_minutes,
        "elapsed_s": elapsed_s,
        "processes": procs,
        "running": running,
        "log": log,
        "result_file_exists": result_file.exists(),
        "now": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def render(info: dict, tail: int) -> str:
    if not info.get("found"):
        return f"no job file found ({info.get('job_file')})"
    out = []
    out.append(f"job            : {info['job_file']}")
    job = info["job"]
    # The job's timeframe whitelist (``None``/absent = unrestricted) is part of
    # the job's identity — the timeframe is a gene, so it decides run time.
    pool = job.get("timeframe_pool")
    if isinstance(pool, (list, tuple)):
        pool = ",".join(str(tf) for tf in pool)
    out.append(f"job params     : pop={job.get('population_size')} "
               f"gens={job.get('generations')} workers={job.get('max_workers')} "
               f"tf_pool={pool or 'unrestricted'} "
               f"window={job.get('date_start')}~{job.get('date_end')} "
               f"symbols={len(job.get('symbols') or [])}")
    out.append(f"progress file  : {info['progress_file']}")
    out.append(f"progress JSON  : {json.dumps(info['progress'])}")
    age = info["progress_age_s"]
    out.append(f"progress age   : {'n/a (missing)' if age is None else f'{age:.0f}s'} "
               f"(stale > {info['stale_minutes']} min: {info['stale']})")
    out.append(f"wall elapsed   : {info['elapsed_s']:.0f}s "
               f"(result written: {info['result_file_exists']})")
    procs = info["processes"]
    out.append(f"process source : {procs['source']}")
    if procs["note"]:
        out.append(f"process note   : {procs['note']}")
    if procs["processes"]:
        for p in procs["processes"]:
            out.append(f"  pid {p['pid']:<7} [{p.get('match', '?')}] "
                       f"cpu={p['cpu_seconds']}s rss={p['rss_mb']}MB "
                       f"{p['cmdline'][:100]}")
        out.append(f"  processes={len(procs['processes'])} "
                   f"(command-line matched: {procs.get('direct_matches')}) "
                   f"total cpu={round(sum(p['cpu_seconds'] or 0 for p in procs['processes']), 1)}s "
                   f"total rss={round(sum(p['rss_mb'] or 0 for p in procs['processes']), 1)}MB")
    else:
        out.append("  (none)")
    log = info["log"]
    if log.get("exists"):
        out.append(f"job log        : {log['path']} mtime={log['mtime_iso']} "
                   f"size={log['size']}B lines={log['lines']}")
        for line in log["tail"][-tail:]:
            out.append(f"  | {line}")
    else:
        out.append(f"job log        : missing ({log['path']})")
    return "\n".join(out)


def exit_code(info: dict) -> int:
    if not info.get("found"):
        return RC_NO_JOB
    if info["stale"]:
        return RC_STALE
    if info["running"] is False:
        return RC_NOT_RUNNING
    return RC_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--job-id", help="ga_0a442907 (or 0a442907) — else the newest job")
    p.add_argument("--job-file", help="explicit path to <job>.json")
    p.add_argument("--jobs-dir", help=f"job directory (default {DEFAULT_JOBS_DIR})")
    p.add_argument("--progress-file", help="override the progress path (tests)")
    p.add_argument("--log-file", help="override the log path (tests)")
    p.add_argument("--stale-minutes", type=float, default=5.0,
                   help="progress mtime older than this is stale (default 5)")
    p.add_argument("--tail", type=int, default=8, help="log lines to print (default 8)")
    p.add_argument("--no-process-check", action="store_true",
                   help="skip the process inventory (no psutil/CIM/tasklist call)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    return p


def main(argv=None) -> int:
    # Log files carry non-UTF-8 bytes from strategy warnings; a cp936 console
    # must not turn a status report into a UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # pragma: no cover - older/odd stdout
        pass
    args = build_parser().parse_args(argv)
    info = collect(args)
    if args.json:
        print(json.dumps(info, default=str, indent=2))
    else:
        print(render(info, args.tail))
    return exit_code(info)


if __name__ == "__main__":
    sys.exit(main())

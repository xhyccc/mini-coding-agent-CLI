#!/usr/bin/env python3
"""Batch runner for GDPval tasks using the codelet CLI agent.

Iterates through all 220 tasks in the GDPval gold subset, creates an isolated
workspace per task with reference files, runs the codelet agent with the task
prompt, and collects deliverables.

Usage:
    python3 examples/run_gdpval_batch.py [--start N] [--end N] [--max-tasks N]

Progress is saved to gdpval_results/progress.json so interrupted runs can resume.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "gdpval_dataset"
PARQUET_PATH = DATASET_DIR / "data" / "train-00000-of-00001.parquet"
RESULTS_DIR = REPO_ROOT / "gdpval_results"
PROGRESS_PATH = RESULTS_DIR / "progress.json"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 10
DEFAULT_TIMEOUT = 1800  # seconds per task (30 min - enough for complex tasks with many API calls)


def load_tasks():
    """Load the GDPval task dataframe."""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")
    return pd.read_parquet(PARQUET_PATH)


def load_progress():
    """Load progress log or return empty state."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "skipped": []}


def save_progress(state):
    """Atomically save progress log."""
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(PROGRESS_PATH)


def setup_task_workspace(task_row, task_idx):
    """Create an isolated workspace for a single task.

    Copies reference files into the workspace so the agent can read them.
    Cleans up any leftover files from previous runs.
    Returns the workspace path.
    """
    task_id = task_row["task_id"]
    workspace = RESULTS_DIR / "workspaces" / f"task_{task_idx:03d}_{task_id[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)

    # Get reference file names to preserve
    ref_files = task_row["reference_files"]
    if hasattr(ref_files, "tolist"):
        ref_files = ref_files.tolist()
    ref_names = {Path(p).name for p in ref_files}

    # Clean up old files from previous runs (except reference files and debug files)
    for item in workspace.iterdir():
        if item.name in ref_names:
            continue
        if item.name.startswith("_") and item.suffix in {".json", ".txt", ".log"}:
            # Keep debug files for analysis
            continue
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Copy reference files into workspace
    missing_refs = []
    for ref_path in ref_files:
        src = DATASET_DIR / ref_path
        if src.exists():
            dst = workspace / Path(ref_path).name
            shutil.copy2(src, dst)
        else:
            missing_refs.append(Path(ref_path).name)

    if missing_refs:
        print(f"  [WARN] Missing reference files for task {task_id}: {missing_refs}")

    # Write task metadata for debugging
    meta = {
        "task_id": task_id,
        "task_index": task_idx,
        "sector": task_row["sector"],
        "occupation": task_row["occupation"],
        "reference_files": ref_files,
        "deliverable_files": task_row["deliverable_files"].tolist() if hasattr(task_row["deliverable_files"], "tolist") else list(task_row["deliverable_files"]),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(workspace / "_task_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return workspace, missing_refs


def run_codelet(workspace: Path, prompt: str, max_steps: int = DEFAULT_MAX_STEPS):
    """Run the codelet CLI agent in machine mode with auto-approval.

    Uses Popen so we can kill the process on timeout and still collect
    whatever output was produced. Returns (returncode, stdout, stderr, elapsed).
    """
    env = os.environ.copy()
    env["CODEXLET_CWD"] = str(workspace)

    # Write a temporary config to disable web_search for GDPval tasks.
    # This is auto-discovered by codelet as .codelet/config.yaml.
    codelet_dir = workspace / ".codelet"
    codelet_dir.mkdir(exist_ok=True)
    config_yaml = codelet_dir / "config.yaml"
    config_yaml.write_text(
        "harness:\n  disable_web_search: true\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable, "-m", "codelet",
        "--cwd", str(workspace),
        "--machine",
        "--approval", "auto",
        "--max-steps", str(max_steps),
        "--max-new-tokens", "16384",
        "--openai-timeout", "300",
        "--no-welcome",
        prompt,
    ]

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def _kill_after_timeout():
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(DEFAULT_TIMEOUT, _kill_after_timeout)
    timer.start()

    try:
        stdout, stderr = proc.communicate()
        timer.cancel()
        elapsed = time.time() - start
        return proc.returncode, stdout, stderr, elapsed
    except subprocess.TimeoutExpired:
        timer.cancel()
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        stdout, stderr = proc.communicate()
        elapsed = time.time() - start
        return -1, stdout, stderr, elapsed
    except Exception as exc:
        timer.cancel()
        proc.kill()
        elapsed = time.time() - start
        return -2, "", str(exc), elapsed


def collect_deliverables(workspace: Path, task_row, task_idx):
    """Collect deliverable files the agent created in the workspace.

    Only counts files that match expected deliverable types (PDF, Excel, Word,
    PowerPoint, images, audio, video, text). Intermediate scripts (.py, .sh, .json,
    .md) are NOT counted as deliverables.

    Returns a list of relative paths (from workspace root) of created files.
    """
    deliverables = []
    ref_files = task_row["reference_files"]
    if hasattr(ref_files, "tolist"):
        ref_files = ref_files.tolist()
    # Names of files that were copied in as reference materials
    ref_names = {Path(p).name for p in ref_files}

    # Expected deliverable extensions
    deliverable_extensions = {
        ".pdf", ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp",
        ".wav", ".mp3", ".mp4", ".avi", ".mov", ".mkv",
        ".csv", ".html", ".zip", ".txt",
    }
    # Intermediate / non-deliverable extensions
    non_deliverable_extensions = {
        ".py", ".sh", ".bash", ".json", ".md", ".log",
        ".yml", ".yaml", ".xml", ".sql", ".js", ".ts", ".css",
    }

    for path in workspace.rglob("*"):
        if path.is_file() and not path.name.startswith("_"):
            rel = path.relative_to(workspace)
            # Skip reference files
            if path.name in ref_names:
                continue
            # Skip intermediate scripts
            if path.suffix.lower() in non_deliverable_extensions:
                continue
            # Only count known deliverable types
            if path.suffix.lower() in deliverable_extensions:
                deliverables.append(str(rel))

    return deliverables


def run_single_task(df, task_idx, progress, max_steps):
    """Run one GDPval task end-to-end.

    Returns updated progress dict.
    """
    task_row = df.iloc[task_idx]
    task_id = task_row["task_id"]

    print(f"\n{'='*60}")
    print(f"Task {task_idx + 1}/{len(df)} | {task_id}")
    print(f"Sector: {task_row['sector']}")
    print(f"Occupation: {task_row['occupation']}")
    print(f"{'='*60}")

    # Skip if already completed
    if task_id in progress["completed"]:
        print("  [SKIP] Already completed.")
        return progress
    if task_id in progress["skipped"]:
        print("  [SKIP] Previously skipped.")
        return progress

    # Setup workspace
    workspace, missing_refs = setup_task_workspace(task_row, task_idx)
    print(f"  Workspace: {workspace}")

    # Build prompt - make it more direct and prescriptive
    prompt = task_row["prompt"]
    
    # Get reference and deliverable file names
    ref_files = task_row["reference_files"]
    if hasattr(ref_files, "tolist"):
        ref_files = ref_files.tolist()
    deliv_files = task_row["deliverable_files"]
    if hasattr(deliv_files, "tolist"):
        deliv_files = deliv_files.tolist()
    
    ref_names = [Path(p).name for p in ref_files]
    expected = [Path(p).name for p in deliv_files]
    
    prompt += (
        f"\n\n[SYSTEM INSTRUCTIONS - FOLLOW EXACTLY]:\n"
        f"1. Reference files available in workspace: {ref_names}\n"
        f"2. Expected deliverable(s): {expected}\n"
        f"3. CRITICAL: The task description above ALREADY contains all specific data, numbers, and values you need. "
        f"Reference files are ONLY for format/template guidance. Do NOT try to perfectly extract every detail from reference files.\n"
        f"4. Read reference files ONCE using run_python (pandas/openpyxl for Excel, pdfplumber for PDF text, python-docx for Word). "
        f"Extract just enough to understand the structure and format - do NOT perform deep analysis.\n"
        f"5. Create the deliverable immediately after reading - do NOT perform exploratory analysis\n"
        f"6. Save all output files in the current workspace directory using write_file or run_python\n"
        f"7. Pre-installed libraries: pandas, openpyxl, python-docx, pdfplumber, fpdf2 (import: from fpdf import FPDF), reportlab, python-pptx. "
        f"Do NOT try to install packages - they are already available. Use simple, tested syntax.\n"
        f"8. Issue <final> as soon as the deliverable file is created"
    )

    # Add note about missing reference files
    if missing_refs:
        prompt += (
            f"\n\nNOTE: The following reference files were NOT available in the dataset: {missing_refs}. "
            f"Do NOT search for them or try to find them. Use the data already provided in the task description above."
        )

    # Run agent
    print(f"  Running codelet (max_steps={max_steps}, timeout={DEFAULT_TIMEOUT}s)...")
    rc, stdout, stderr, elapsed = run_codelet(workspace, prompt, max_steps)

    # Collect results
    deliverables = collect_deliverables(workspace, task_row, task_idx)

    result = {
        "task_id": task_id,
        "task_index": task_idx,
        "returncode": rc,
        "elapsed_seconds": round(elapsed, 1),
        "workspace": str(workspace),
        "deliverables_found": deliverables,
        "stdout_lines": len(stdout.splitlines()),
        "stderr_lines": len(stderr.splitlines()),
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }

    # Save raw stdout/stderr for debugging
    (workspace / "_agent_stdout.txt").write_text(stdout, encoding="utf-8")
    (workspace / "_agent_stderr.txt").write_text(stderr, encoding="utf-8")
    (workspace / "_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Update progress
    if deliverables:
        # Accept deliverables even if process timed out (rc=-1) or had non-zero exit
        print(f"  [OK] Completed in {elapsed:.1f}s. Deliverables: {deliverables}")
        progress["completed"].append(task_id)
    elif rc == 0:
        print(f"  [WARN] Completed in {elapsed:.1f}s but no deliverables found.")
        progress["failed"].append(task_id)
    else:
        print(f"  [FAIL] Exit code {rc} after {elapsed:.1f}s.")
        progress["failed"].append(task_id)

    save_progress(progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description="Run GDPval tasks with codelet")
    parser.add_argument("--start", type=int, default=0, help="Start task index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End task index (exclusive)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per task")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without running")
    args = parser.parse_args()

    # Ensure results dir exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks
    df = load_tasks()
    total = len(df)
    print(f"Loaded {total} GDPval tasks from {PARQUET_PATH}")

    # Determine range
    start = max(0, args.start)
    end = min(total, args.end) if args.end is not None else total
    if args.max_tasks is not None:
        end = min(end, start + args.max_tasks)

    print(f"Running tasks {start} to {end - 1} ({end - start} tasks)")

    if args.dry_run:
        for i in range(start, end):
            row = df.iloc[i]
            print(f"  {i:3d}: {row['task_id'][:8]} | {row['occupation'][:40]:40s} | {row['sector'][:30]}")
        return

    # Load progress
    progress = load_progress()
    print(f"Progress: {len(progress['completed'])} completed, {len(progress['failed'])} failed, {len(progress['skipped'])} skipped")

    # Run tasks
    for i in range(start, end):
        try:
            progress = run_single_task(df, i, progress, args.max_steps)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")
            save_progress(progress)
            sys.exit(130)
        except Exception as exc:
            print(f"  [ERROR] Unexpected error: {exc}")
            progress["failed"].append(df.iloc[i]["task_id"])
            save_progress(progress)

    # Final summary
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"  Completed: {len(progress['completed'])}")
    print(f"  Failed:    {len(progress['failed'])}")
    print(f"  Skipped:   {len(progress['skipped'])}")
    print(f"  Total:     {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

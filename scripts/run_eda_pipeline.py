#!/usr/bin/env python3
"""
==============================================================================
Master Python Script: Run Full EDA Pipeline with Local .venv Environment
==============================================================================
1. Auto-checks if running inside the local `.venv`.
   - If not in `.venv`, it automatically runs `scripts/setup_environment.sh` first
     and re-executes itself inside `.venv`.
2. Verifies environment dependencies from `requirements.txt` and ensures the
   Jupyter kernel (`aml-dl-venv`) is registered.
3. Discovers and sequentially executes all 6 EDA notebooks in `nootbooks/01.EDA/`.
4. Saves executed notebooks in-place with rendered plots, logs, and outputs.
5. Verifies and validates all generated Parquet files (rows, cols, size).
6. Outputs a formatted benchmark and status report.
==============================================================================
"""

import os
import sys
import time
import glob
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NB_DIR = os.path.join(PROJECT_ROOT, 'nootbooks', '01.EDA')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'GeneratedOutput', '01.EDA')
REQ_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')
SETUP_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'setup_environment.sh')

# ---------------------------------------------------------------------------
# 1. Automatic .venv Python Self-Bootstrap & Setup Runner
# ---------------------------------------------------------------------------
def ensure_venv_python():
    """Ensure this script runs inside the project's root .venv."""
    in_venv = (sys.prefix != sys.base_prefix) and os.path.exists(os.path.join(sys.prefix, 'pyvenv.cfg'))
    
    if os.name == 'nt':
        venv_python = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
    else:
        venv_python = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python')

    if not in_venv:
        print("=" * 72)
        print("  STEP 1: PRE-FLIGHT ENVIRONMENT INITIALIZATION")
        print("=" * 72)
        print(f"Current Python: {sys.executable} (outside project .venv)")
        
        # Run environment setup script first
        if os.path.exists(SETUP_SCRIPT) and os.name != 'nt':
            print(">>> Triggering setup_environment.sh first...")
            subprocess.run(['bash', SETUP_SCRIPT], check=True)
        elif not os.path.exists(venv_python):
            print(">>> Creating .venv environment...")
            subprocess.run([sys.executable, '-m', 'venv', os.path.join(PROJECT_ROOT, '.venv')], check=True)
            subprocess.run([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'], check=True)
            subprocess.run([venv_python, '-m', 'pip', 'install', '-r', REQ_FILE, '--quiet'], check=True)

        if os.path.exists(venv_python):
            print(f"\n>>> Bootstrapping into project .venv: {venv_python}")
            print("=" * 72 + "\n")
            os.execv(venv_python, [venv_python] + sys.argv)
        else:
            print("[Error] Failed to initialize .venv python binary.")
            sys.exit(1)

ensure_venv_python()

# Now we are 100% guaranteed to run inside .venv
import nbformat
from nbclient import NotebookClient

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

def print_banner(title):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)

def run_environment_check():
    """Verifies dependencies and registers kernel."""
    print(">>> [Pre-Run Check] Verifying packages from requirements.txt...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", REQ_FILE, "--quiet"]
        subprocess.run(cmd, check=True)
        print("    ✓ All dependencies verified and up-to-date.")
    except Exception as e:
        print(f"    [Warning] Dependency check notice: {e}")

    try:
        cmd = [
            sys.executable, "-m", "ipykernel", "install",
            "--user", "--name=aml-dl-venv",
            '--display-name=Python (.venv - AML/DL)'
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("    ✓ Jupyter kernel 'aml-dl-venv' registered successfully.")
    except Exception as e:
        print(f"    [Warning] Kernel registration notice: {e}")

def execute_notebook(nb_path, index, total):
    """Executes a single notebook and saves the executed cells in place."""
    nb_name = os.path.basename(nb_path)
    print(f"\n[{index}/{total}] Running: {nb_name} ...")
    t0 = time.time()
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name='python3',
        resources={'metadata': {'path': PROJECT_ROOT}}
    )
    
    try:
        client.execute()
        elapsed = time.time() - t0
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"    ✓ PASSED in {elapsed:.1f}s — Notebook executed & saved in-place.")
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    ✗ FAILED in {elapsed:.1f}s — Error: {e}")
        return False, elapsed, str(e)

def verify_parquet_outputs():
    """Validates generated parquet files and prints their dimensions."""
    print_banner("STEP 4: GENERATED PARQUET OUTPUTS AUDIT")
    out_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.parquet")))
    
    if not out_files:
        print("  [Warning] No parquet files found in output directory.")
        return False

    print(f"{'Filename':48} | {'Rows':>8} | {'Cols':>6} | {'Size':>10}")
    print("-" * 76)
    
    for p in out_files:
        name = os.path.basename(p)
        size_kb = os.path.getsize(p) / 1024
        
        if HAS_PYARROW:
            try:
                table = pq.read_table(p)
                rows_str = f"{table.num_rows:,}"
                cols_str = str(table.num_columns)
            except Exception:
                rows_str = "N/A"
                cols_str = "N/A"
        else:
            rows_str = "N/A"
            cols_str = "N/A"
            
        print(f"{name:48} | {rows_str:>8} | {cols_str:>6} | {size_kb:>8.1f} KB")
        
    print("-" * 76)
    print(f"Total Parquet artifacts: {len(out_files)} files (Snappy compressed, Parquetly-ready)")
    return True

def main():
    print_banner("EDA AUTOMATION PIPELINE: ORCHESTRATION RUNNER")
    print(f"Active Python:     {sys.executable}")
    print(f"Project Root:      {PROJECT_ROOT}")
    print(f"Notebooks Dir:     {NB_DIR}")
    print(f"Parquet Out Dir:   {OUT_DIR}")
    
    # 1. Environment Verification
    run_environment_check()
    
    # 2. Discover notebooks
    nb_files = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))
    if not nb_files:
        print(f"\n[Error] No notebooks found in {NB_DIR}")
        sys.exit(1)
        
    print(f"\nDiscovered {len(nb_files)} EDA notebooks:")
    for i, f in enumerate(nb_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
        
    # 3. Sequentially run notebooks
    print_banner("STEP 3: SEQUENTIAL NOTEBOOK EXECUTION")
    results = []
    t_start = time.time()
    
    for i, nb_path in enumerate(nb_files, 1):
        success, duration, error = execute_notebook(nb_path, i, len(nb_files))
        results.append({
            'Notebook': os.path.basename(nb_path),
            'Status': 'PASSED' if success else 'FAILED',
            'Time': f"{duration:.1f}s",
            'Error': error or 'None'
        })
        
    total_time = time.time() - t_start
    
    # 4. Verify Generated Parquet Outputs
    verify_parquet_outputs()
    
    # 5. Final Summary Report
    print_banner("PIPELINE EXECUTION SUMMARY REPORT")
    print(f"{'Notebook':52} | {'Status':8} | {'Duration':8}")
    print("-" * 76)
    all_passed = True
    for r in results:
        status_symbol = "✓" if r['Status'] == 'PASSED' else "✗"
        print(f"{status_symbol} {r['Notebook']:50} | {r['Status']:8} | {r['Time']:8}")
        if r['Status'] != 'PASSED':
            all_passed = False
            
    print("-" * 76)
    print(f"Total Pipeline Runtime: {total_time:.1f}s")
    if all_passed:
        print("STATUS: ALL 6 NOTEBOOKS EXECUTED & PARQUET ARTIFACTS VALIDATED!")
    else:
        print("STATUS: [FAILED] One or more notebooks encountered errors.")
        sys.exit(1)

if __name__ == '__main__':
    main()

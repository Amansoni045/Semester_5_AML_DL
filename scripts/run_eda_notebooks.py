#!/usr/bin/env python3
"""
Master EDA Notebook Runner.
Executes all 6 pure EDA notebooks in sequence using the project virtual environment (.venv),
captures outputs, verifies generated parquet files, and reports detailed benchmark timings.
"""

import os
import sys
import time
import glob
import subprocess
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NB_DIR = os.path.join(PROJECT_ROOT, 'nootbooks', '01.EDA')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'GeneratedOutput', '01.EDA')
REQ_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def verify_and_update_env():
    """Ensure dependencies from requirements.txt are installed."""
    print(">>> Checking and updating virtual environment dependencies...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        cmd = [sys.executable, "-m", "pip", "install", "-r", REQ_FILE]
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("    Virtual environment dependencies are up to date.")
    except Exception as e:
        print(f"    Warning during environment update: {e}")

def run_notebook(nb_path):
    """Execute a single notebook and save in place with fresh outputs."""
    nb_name = os.path.basename(nb_path)
    print(f"\n--- Running: {nb_name} ---")
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
        
        # Save executed notebook in-place
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"    PASSED ({elapsed:.1f}s) — Saved executed notebook.")
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED ({elapsed:.1f}s) — Error: {e}")
        return False, elapsed, str(e)

def main():
    print_header("EDA PIPELINE: AUTOMATED EXECUTION SUITE")
    print(f"Python Binary:  {sys.executable}")
    print(f"Project Root:   {PROJECT_ROOT}")
    print(f"Notebooks Dir:  {NB_DIR}")
    print(f"Output Data Dir:{OUT_DIR}")
    
    # Step 1: Verify & Update Environment
    verify_and_update_env()
    
    # Step 2: Find all notebooks
    nb_files = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))
    if not nb_files:
        print(f"\n[Error] No notebooks found in {NB_DIR}")
        sys.exit(1)
        
    print(f"\nFound {len(nb_files)} EDA notebooks to execute:")
    for f in nb_files:
        print(f"  • {os.path.basename(f)}")
        
    # Step 3: Run notebooks one by one
    results = []
    t_start = time.time()
    
    for nb_path in nb_files:
        success, duration, error = run_notebook(nb_path)
        results.append({
            'Notebook': os.path.basename(nb_path),
            'Status': 'PASSED' if success else 'FAILED',
            'Time (s)': f"{duration:.1f}s",
            'Error': error or 'None'
        })
        
    total_time = time.time() - t_start
    
    # Step 4: Verify Output Parquet Files
    print_header("GENERATED DATA OUTPUTS VERIFICATION")
    out_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.parquet")))
    if out_files:
        for p in out_files:
            size_kb = os.path.getsize(p) / 1024
            print(f"  ✓ {os.path.basename(p):55} ({size_kb:.1f} KB)")
    else:
        print("  [Warning] No parquet files found in output directory.")
        
    # Step 5: Summary Table
    print_header("EXECUTION SUMMARY REPORT")
    print(f"{'Notebook':55} | {'Status':8} | {'Duration':8}")
    print("-" * 75)
    all_passed = True
    for r in results:
        print(f"{r['Notebook']:55} | {r['Status']:8} | {r['Time (s)']:8}")
        if r['Status'] != 'PASSED':
            all_passed = False
            
    print("-" * 75)
    print(f"Total Pipeline Runtime: {total_time:.1f}s")
    if all_passed:
        print("ALL NOTEBOOKS EXECUTED AND VERIFIED SUCCESSFULLY!")
    else:
        print("[Warning] One or more notebooks failed. See logs above.")
        sys.exit(1)

if __name__ == '__main__':
    main()

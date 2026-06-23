#!/usr/bin/env python3
"""
=============================================================
MASTER PIPELINE — RdRp Cryptic Allosteric Sites Study
=============================================================
Runs all phases in sequence. Tracks status so you can
resume from any point without re-running completed phases.

Usage:
  python run_pipeline.py              # Run all phases
  python run_pipeline.py --from 05   # Resume from Phase 05
  python run_pipeline.py --only 01   # Run only Phase 01
  python run_pipeline.py --status    # Show what's been done
=============================================================
"""

import os
import sys
import json
import argparse
import subprocess
import datetime

STATUS_FILE = "logs/pipeline_status.json"
os.makedirs("logs", exist_ok=True)

PHASES = [
    {
        "num"     : "01",
        "name"    : "Structure Download & QC",
        "script"  : "01_download_qc.py",
        "outputs" : ["structures/raw/6QPF.pdb", "structures/raw/4C11.pdb",
                     "structures/raw/9GJT.pdb", "structures/raw/6PZK.pdb",
                     "structures/raw/3BSO.pdb"],
        "time_est": "5 min"
    },
    {
        "num"     : "02",
        "name"    : "RdRp Domain Extraction",
        "script"  : "02_extract_domains.py",
        "outputs" : ["structures/domains/H5N1_Influenza_RdRp_domain.pdb"],
        "time_est": "2 min"
    },
    {
        "num"     : "03",
        "name"    : "Structure Preparation",
        "script"  : "03_prepare_structures.py",
        "outputs" : ["structures/prepared/H5N1_Influenza_prepared.pdb"],
        "time_est": "15 min"
    },
    {
        "num"     : "04",
        "name"    : "GROMACS MD Simulation",
        "script"  : "04_gromacs_md.sh",
        "outputs" : ["md/H5N1_Influenza/md_prod_noPBC.xtc"],
        "time_est": "24-72 hrs per structure (GPU recommended)",
        "is_bash" : True,
        "hpc_note": "Submit to HPC — run all 5 structures in PARALLEL"
    },
    {
        "num"     : "05",
        "name"    : "Cryptic Pocket Detection",
        "script"  : "05_pocket_detection.py",
        "outputs" : ["pockets/pocket_catalogue.csv"],
        "time_est": "1-2 hrs"
    },
    {
        "num"     : "06",
        "name"    : "Sequence Analysis & ConSurf",
        "script"  : "06_sequence_analysis.py",
        "outputs" : ["sequences/alignments/"],
        "time_est": "30 min + ConSurf web submission (2-4 hrs)"
    },
    {
        "num"     : "07_08",
        "name"    : "Foldseek + ML Druggability",
        "script"  : "07_08_foldseek_druggability.py",
        "outputs" : ["pockets/ml_scoring/pocket_druggability_scores.csv"],
        "time_est": "1 hr"
    },
    {
        "num"     : "09",
        "name"    : "Virtual Screening (AutoDock Vina)",
        "script"  : "09_virtual_screening.py",
        "outputs" : ["screening/results/docking_hits.csv"],
        "time_est": "2-4 hrs"
    },
    {
        "num"     : "10_11",
        "name"    : "ADME Profiling + Figure Generation",
        "script"  : "10_11_adme_figures.py",
        "outputs" : ["figures/summary_heatmap.png", "figures/adme_radar_plots.png"],
        "time_est": "1 hr"
    }
]


def load_status() -> dict:
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_status(status: dict):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def check_outputs_exist(phase: dict) -> bool:
    """Check if phase outputs already exist."""
    outputs = phase.get('outputs', [])
    return all(os.path.exists(o) for o in outputs if o)


def run_phase(phase: dict) -> bool:
    """Execute a single pipeline phase."""
    script = phase['script']
    is_bash = phase.get('is_bash', False)

    print(f"\n{'='*60}")
    print(f"  PHASE {phase['num']}: {phase['name']}")
    print(f"  Estimated time: {phase['time_est']}")
    print(f"{'='*60}")

    if not os.path.exists(script):
        print(f"  [ERROR] Script not found: {script}")
        return False

    if is_bash:
        cmd = ["bash", script]
    else:
        cmd = [sys.executable, script]

    print(f"  Running: {' '.join(cmd)}\n")
    start_time = datetime.datetime.now()

    result = subprocess.run(cmd, text=True)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    success = result.returncode == 0

    if success:
        print(f"\n  ✓ Phase {phase['num']} completed in {duration:.0f}s")
    else:
        print(f"\n  ✗ Phase {phase['num']} failed (return code: {result.returncode})")

    return success


def print_status(status: dict):
    """Print pipeline progress summary."""
    print("\n" + "="*65)
    print("  RdRp PIPELINE STATUS")
    print("="*65)
    print(f"  {'Phase':<8} {'Name':<35} {'Status':<12} {'Time'}")
    print(f"  {'─'*7} {'─'*34} {'─'*11} {'─'*12}")

    for phase in PHASES:
        num    = phase['num']
        name   = phase['name'][:34]
        done   = status.get(num, {}).get('done', False)
        ts     = status.get(num, {}).get('timestamp', '')[:16]
        exists = check_outputs_exist(phase)

        if done:
            symbol = "✓ DONE"
        elif exists:
            symbol = "~ SKIP (cached)"
        else:
            symbol = "○ Pending"

        print(f"  {num:<8} {name:<35} {symbol:<14} {ts}")

    print("="*65)

    # HPC note
    phase04_done = status.get('04', {}).get('done', False)
    if not phase04_done:
        print("""
  ⚡ HPC SUBMISSION NOTE (Phase 04):
     Submit each structure as a separate SLURM job:
     sbatch --gres=gpu:1 --time=72:00:00 04_gromacs_md.sh H5N1_Influenza
     sbatch --gres=gpu:1 --time=72:00:00 04_gromacs_md.sh Dengue_DENV3
     sbatch --gres=gpu:1 --time=72:00:00 04_gromacs_md.sh Nipah_Virus
     sbatch --gres=gpu:1 --time=72:00:00 04_gromacs_md.sh RSV
     sbatch --gres=gpu:1 --time=72:00:00 04_gromacs_md.sh Norovirus
        """)


def main():
    parser = argparse.ArgumentParser(description='RdRp Pipeline Master Runner')
    parser.add_argument('--from',   dest='from_phase', help='Resume from phase number (e.g. 05)')
    parser.add_argument('--only',   dest='only_phase', help='Run only one phase (e.g. 01)')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    parser.add_argument('--force',  action='store_true', help='Re-run even if outputs exist')
    args = parser.parse_args()

    status = load_status()

    if args.status:
        print_status(status)
        return

    print("\n" + "="*65)
    print("  RdRp CRYPTIC ALLOSTERIC SITES — MASTER PIPELINE")
    print("  PI: [Your Name] | Co-PI: Claude (Mentor)")
    print("="*65)

    phases_to_run = PHASES

    if args.only_phase:
        phases_to_run = [p for p in PHASES if p['num'] == args.only_phase]
        if not phases_to_run:
            print(f"Phase '{args.only_phase}' not found.")
            return

    if args.from_phase:
        start_idx = next(
            (i for i, p in enumerate(PHASES) if p['num'] == args.from_phase), 0
        )
        phases_to_run = PHASES[start_idx:]

    for phase in phases_to_run:
        num = phase['num']

        # Skip if already done and outputs exist
        already_done = status.get(num, {}).get('done', False)
        outputs_exist = check_outputs_exist(phase)

        if already_done and outputs_exist and not args.force:
            print(f"\n  [SKIP] Phase {num}: {phase['name']} (already complete)")
            continue

        # HPC phases need manual submission
        if phase.get('hpc_note'):
            print(f"\n  [HPC] Phase {num}: {phase['name']}")
            print(f"        {phase['hpc_note']}")
            print(f"        Run manually on HPC, then press Enter to continue...")
            input()
            status[num] = {'done': True, 'timestamp': str(datetime.datetime.now())}
            save_status(status)
            continue

        success = run_phase(phase)
        status[num] = {
            'done'      : success,
            'timestamp' : str(datetime.datetime.now())
        }
        save_status(status)

        if not success:
            print(f"\n  Pipeline stopped at Phase {num}.")
            print(f"  Fix the error and resume with: python run_pipeline.py --from {num}")
            break

    print_status(status)


if __name__ == "__main__":
    main()

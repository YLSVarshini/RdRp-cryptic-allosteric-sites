"""
=============================================================
PHASE 05: Cryptic Pocket Detection
=============================================================
Two complementary approaches:
  A) fpocket on MD trajectory frames — detects pockets that
     APPEAR during simulation but absent in crystal structure
  B) PocketMiner API — ML prediction from static structure
     (orthogonal validation, much faster)

A pocket flagged by BOTH is a high-confidence cryptic site.
=============================================================
Run: python 05_pocket_detection.py
"""

import os
import sys
import json
import subprocess
import requests
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, rmsf
    HAS_MDA = True
except ImportError:
    print("[WARN] MDAnalysis not installed (pip install mdanalysis)")
    HAS_MDA = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
STRUCTURES = {
    "H5N1_Influenza" : {
        "tpr"    : "md/H5N1_Influenza/md_prod.tpr",
        "xtc"    : "md/H5N1_Influenza/md_prod_noPBC.xtc",
        "domain" : "structures/domains/H5N1_Influenza_RdRp_domain.pdb",
        "chain"  : "B"
    },
    "Dengue_DENV3"   : {
        "tpr"    : "md/Dengue_DENV3/md_prod.tpr",
        "xtc"    : "md/Dengue_DENV3/md_prod_noPBC.xtc",
        "domain" : "structures/domains/Dengue_DENV3_RdRp_domain.pdb",
        "chain"  : "A"
    },
    "Nipah_Virus"    : {
        "tpr"    : "md/Nipah_Virus/md_prod.tpr",
        "xtc"    : "md/Nipah_Virus/md_prod_noPBC.xtc",
        "domain" : "structures/domains/Nipah_Virus_RdRp_domain.pdb",
        "chain"  : "A"
    },
    "RSV"            : {
        "tpr"    : "md/RSV/md_prod.tpr",
        "xtc"    : "md/RSV/md_prod_noPBC.xtc",
        "domain" : "structures/domains/RSV_RdRp_domain.pdb",
        "chain"  : "A"
    },
    "Norovirus"      : {
        "tpr"    : "md/Norovirus/md_prod.tpr",
        "xtc"    : "md/Norovirus/md_prod_noPBC.xtc",
        "domain" : "structures/domains/Norovirus_RdRp_domain.pdb",
        "chain"  : "A"
    }
}

POCKET_DIR  = "pockets"
FRAMES_DIR  = "pockets/frames"
FPOCKET_DIR = "pockets/fpocket"
PM_DIR      = "pockets/pocketminer"
FIGURES_DIR = "figures"

for d in [POCKET_DIR, FRAMES_DIR, FPOCKET_DIR, PM_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Cryptic pocket thresholds
DRUGGABILITY_CUTOFF = 0.5    # fpocket druggability score
FREQUENCY_CUTOFF    = 0.10   # pocket must appear in >10% of frames
VOLUME_MIN          = 300    # Å³ — minimum viable drug-binding volume
VOLUME_MAX          = 2000   # Å³ — too large = not specific

# ─────────────────────────────────────────────
# PART A: FPOCKET ON MD TRAJECTORY
# ─────────────────────────────────────────────

def extract_frames(name: str, info: dict, every_nth: int = 10) -> list:
    """
    Extract every Nth frame from MD trajectory as individual PDB files.
    For 2000 frames total, every_nth=10 gives 200 frames for fpocket.
    """
    frame_dir = os.path.join(FRAMES_DIR, name)
    os.makedirs(frame_dir, exist_ok=True)

    if not HAS_MDA:
        print(f"  [SKIP] MDAnalysis not available")
        return []

    tpr = info['tpr']
    xtc = info['xtc']

    if not os.path.exists(tpr) or not os.path.exists(xtc):
        print(f"  [WARN] Trajectory files not found — using crystal structure only")
        # Fallback: run fpocket on crystal structure alone
        return [info['domain']]

    print(f"  Loading trajectory...")
    u = mda.Universe(tpr, xtc)
    protein = u.select_atoms("protein")

    n_frames  = len(u.trajectory)
    frames_to_use = list(range(0, n_frames, every_nth))
    print(f"  Trajectory: {n_frames} frames total")
    print(f"  Extracting every {every_nth}th frame: {len(frames_to_use)} frames")

    frame_files = []
    for i, frame_idx in enumerate(frames_to_use):
        u.trajectory[frame_idx]
        out_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.pdb")
        protein.write(out_path)
        frame_files.append(out_path)

        if i % 50 == 0:
            print(f"  Extracted {i+1}/{len(frames_to_use)} frames...")

    print(f"  [OK] {len(frame_files)} frames extracted → {frame_dir}/")
    return frame_files


def run_fpocket_on_frame(frame_pdb: str) -> dict:
    """Run fpocket on a single PDB frame and parse results."""
    if not os.path.exists(frame_pdb):
        return {}

    # Run fpocket
    result = subprocess.run(
        ["fpocket", "-f", frame_pdb],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        return {}

    # Parse fpocket output
    # fpocket creates a directory: {pdb_name}_out/
    base     = os.path.splitext(frame_pdb)[0]
    out_dir  = f"{base}_out"
    info_file = os.path.join(out_dir, f"{os.path.basename(base)}_info.txt")

    pockets = []
    if os.path.exists(info_file):
        pockets = parse_fpocket_info(info_file)

    # Clean up fpocket output to save disk space
    # subprocess.run(["rm", "-rf", out_dir])  # Uncomment to save space

    return pockets


def parse_fpocket_info(info_file: str) -> list:
    """Parse fpocket _info.txt file to extract pocket properties."""
    pockets = []
    current_pocket = {}

    with open(info_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("Pocket"):
                if current_pocket:
                    pockets.append(current_pocket)
                pocket_num = int(line.split()[1])
                current_pocket = {"pocket_id": pocket_num}

            elif ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                val = val.strip()
                try:
                    current_pocket[key] = float(val)
                except ValueError:
                    current_pocket[key] = val

    if current_pocket:
        pockets.append(current_pocket)

    return pockets


def run_fpocket_all_frames(name: str, frame_files: list) -> pd.DataFrame:
    """Run fpocket on all frames and aggregate results."""
    print(f"\n  Running fpocket on {len(frame_files)} frames...")
    print(f"  (This may take 10-30 minutes depending on system size)")

    all_pockets = []

    for i, frame_pdb in enumerate(frame_files):
        pockets = run_fpocket_on_frame(frame_pdb)

        for pocket in pockets:
            pocket['frame']  = i
            pocket['source'] = name
            all_pockets.append(pocket)

        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(frame_files)} frames...")

    if not all_pockets:
        print(f"  [WARN] No pocket data — fpocket may not be installed")
        print(f"         Install: sudo apt install fpocket")
        print(f"         Or: https://github.com/Discngine/fpocket")
        return pd.DataFrame()

    df = pd.DataFrame(all_pockets)
    print(f"  [OK] {len(df)} total pocket detections across {len(frame_files)} frames")
    return df


def identify_cryptic_pockets(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Identify cryptic pockets from fpocket trajectory data.
    Cryptic = druggable pocket appearing transiently (not in crystal structure).
    """
    if df.empty:
        return pd.DataFrame()

    # Key fpocket columns (names may vary by version)
    vol_col    = next((c for c in df.columns if 'volume' in c.lower()), None)
    drug_col   = next((c for c in df.columns if 'drug' in c.lower()), None)
    score_col  = next((c for c in df.columns if 'score' in c.lower()), None)

    if vol_col is None:
        print(f"  [WARN] Could not find volume column in fpocket output")
        print(f"  Columns found: {list(df.columns)}")
        return df

    # Filter for druggable pockets
    druggable = df.copy()
    if drug_col:
        druggable = druggable[druggable[drug_col] >= DRUGGABILITY_CUTOFF]
    if vol_col:
        druggable = druggable[
            (druggable[vol_col] >= VOLUME_MIN) &
            (druggable[vol_col] <= VOLUME_MAX)
        ]

    # Calculate pocket frequency (how often does this pocket appear?)
    n_frames = df['frame'].nunique()
    pocket_freq = druggable.groupby('pocket_id').size() / n_frames

    frequent_pockets = pocket_freq[pocket_freq >= FREQUENCY_CUTOFF].index
    cryptic = druggable[druggable['pocket_id'].isin(frequent_pockets)].copy()
    cryptic['frequency'] = cryptic['pocket_id'].map(pocket_freq)

    print(f"  Druggable pockets detected: {len(druggable['pocket_id'].unique())}")
    print(f"  Cryptic candidates (freq > {FREQUENCY_CUTOFF*100:.0f}%): {len(frequent_pockets)}")

    return cryptic


def plot_pocket_frequency(all_results: dict, name: str):
    """Plot pocket frequency across trajectory — key result figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cryptic Pocket Detection: {name}", fontsize=14, fontweight='bold')

    if all_results.get('fpocket_data') is not None and not all_results['fpocket_data'].empty:
        df = all_results['fpocket_data']
        vol_col = next((c for c in df.columns if 'volume' in c.lower()), None)

        if vol_col:
            # Panel A: Pocket volume distribution
            axes[0].hist(df[vol_col], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
            axes[0].axvline(VOLUME_MIN, color='red', linestyle='--', label=f'Min ({VOLUME_MIN} Å³)')
            axes[0].axvline(VOLUME_MAX, color='red', linestyle='--', label=f'Max ({VOLUME_MAX} Å³)')
            axes[0].set_xlabel("Pocket Volume (Å³)", fontsize=12)
            axes[0].set_ylabel("Count", fontsize=12)
            axes[0].set_title("A. Pocket Volume Distribution", fontsize=12)
            axes[0].legend()

    # Panel B: Pocket frequency
    if all_results.get('cryptic_pockets') is not None and not all_results['cryptic_pockets'].empty:
        cp = all_results['cryptic_pockets']
        pocket_freq = cp.groupby('pocket_id')['frequency'].mean().sort_values(ascending=False)

        bars = axes[1].bar(range(len(pocket_freq)), pocket_freq.values,
                           color=plt.cm.RdYlGn(pocket_freq.values), edgecolor='grey', linewidth=0.5)
        axes[1].axhline(FREQUENCY_CUTOFF, color='red', linestyle='--',
                        label=f'Threshold ({FREQUENCY_CUTOFF*100:.0f}%)')
        axes[1].set_xlabel("Pocket ID", fontsize=12)
        axes[1].set_ylabel("Frequency (fraction of frames)", fontsize=12)
        axes[1].set_title("B. Cryptic Pocket Frequency", fontsize=12)
        axes[1].set_xticks(range(len(pocket_freq)))
        axes[1].set_xticklabels(pocket_freq.index, rotation=45)
        axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"pocket_frequency_{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [FIGURE] Saved → {out_path}")


# ─────────────────────────────────────────────
# PART B: POCKETMINER API (ML prediction)
# ─────────────────────────────────────────────

def run_pocketminer(name: str, pdb_path: str) -> dict:
    """
    Submit structure to PocketMiner web server.
    PocketMiner (Meller et al., Nature Communications 2023)
    predicts cryptic pocket probability from static structure.
    Returns per-residue cryptic pocket probability scores.
    """
    print(f"\n  Submitting to PocketMiner...")

    if not os.path.exists(pdb_path):
        print(f"  [SKIP] PDB file not found: {pdb_path}")
        return {}

    # PocketMiner API endpoint
    # NOTE: Check https://pocketminer.bio for current API documentation
    API_URL = "https://pocketminer.bio/api/submit"

    try:
        with open(pdb_path, 'rb') as pdb_file:
            response = requests.post(
                API_URL,
                files={"pdb": pdb_file},
                timeout=300
            )

        if response.status_code == 200:
            result = response.json()
            # Save raw result
            out_path = os.path.join(PM_DIR, f"{name}_pocketminer.json")
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  [OK] PocketMiner result → {out_path}")
            return result

        else:
            print(f"  [WARN] PocketMiner API returned {response.status_code}")
            print(f"         Try manual submission at: https://pocketminer.bio")
            return {}

    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Cannot reach PocketMiner API")
        print(f"         Manual submission: https://pocketminer.bio")
        print(f"         Upload file: {pdb_path}")
        return {"status": "manual_submission_required", "file": pdb_path}
    except Exception as e:
        print(f"  [WARN] PocketMiner error: {e}")
        return {}


def parse_pocketminer_results(result: dict, name: str) -> pd.DataFrame:
    """Parse PocketMiner per-residue probability scores."""
    if not result or 'predictions' not in result:
        return pd.DataFrame()

    predictions = result['predictions']
    df = pd.DataFrame(predictions, columns=['residue_id', 'chain', 'resname', 'probability'])
    
    # Flag high-probability cryptic sites (>0.5 = likely cryptic)
    df['cryptic_site'] = df['probability'] > 0.5
    df['source'] = name

    high_prob = df[df['cryptic_site']]
    print(f"  PocketMiner: {len(high_prob)} residues with cryptic probability > 0.5")
    
    return df


# ─────────────────────────────────────────────
# COMPILE RESULTS & SAVE
# ─────────────────────────────────────────────

def compile_pocket_catalogue(all_results: dict) -> pd.DataFrame:
    """
    Compile all pocket detection results into a master table.
    This becomes Table 1 in your Results section.
    """
    rows = []

    for name, data in all_results.items():
        cp = data.get('cryptic_pockets', pd.DataFrame())
        pm = data.get('pocketminer_df', pd.DataFrame())

        n_fpocket = len(cp['pocket_id'].unique()) if not cp.empty else 0
        n_pm      = pm['cryptic_site'].sum() if not pm.empty and 'cryptic_site' in pm.columns else "N/A"

        row = {
            "Virus"                         : name,
            "fpocket_cryptic_pockets"       : n_fpocket,
            "pocketminer_cryptic_residues"  : n_pm,
            "top_pocket_frequency"          : cp['frequency'].max() if not cp.empty else 0,
        }
        rows.append(row)

    catalogue = pd.DataFrame(rows)

    # Save
    out_path = os.path.join(POCKET_DIR, "pocket_catalogue.csv")
    catalogue.to_csv(out_path, index=False)
    print(f"\n[SAVED] Pocket catalogue → {out_path}")
    return catalogue


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PHASE 05: Cryptic Pocket Detection")
    print("="*60)
    print("\n  Method A: fpocket on MD trajectory frames")
    print("  Method B: PocketMiner ML prediction (orthogonal)")
    print("  Confidence: pockets detected by BOTH = high confidence\n")

    all_results = {}

    for name, info in STRUCTURES.items():
        print(f"\n{'─'*50}")
        print(f"  Processing: {name}")
        print(f"{'─'*50}")

        results = {}

        # ── Method A: fpocket ──────────────────────────
        print("\n  [A] fpocket trajectory analysis")
        frame_files    = extract_frames(name, info, every_nth=10)
        fpocket_data   = run_fpocket_all_frames(name, frame_files)
        cryptic_pockets = identify_cryptic_pockets(fpocket_data, name)
        results['fpocket_data']   = fpocket_data
        results['cryptic_pockets'] = cryptic_pockets

        # ── Method B: PocketMiner ─────────────────────
        print("\n  [B] PocketMiner ML prediction")
        pm_result = run_pocketminer(name, info['domain'])
        pm_df     = parse_pocketminer_results(pm_result, name)
        results['pocketminer_result'] = pm_result
        results['pocketminer_df']     = pm_df

        # ── Save per-structure data ────────────────────
        out_csv = os.path.join(FPOCKET_DIR, f"{name}_cryptic_pockets.csv")
        if not cryptic_pockets.empty:
            cryptic_pockets.to_csv(out_csv, index=False)
            print(f"\n  [SAVED] Cryptic pockets → {out_csv}")

        # ── Generate figure ────────────────────────────
        plot_pocket_frequency(results, name)

        all_results[name] = results

    # ── Master catalogue ──────────────────────────────
    catalogue = compile_pocket_catalogue(all_results)

    print("\n" + "="*60)
    print("  PHASE 05 COMPLETE — Pocket Detection Summary")
    print("="*60)
    print(catalogue.to_string(index=False))
    print("\n  Next → Phase 06: Evolutionary Conservation (ConSurf)")
    print()


if __name__ == "__main__":
    main()

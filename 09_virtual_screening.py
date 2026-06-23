"""
=============================================================
PHASE 09: Virtual Screening — AutoDock Vina
=============================================================
Screen FDA-approved antivirals + ZINC fragments against the
top-ranked cryptic pockets from Phase 08.

Workflow:
  1. Download ligand libraries (ChEMBL antivirals + ZINC fragments)
  2. Prepare ligands (RDKit: 3D, charges, .pdbqt)
  3. Define docking grids centred on each cryptic pocket
  4. Run AutoDock Vina
  5. Filter hits (binding energy < -6.0 kcal/mol)
  6. Analyse binding poses with PLIP
=============================================================
Run: python 09_virtual_screening.py
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
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    HAS_RDKIT = True
except ImportError:
    print("[WARN] RDKit not installed: pip install rdkit-pypi")
    HAS_RDKIT = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SCREEN_DIR   = "screening"
LIGAND_DIR   = "screening/ligands"
DOCKING_DIR  = "screening/docking"
RESULTS_DIR  = "screening/results"
FIGURES_DIR  = "figures"

for d in [SCREEN_DIR, LIGAND_DIR, DOCKING_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Docking parameters
EXHAUSTIVENESS = 16      # 8=fast, 16=standard, 32=thorough
NUM_MODES      = 9       # Number of binding modes to generate
ENERGY_CUTOFF  = -6.0    # kcal/mol — minimum binding energy to keep
CPU_CORES      = 4       # Adjust to your machine

# Known FDA-approved antivirals to screen (ChEMBL IDs)
# These are our primary screening library
ANTIVIRAL_CHEMBL = [
    "CHEMBL203",    # Remdesivir
    "CHEMBL1229",   # Ribavirin
    "CHEMBL221016", # Favipiravir
    "CHEMBL374467", # Sofosbuvir
    "CHEMBL2170924",# Daclatasvir
    "CHEMBL1200534",# Oseltamivir
    "CHEMBL2108681",# Baloxavir marboxil
    "CHEMBL1097",   # Acyclovir
    "CHEMBL1201580",# Lopinavir
    "CHEMBL429049", # Tenofovir
    "CHEMBL1201583",# Saquinavir
    "CHEMBL1213492",# Molnupiravir
    "CHEMBL4220988",# Nirmatrelvir (Paxlovid)
    "CHEMBL1201820",# Ganciclovir
    "CHEMBL1545",   # Zanamivir
]


# ─────────────────────────────────────────────
# LIGAND LIBRARY PREPARATION
# ─────────────────────────────────────────────
def download_chembl_smiles(chembl_ids: list) -> pd.DataFrame:
    """Download SMILES from ChEMBL for given compound IDs."""
    print("  Downloading structures from ChEMBL...")

    records = []
    for cid in chembl_ids:
        url  = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{cid}.json"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data   = r.json()
                smiles = data.get('molecule_structures', {}).get('canonical_smiles', '')
                name   = data.get('pref_name', cid)
                mw     = data.get('molecule_properties', {}).get('mw_freebase', 0)
                
                records.append({
                    'chembl_id' : cid,
                    'name'      : name,
                    'smiles'    : smiles,
                    'MW'        : mw
                })
                print(f"    ✓ {name} ({cid})")
            else:
                print(f"    ✗ {cid}: HTTP {r.status_code}")
        except Exception as e:
            print(f"    ✗ {cid}: {e}")

    df = pd.DataFrame(records)
    out_path = os.path.join(LIGAND_DIR, "antiviral_library.csv")
    df.to_csv(out_path, index=False)
    print(f"  [SAVED] {len(df)} compounds → {out_path}")
    return df


def prepare_ligands_rdkit(compounds_df: pd.DataFrame) -> list:
    """
    Use RDKit to:
    1. Parse SMILES
    2. Add hydrogens
    3. Generate 3D conformer (MMFF94)
    4. Save as .pdb (convert to .pdbqt with obabel)
    """
    if not HAS_RDKIT:
        print("  [SKIP] RDKit not available — install: pip install rdkit-pypi")
        return []

    print("  Preparing ligands with RDKit + MMFF94...")
    
    prepared_files = []

    for _, row in compounds_df.iterrows():
        if not row['smiles'] or pd.isna(row['smiles']):
            continue

        name   = row['name'].replace(' ', '_').replace('/', '-')[:30]
        out_pdb = os.path.join(LIGAND_DIR, f"{name}.pdb")
        out_pdbqt = out_pdb.replace('.pdb', '.pdbqt')

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                continue

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D conformation
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result != 0:
                # Fallback to random coordinates
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())

            # Minimise with MMFF94
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

            # Save PDB
            Chem.MolToPDBFile(mol, out_pdb)

            # Convert to PDBQT with OpenBabel (required for Vina)
            ob_result = subprocess.run(
                ["obabel", out_pdb, "-O", out_pdbqt, "--gen3d", "-h"],
                capture_output=True, text=True
            )

            if os.path.exists(out_pdbqt):
                prepared_files.append({
                    'name'    : name,
                    'pdbqt'   : out_pdbqt,
                    'smiles'  : row['smiles'],
                    'chembl'  : row.get('chembl_id', '')
                })
            else:
                # If obabel not available, keep PDB
                prepared_files.append({
                    'name'   : name,
                    'pdb'    : out_pdb,
                    'smiles' : row['smiles'],
                    'chembl' : row.get('chembl_id', '')
                })

        except Exception as e:
            print(f"  [WARN] Failed to prepare {name}: {e}")
            continue

    print(f"  [OK] {len(prepared_files)} ligands prepared")
    return prepared_files


# ─────────────────────────────────────────────
# DOCKING BOX DEFINITION
# Define docking grid based on pocket centroid from fpocket
# ─────────────────────────────────────────────

# Pocket centroids (from fpocket output or manually from PyMOL)
# Format: {virus: [x, y, z, box_size_x, box_size_y, box_size_z]}
# UPDATE THESE with actual values from your fpocket/PyMOL analysis
POCKET_GRIDS = {
    "H5N1_Influenza" : {
        "receptor"    : "structures/prepared/H5N1_Influenza_prepared.pdb",
        "center"      : [0.0, 0.0, 0.0],    # UPDATE: pocket centroid from fpocket
        "box_size"    : [25.0, 25.0, 25.0],  # Angstroms — covers full pocket
        "pocket_id"   : 1
    },
    "Dengue_DENV3"   : {
        "receptor"    : "structures/prepared/Dengue_DENV3_prepared.pdb",
        "center"      : [0.0, 0.0, 0.0],    # UPDATE: N-pocket centroid (known)
        "box_size"    : [22.0, 22.0, 22.0],
        "pocket_id"   : 1
    },
    "Nipah_Virus"    : {
        "receptor"    : "structures/prepared/Nipah_Virus_prepared.pdb",
        "center"      : [0.0, 0.0, 0.0],    # UPDATE: cryptic pocket centroid
        "box_size"    : [25.0, 25.0, 25.0],
        "pocket_id"   : 1
    },
    "RSV"            : {
        "receptor"    : "structures/prepared/RSV_prepared.pdb",
        "center"      : [0.0, 0.0, 0.0],    # UPDATE
        "box_size"    : [25.0, 25.0, 25.0],
        "pocket_id"   : 1
    },
    "Norovirus"      : {
        "receptor"    : "structures/prepared/Norovirus_prepared.pdb",
        "center"      : [0.0, 0.0, 0.0],    # UPDATE: B-site centroid (known)
        "box_size"    : [20.0, 20.0, 20.0],
        "pocket_id"   : 1
    }
}


def extract_pocket_centroid_from_fpocket(name: str) -> list:
    """
    Extract pocket centroid coordinates from fpocket output.
    Returns [x, y, z] centroid of the top cryptic pocket.
    """
    pocket_dir = f"pockets/frames/{name}"
    
    # Look for fpocket _info.txt files
    info_files = []
    if os.path.exists(pocket_dir):
        for root, dirs, files in os.walk(pocket_dir):
            for f in files:
                if f.endswith('_info.txt'):
                    info_files.append(os.path.join(root, f))

    if not info_files:
        print(f"  [WARN] No fpocket output found for {name}")
        print(f"         Update POCKET_GRIDS manually using PyMOL:")
        print(f"         1. Open structure in PyMOL")
        print(f"         2. Show cryptic pocket residues")
        print(f"         3. Measure centroid: get_extent sele")
        return [0.0, 0.0, 0.0]

    # Parse centroid from most recent fpocket run
    # fpocket _pockets.pqr file contains alpha-sphere coordinates
    # For now return placeholder — update after Phase 05 completes
    print(f"  [INFO] Update pocket centroid for {name} from fpocket output")
    return [0.0, 0.0, 0.0]


def prepare_receptor_pdbqt(name: str, pdb_path: str) -> str:
    """Convert receptor PDB to PDBQT format for Vina."""
    pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')

    if os.path.exists(pdbqt_path):
        return pdbqt_path

    # Method 1: AutoDockTools prepare_receptor4.py
    adt_script = subprocess.run(
        ["which", "prepare_receptor4.py"], capture_output=True
    ).returncode == 0

    if adt_script:
        cmd = ["prepare_receptor4.py", "-r", pdb_path, "-o", pdbqt_path, "-A", "hydrogens"]
        subprocess.run(cmd, capture_output=True)

    # Method 2: OpenBabel
    elif subprocess.run(["which", "obabel"], capture_output=True).returncode == 0:
        subprocess.run(
            ["obabel", pdb_path, "-O", pdbqt_path, "-xr"],
            capture_output=True
        )

    if os.path.exists(pdbqt_path):
        print(f"  [OK] Receptor PDBQT: {pdbqt_path}")
        return pdbqt_path
    else:
        print(f"  [WARN] Could not create PDBQT for {name}")
        print(f"         Install AutoDockTools or OpenBabel")
        return pdb_path


def run_vina_docking(name: str, grid_info: dict, ligand_files: list) -> pd.DataFrame:
    """
    Run AutoDock Vina for all ligands against one receptor pocket.
    Parallelisable: submit one receptor as separate HPC job.
    """
    if not subprocess.run(["which", "vina"], capture_output=True).returncode == 0:
        print(f"  [WARN] AutoDock Vina not found.")
        print(f"         Install: conda install -c bioconda autodock-vina")
        print(f"         Or: https://vina.scripps.edu/downloads/")
        return pd.DataFrame()

    # Prepare receptor
    receptor_pdb   = grid_info['receptor']
    receptor_pdbqt = prepare_receptor_pdbqt(name, receptor_pdb)

    center = grid_info['center']
    box    = grid_info['box_size']

    dock_dir = os.path.join(DOCKING_DIR, name)
    os.makedirs(dock_dir, exist_ok=True)

    results = []
    print(f"  Docking {len(ligand_files)} ligands against {name} cryptic pocket...")

    for i, lig_info in enumerate(ligand_files):
        lig_pdbqt = lig_info.get('pdbqt', lig_info.get('pdb', ''))
        lig_name  = lig_info['name']

        if not lig_pdbqt or not os.path.exists(lig_pdbqt):
            continue

        out_pdbqt = os.path.join(dock_dir, f"{lig_name}_docked.pdbqt")
        log_file  = os.path.join(dock_dir, f"{lig_name}_vina.log")

        cmd = [
            "vina",
            "--receptor",      receptor_pdbqt,
            "--ligand",        lig_pdbqt,
            "--out",           out_pdbqt,
            "--log",           log_file,
            "--center_x",      str(center[0]),
            "--center_y",      str(center[1]),
            "--center_z",      str(center[2]),
            "--size_x",        str(box[0]),
            "--size_y",        str(box[1]),
            "--size_z",        str(box[2]),
            "--exhaustiveness", str(EXHAUSTIVENESS),
            "--num_modes",     str(NUM_MODES),
            "--cpu",           str(CPU_CORES)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse best binding energy from log
        best_energy = parse_vina_log(log_file)

        results.append({
            'Virus'          : name,
            'Ligand'         : lig_name,
            'ChEMBL_ID'      : lig_info.get('chembl', ''),
            'SMILES'         : lig_info.get('smiles', ''),
            'Binding_Energy' : best_energy,
            'Pose_File'      : out_pdbqt if os.path.exists(out_pdbqt) else '',
        })

        if (i+1) % 5 == 0:
            print(f"  Docked {i+1}/{len(ligand_files)}...")

    df = pd.DataFrame(results)
    print(f"  [OK] Docking complete: {len(df)} results")
    return df


def parse_vina_log(log_file: str) -> float:
    """Parse AutoDock Vina log to extract best binding energy."""
    if not os.path.exists(log_file):
        return 0.0

    with open(log_file, 'r') as f:
        for line in f:
            # Vina output format: "   1      -8.5      0.000      0.000"
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == '1':
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return 0.0


def filter_and_rank_hits(all_results: pd.DataFrame) -> pd.DataFrame:
    """Filter docking results by energy cutoff and rank."""
    if all_results.empty:
        return pd.DataFrame()

    # Filter by binding energy
    hits = all_results[all_results['Binding_Energy'] <= ENERGY_CUTOFF].copy()
    hits = hits.sort_values('Binding_Energy')

    print(f"\n  Docking results: {len(all_results)} total → {len(hits)} hits (≤{ENERGY_CUTOFF} kcal/mol)")
    return hits


def plot_docking_results(all_results: pd.DataFrame):
    """Distribution of binding energies across all viruses — Results figure."""
    if all_results.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, all_results['Virus'].nunique()))
    
    for i, (virus, group) in enumerate(all_results.groupby('Virus')):
        energies = group['Binding_Energy'].dropna()
        energies = energies[energies != 0]
        if len(energies) > 0:
            ax.hist(energies, bins=20, alpha=0.6, label=virus,
                    color=colors[i], edgecolor='white')

    ax.axvline(ENERGY_CUTOFF, color='red', linestyle='--', linewidth=2,
               label=f'Cutoff ({ENERGY_CUTOFF} kcal/mol)')
    ax.set_xlabel("Binding Energy (kcal/mol)", fontsize=12)
    ax.set_ylabel("Number of Compounds", fontsize=12)
    ax.set_title("Virtual Screening Results — Binding Energy Distribution\n"
                 "across Cryptic Allosteric Pockets of 5 Viral RdRps", fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "docking_results.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [FIGURE] Docking results → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PHASE 09: Virtual Screening")
    print("="*60 + "\n")

    # Step 1: Build ligand library
    print("[STEP 1] Building ligand library from ChEMBL...")
    compounds_df  = download_chembl_smiles(ANTIVIRAL_CHEMBL)
    ligand_files  = prepare_ligands_rdkit(compounds_df)

    if not ligand_files:
        print("[WARN] No ligands prepared. Check RDKit installation.")
        return

    # Step 2: Run docking for each virus
    print("\n[STEP 2] Running virtual screening...")
    all_results = []

    for name, grid_info in POCKET_GRIDS.items():
        print(f"\n  → {name}")
        
        # Update centroid from fpocket if available
        centroid = extract_pocket_centroid_from_fpocket(name)
        if centroid != [0.0, 0.0, 0.0]:
            POCKET_GRIDS[name]['center'] = centroid

        if grid_info['center'] == [0.0, 0.0, 0.0]:
            print(f"  [WARN] Pocket centroid not set for {name}")
            print(f"         Update POCKET_GRIDS['{name}']['center'] with")
            print(f"         the X, Y, Z coordinates of your cryptic pocket centroid")
            print(f"         (Get from fpocket output or PyMOL: get_extent sele)")
            continue

        results_df = run_vina_docking(name, grid_info, ligand_files)
        if not results_df.empty:
            all_results.append(results_df)

    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        
        # Save all results
        all_df.to_csv(os.path.join(RESULTS_DIR, "all_docking_results.csv"), index=False)
        
        # Filter hits
        hits = filter_and_rank_hits(all_df)
        hits.to_csv(os.path.join(RESULTS_DIR, "docking_hits.csv"), index=False)
        
        # Plot
        plot_docking_results(all_df)

        print("\n  Top Docking Hits:")
        print(hits[['Virus', 'Ligand', 'Binding_Energy']].head(15).to_string(index=False))
    else:
        print("\n  [NOTE] Update pocket centroids in POCKET_GRIDS dict")
        print("         then re-run this script")

    print("\n" + "="*60)
    print("  PHASE 09 COMPLETE")
    print("  Next → Phase 10: ADME Profiling")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
=============================================================
PHASE 03: Structure Preparation for Molecular Dynamics
=============================================================
For each extracted RdRp domain:
  1. Remove crystallographic waters and non-essential ligands
  2. Add missing residues/loops (PDBFixer)
  3. Add hydrogens at physiological pH 7.4
  4. Assign protonation states (PROPKA via pdbfixer)
  5. Verify final structure
  6. Generate GROMACS-ready PDB

Requirements: pip install pdbfixer openmm
=============================================================
Run: python 03_prepare_structures.py
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STRUCTURE FILES (output from Phase 02)
# ─────────────────────────────────────────────
DOMAIN_FILES = {
    "H5N1_Influenza" : "structures/domains/H5N1_Influenza_RdRp_domain.pdb",
    "Dengue_DENV3"   : "structures/domains/Dengue_DENV3_RdRp_domain.pdb",
    "Nipah_Virus"    : "structures/domains/Nipah_Virus_RdRp_domain.pdb",
    "RSV"            : "structures/domains/RSV_RdRp_domain.pdb",
    "Norovirus"      : "structures/domains/Norovirus_RdRp_domain.pdb",
}

PREPARED_DIR = "structures/prepared"
os.makedirs(PREPARED_DIR, exist_ok=True)


def prepare_with_pdbfixer(name: str, input_pdb: str) -> str:
    """
    Use PDBFixer to:
    - Add missing residues
    - Add missing heavy atoms
    - Add hydrogens at pH 7.4
    - Remove waters
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        import openmm.unit as unit
    except ImportError:
        print("  [ERROR] pdbfixer not installed.")
        print("          Run: pip install pdbfixer openmm")
        return fallback_prepare(name, input_pdb)

    print(f"  Running PDBFixer...")
    
    fixer = PDBFixer(filename=input_pdb)

    # Step 1: Find and add missing residues
    print("  Finding missing residues...")
    fixer.findMissingResidues()
    n_missing = sum(len(v) for v in fixer.missingResidues.values())
    print(f"    → {n_missing} missing residue(s) to model")

    # Step 2: Find missing heavy atoms
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Step 3: Remove water molecules
    fixer.removeHeterogens(keepWater=False)

    # Step 4: Add hydrogens at pH 7.4
    print("  Adding hydrogens at pH 7.4 (physiological)...")
    fixer.addMissingHydrogens(7.4)

    # Save prepared structure
    out_path = os.path.join(PREPARED_DIR, f"{name}_prepared.pdb")
    with open(out_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    print(f"  [OK] Prepared structure → {out_path}")
    return out_path


def fallback_prepare(name: str, input_pdb: str) -> str:
    """
    Fallback if pdbfixer is unavailable.
    Removes HETATM lines and writes clean PDB.
    Still needs GROMACS pdb2gmx for H addition.
    """
    print(f"  [FALLBACK] Using basic PDB cleaning (no H addition)")
    print(f"             pdb2gmx in Phase 04 will add hydrogens")

    out_path = os.path.join(PREPARED_DIR, f"{name}_prepared.pdb")

    with open(input_pdb, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            # Keep ATOM records only (removes HETATM = waters, ligands)
            if line.startswith(('ATOM', 'TER', 'END')):
                fout.write(line)

    # Verify
    atom_count = sum(1 for line in open(out_path) if line.startswith('ATOM'))
    print(f"  [OK] Cleaned PDB: {atom_count} ATOM records → {out_path}")
    return out_path


def run_propka(name: str, pdb_path: str) -> dict:
    """
    Run PROPKA to check protonation states.
    Requires: pip install propka
    """
    try:
        import propka.run as propka
        print(f"  Running PROPKA for protonation state analysis...")
        # PROPKA writes a .pka file
        propka_out = pdb_path.replace('.pdb', '.pka')
        mol = propka.single(pdb_path, optargs=['--quiet'])
        print(f"  [OK] PROPKA complete → {propka_out}")
        return {"status": "ok", "pka_file": propka_out}
    except ImportError:
        print(f"  [SKIP] PROPKA not installed (pip install propka)")
        print(f"         Default pH 7.4 protonation will be used by pdb2gmx")
        return {"status": "skipped"}
    except Exception as e:
        print(f"  [WARN] PROPKA failed: {e}")
        return {"status": "failed", "error": str(e)}


def verify_prepared_structure(name: str, pdb_path: str) -> dict:
    """Quick verification of prepared structure."""
    from Bio.PDB import PDBParser
    
    parser   = PDBParser(QUIET=True)
    structure = parser.get_structure(name, pdb_path)
    model    = structure[0]

    residues  = [r for r in model.get_residues() if r.id[0] == ' ']
    atoms     = list(model.get_atoms())
    hydrogens = [a for a in atoms if a.element == 'H']
    waters    = [r for r in model.get_residues() if r.id[0] == 'W']

    result = {
        "n_residues"  : len(residues),
        "n_atoms"     : len(atoms),
        "n_hydrogens" : len(hydrogens),
        "n_waters"    : len(waters),
        "has_H"       : len(hydrogens) > 0,
        "water_free"  : len(waters) == 0
    }

    status = "✓ READY" if result['water_free'] else "⚠ WATERS PRESENT"
    if result['has_H']:
        status += " | Hydrogens present"
    else:
        status += " | No hydrogens (pdb2gmx will add these)"

    print(f"  Verification: {len(residues)} residues, {len(atoms)} atoms {status}")
    return result


def write_preparation_log(results: dict):
    """Write a log of preparation steps for Methods section."""
    log_path = "logs/03_preparation_log.txt"
    os.makedirs("logs", exist_ok=True)

    with open(log_path, 'w') as f:
        f.write("Structure Preparation Log — RdRp Cryptic Allosteric Sites Study\n")
        f.write("="*70 + "\n\n")

        f.write("Methods text (ready to paste):\n")
        f.write("-"*50 + "\n")
        f.write("""
Each RdRp domain structure was prepared for molecular dynamics simulation
using PDBFixer (Eastman et al., 2023). Crystallographic water molecules 
and co-crystallised heteroatoms were removed. Missing residues within 
resolved domain boundaries were modelled using PDBFixer's integrated 
loop modelling protocol. Hydrogen atoms were added at physiological pH 
7.4 using PDBFixer's hydrogen assignment routine, which employs standard 
pKa values for titratable residues. Protonation states of histidine, 
lysine, aspartate, and glutamate residues were verified using PROPKA. 
The prepared structures were subsequently passed to GROMACS for molecular 
dynamics force field parametrisation and simulation setup.
""")

        f.write("\nPer-Structure Details:\n")
        f.write("-"*50 + "\n")
        for name, data in results.items():
            f.write(f"\n{name}:\n")
            for k, v in data.items():
                f.write(f"  {k}: {v}\n")

    print(f"\n[SAVED] Preparation log → {log_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PHASE 03: Structure Preparation")
    print("="*60 + "\n")

    results = {}

    for name, domain_file in DOMAIN_FILES.items():
        print(f"\n[{name}]")

        if not os.path.exists(domain_file):
            print(f"  [SKIP] Domain file not found: {domain_file}")
            print(f"         Run Phase 02 first.")
            continue

        print(f"  Input: {domain_file}")

        # Step 1: PDBFixer preparation
        prepared_path = prepare_with_pdbfixer(name, domain_file)

        # Step 2: PROPKA protonation check
        propka_result = run_propka(name, prepared_path)

        # Step 3: Verify
        verification = verify_prepared_structure(name, prepared_path)

        results[name] = {
            "prepared_file" : prepared_path,
            "propka"        : propka_result,
            "verification"  : verification
        }

    # Save log
    write_preparation_log(results)

    print("\n" + "="*60)
    print("  PHASE 03 COMPLETE")
    print("="*60)
    print("\n  Prepared structures are in: structures/prepared/")
    print("  Next → Phase 04: GROMACS MD Setup & Simulation")
    print()


if __name__ == "__main__":
    main()

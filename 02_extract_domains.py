"""
=============================================================
PHASE 02: RdRp Core Domain Extraction
=============================================================
Extracts only the RdRp core domain (palm + finger + thumb)
from each structure. Large L proteins (Nipah, RSV) are 250kDa
- simulating the full complex would take weeks. Extracting
the RdRp domain is scientifically standard and explicitly
justified in your Methods section.

Outputs: 5 domain PDB files ready for structure preparation.
=============================================================
Run: python 02_extract_domains.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser, PDBIO, Select, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

# ─────────────────────────────────────────────
# DOMAIN BOUNDARIES
# These define which residues constitute the RdRp
# core domain for each structure.
# Literature references for each boundary are given.
# ─────────────────────────────────────────────
DOMAIN_DEFINITIONS = {
    "H5N1_Influenza": {
        "pdb_id"       : "6QPF",
        "file"         : "structures/raw/6QPF.pdb",
        "chain"        : "B",      # PB1 subunit — contains RdRp catalytic core
        "res_start"    : 1,
        "res_end"      : 757,      # Full PB1 — IS the RdRp subunit
        "note"         : "PB1 IS the polymerase subunit. Palm (residues ~250-530), "
                         "Finger (~1-250), Thumb (~530-757). Keep full PB1.",
        "keep_chains"  : ["B"],    # Keep only PB1 for isolated domain MD
        # For MD we use only PB1. For structural comparison we use the trimer.
        "full_complex" : True,     # Also save full trimer for Foldseek
    },
    "Dengue_DENV3": {
        "pdb_id"       : "4C11",
        "file"         : "structures/raw/4C11.pdb",
        "chain"        : "A",
        "res_start"    : 265,      # NS5 RdRp domain starts at ~265 (with linker)
        "res_end"      : 900,      # NS5 ends at 900
        "note"         : "NS5 RdRp domain. MTase (1-264) excluded. "
                         "RdRp core: fingers(265-415), palm(416-600), thumb(601-900).",
        "keep_chains"  : ["A"],
        "full_complex" : False,
    },
    "Nipah_Virus": {
        "pdb_id"       : "9GJT",
        "file"         : "structures/raw/9GJT.pdb",
        "chain"        : "A",      # L protein
        "res_start"    : 1,        # Will be determined from structure
        "res_end"      : 1800,     # L protein RdRp domain is N-terminal (~1-570)
        "note"         : "NiV L protein. RdRp domain: residues 1-570 approx. "
                         "PRNTase follows. We extract RdRp core for MD. "
                         "See Sala et al. 2025 for domain boundaries.",
        "keep_chains"  : ["A"],
        "rdRp_end"     : 570,      # Approximate RdRp domain boundary in L protein
        "full_complex" : True,
    },
    "RSV": {
        "pdb_id"       : "6PZK",
        "file"         : "structures/raw/6PZK.pdb",
        "chain"        : "A",      # L protein
        "res_start"    : 1,
        "res_end"      : 1700,
        "note"         : "RSV L protein. RdRp core: residues 1-570 approx. "
                         "Known allosteric site at interface of RdRp-PRNTase. "
                         "See Gilman et al. 2019 (6PZK paper) for domain map.",
        "keep_chains"  : ["A"],
        "rdRp_end"     : 580,
        "full_complex" : True,
    },
    "Norovirus": {
        "pdb_id"       : "3BSO",
        "file"         : "structures/raw/3BSO.pdb",
        "chain"        : "A",
        "res_start"    : 1,
        "res_end"      : 500,      # Full NS7 RdRp
        "note"         : "Murine Norovirus NS7 RdRp — standalone polymerase, "
                         "no domain extraction needed. Keep full protein.",
        "keep_chains"  : ["A"],
        "full_complex" : False,
    }
}

DOMAIN_DIR  = "structures/domains"
COMPLEX_DIR = "structures/raw/complexes"
os.makedirs(DOMAIN_DIR, exist_ok=True)
os.makedirs(COMPLEX_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SELECTOR CLASSES
# ─────────────────────────────────────────────
class DomainSelect(Select):
    """Select specific chain and residue range."""
    def __init__(self, chain_id: str, res_start: int, res_end: int):
        self.chain_id  = chain_id
        self.res_start = res_start
        self.res_end   = res_end

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        if residue.id[0] != ' ':  # Skip HETATM (waters, ligands)
            return False
        res_num = residue.id[1]
        return self.res_start <= res_num <= self.res_end

    def accept_atom(self, atom):
        return True  # Keep all atoms


class ChainSelect(Select):
    """Select specific chains (no residue filtering)."""
    def __init__(self, chain_ids: list):
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.id[0] == ' '  # Standard residues only

    def accept_atom(self, atom):
        return True


# ─────────────────────────────────────────────
# EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────
def load_structure(name: str, filepath: str):
    """Load structure — handles both PDB and CIF."""
    ext = os.path.splitext(filepath)[1]
    if ext == '.cif':
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(name, filepath)


def get_actual_residue_range(structure, chain_id: str) -> tuple:
    """Get actual first and last residue numbers from a chain."""
    try:
        chain = structure[0][chain_id]
        residues = [r for r in chain.get_residues() if r.id[0] == ' ']
        if residues:
            return residues[0].id[1], residues[-1].id[1]
    except KeyError:
        pass
    return None, None


def extract_domain(name: str, defn: dict) -> str:
    """Extract RdRp domain and save to new PDB file."""
    filepath = defn['file']

    # Try .pdb first, then .cif
    if not os.path.exists(filepath):
        filepath = filepath.replace('.pdb', '.cif')
    if not os.path.exists(filepath):
        print(f"  [ERROR] Structure file not found: {defn['file']}")
        print(f"          Run Phase 01 first to download structures.")
        return None

    structure = load_structure(name, filepath)
    chain_id  = defn['chain']

    # Get actual range
    actual_start, actual_end = get_actual_residue_range(structure, chain_id)
    
    if actual_start is None:
        print(f"  [ERROR] Chain {chain_id} not found in {filepath}")
        # List available chains
        chains = [c.id for c in structure[0].get_chains()]
        print(f"          Available chains: {chains}")
        return None

    print(f"  Chain {chain_id}: residues {actual_start}–{actual_end} in structure")

    # Use defined boundaries, clipped to actual
    res_start = max(defn.get('res_start', 1), actual_start)
    res_end   = min(defn.get('rdRp_end', defn['res_end']), actual_end)
    
    print(f"  Extracting RdRp domain: residues {res_start}–{res_end}")

    # Output path
    out_path = os.path.join(DOMAIN_DIR, f"{name}_RdRp_domain.pdb")

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, DomainSelect(chain_id, res_start, res_end))

    # Count output residues
    parser_out = PDBParser(QUIET=True)
    struct_out = parser_out.get_structure(name, out_path)
    residues_out = list(struct_out[0][chain_id].get_residues())
    
    print(f"  [OK] Extracted {len(residues_out)} residues → {out_path}")
    return out_path


def save_full_complex(name: str, defn: dict) -> str:
    """Save full complex (for Foldseek structural comparison)."""
    if not defn.get('full_complex', False):
        return None

    filepath = defn['file']
    if not os.path.exists(filepath):
        filepath = filepath.replace('.pdb', '.cif')
    if not os.path.exists(filepath):
        return None

    structure = load_structure(name, filepath)
    out_path  = os.path.join(COMPLEX_DIR, f"{name}_full_complex.pdb")

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, ChainSelect(defn['keep_chains']))
    print(f"  [OK] Full complex saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PHASE 02: RdRp Domain Extraction")
    print("="*60 + "\n")

    print("Scientific justification:")
    print("  Large L proteins (Nipah, RSV) are ~250 kDa. Simulating the")
    print("  full L-P complex is computationally intractable at this stage.")
    print("  Isolating the RdRp core domain is standard practice in the")
    print("  field (see: Peng et al. 2020, Dulin et al. 2015).")
    print("  This approach is explicitly documented in the Methods section.\n")

    results = {}

    for name, defn in DOMAIN_DEFINITIONS.items():
        print(f"\n[{name}] — {defn['pdb_id']}")
        print(f"  {defn['note'][:80]}...")

        # Extract RdRp domain for MD
        domain_path = extract_domain(name, defn)
        
        # Save full complex for Foldseek
        complex_path = save_full_complex(name, defn)

        results[name] = {
            "domain_file"  : domain_path,
            "complex_file" : complex_path
        }

    # Summary
    print("\n" + "="*60)
    print("  EXTRACTION SUMMARY")
    print("="*60)
    for name, paths in results.items():
        status = "✓" if paths['domain_file'] else "✗"
        print(f"  {status} {name}")
        if paths['domain_file']:
            print(f"      Domain : {paths['domain_file']}")
        if paths['complex_file']:
            print(f"      Complex: {paths['complex_file']}")

    print("\n  Next step → Phase 03: Structure Preparation")
    print("  (Remove waters, add hydrogens, assign protonation states)\n")


if __name__ == "__main__":
    main()

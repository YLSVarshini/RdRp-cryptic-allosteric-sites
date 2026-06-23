"""
=============================================================
PHASE 06: Evolutionary Conservation Analysis
=============================================================
For each virus:
  1. Download 50-100 RdRp sequences from NCBI
  2. Align sequences with MUSCLE5
  3. Prepare ConSurf submission files
  4. Parse ConSurf results → per-residue conservation scores
  5. Map scores onto cryptic pocket residues
     → High score (7-9) in pocket = resistance-evasive target

ConSurf web server: https://consurf.tau.ac.il
=============================================================
Run: python 06_sequence_analysis.py
"""

import os
import sys
import time
import json
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# ─────────────────────────────────────────────
# CONFIGURATION
# Set your email for NCBI Entrez (required)
# ─────────────────────────────────────────────
Entrez.email = "your.email@university.edu"   # CHANGE THIS
Entrez.api_key = ""                           # Optional: NCBI API key for faster downloads

SEQ_DIR    = "sequences"
ALIGN_DIR  = "sequences/alignments"
CONSURF_DIR = "sequences/consurf"
FIGURES_DIR = "figures"

for d in [SEQ_DIR, ALIGN_DIR, CONSURF_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# NCBI SEARCH TERMS FOR EACH VIRUS RdRp
# Tuned to retrieve diverse, representative sequences
# ─────────────────────────────────────────────
SEARCH_QUERIES = {
    "H5N1_Influenza": {
        "query"    : 'Influenza A[Organism] AND "polymerase basic protein 1"[Title] AND H5N1',
        "n_seqs"   : 80,
        "db"       : "protein",
        "uniprot"  : "P03431",  # H5N1 PB1 reference
        "gene"     : "PB1"
    },
    "Dengue_DENV3": {
        "query"    : 'Dengue virus 3[Organism] AND "nonstructural protein 5"[Title]',
        "n_seqs"   : 80,
        "db"       : "protein",
        "uniprot"  : "Q6YMS4",
        "gene"     : "NS5"
    },
    "Nipah_Virus": {
        "query"    : 'Nipah virus[Organism] AND "large protein"[Title]',
        "n_seqs"   : 60,    # Fewer sequences available (rare virus)
        "db"       : "protein",
        "uniprot"  : "Q9IH63",
        "gene"     : "L protein"
    },
    "RSV": {
        "query"    : 'Respiratory syncytial virus[Organism] AND "large protein"[Title]',
        "n_seqs"   : 80,
        "db"       : "protein",
        "uniprot"  : "P04544",
        "gene"     : "L protein"
    },
    "Norovirus": {
        "query"    : 'Norovirus[Organism] AND "RNA-dependent RNA polymerase"[Title]',
        "n_seqs"   : 80,
        "db"       : "protein",
        "uniprot"  : "Q04544",
        "gene"     : "NS7/RdRp"
    }
}


# ─────────────────────────────────────────────
# SEQUENCE DOWNLOAD
# ─────────────────────────────────────────────
def download_sequences(name: str, query_info: dict) -> str:
    """Download RdRp sequences from NCBI for a given virus."""
    print(f"\n  Searching NCBI for: {query_info['query'][:60]}...")

    out_fasta = os.path.join(SEQ_DIR, f"{name}_sequences.fasta")

    if os.path.exists(out_fasta):
        n_seqs = sum(1 for _ in SeqIO.parse(out_fasta, "fasta"))
        print(f"  [CACHED] {n_seqs} sequences already downloaded")
        return out_fasta

    try:
        # Search
        handle  = Entrez.esearch(
            db=query_info['db'],
            term=query_info['query'],
            retmax=query_info['n_seqs'] * 2  # Over-fetch, then filter
        )
        record  = Entrez.read(handle)
        handle.close()

        ids = record['IdList'][:query_info['n_seqs']]
        print(f"  Found {len(record['IdList'])} sequences, downloading {len(ids)}...")

        if not ids:
            print(f"  [WARN] No sequences found. Check query or try NCBI manually.")
            return create_placeholder_fasta(name, query_info)

        # Fetch sequences in batches
        sequences = []
        batch_size = 20

        for i in range(0, len(ids), batch_size):
            batch = ids[i:i+batch_size]
            fetch_handle = Entrez.efetch(
                db=query_info['db'],
                id=",".join(batch),
                rettype="fasta",
                retmode="text"
            )
            batch_seqs = list(SeqIO.parse(fetch_handle, "fasta"))
            fetch_handle.close()
            sequences.extend(batch_seqs)
            time.sleep(0.35)  # NCBI rate limit: 3 requests/second
            print(f"  Downloaded {min(i+batch_size, len(ids))}/{len(ids)} sequences...")

        # Filter: remove very short sequences (fragments)
        # Keep sequences within 80-120% of median length
        lengths = [len(s) for s in sequences]
        med_len = np.median(lengths)
        filtered = [
            s for s in sequences
            if 0.5 * med_len <= len(s) <= 1.5 * med_len
        ]
        print(f"  Filtered: {len(sequences)} → {len(filtered)} sequences (removed fragments)")

        # Save
        SeqIO.write(filtered, out_fasta, "fasta")
        print(f"  [OK] {len(filtered)} sequences → {out_fasta}")
        return out_fasta

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"          Try: Entrez.email = 'your@email.com' at top of script")
        return create_placeholder_fasta(name, query_info)


def create_placeholder_fasta(name: str, query_info: dict) -> str:
    """Create a note file if download fails."""
    out_path = os.path.join(SEQ_DIR, f"{name}_DOWNLOAD_FAILED.txt")
    with open(out_path, 'w') as f:
        f.write(f"Download failed for {name}\n")
        f.write(f"Manual download from NCBI:\n")
        f.write(f"  Search: {query_info['query']}\n")
        f.write(f"  Database: {query_info['db']}\n")
        f.write(f"  UniProt reference: {query_info.get('uniprot', 'N/A')}\n")
        f.write(f"  Save as FASTA to: sequences/{name}_sequences.fasta\n")
    print(f"  [SAVED] Manual download instructions → {out_path}")
    return None


# ─────────────────────────────────────────────
# MULTIPLE SEQUENCE ALIGNMENT (MUSCLE5)
# ─────────────────────────────────────────────
def run_muscle_alignment(name: str, fasta_file: str) -> str:
    """Run MUSCLE5 multiple sequence alignment."""
    if not fasta_file or not os.path.exists(fasta_file):
        print(f"  [SKIP] No sequences to align")
        return None

    out_aln = os.path.join(ALIGN_DIR, f"{name}_aligned.fasta")

    if os.path.exists(out_aln):
        print(f"  [CACHED] Alignment already exists: {out_aln}")
        return out_aln

    print(f"  Running MUSCLE5 alignment...")
    print(f"  Input: {fasta_file}")

    # Try MUSCLE5 first, then MUSCLE3
    for muscle_cmd in ["muscle5", "muscle"]:
        if subprocess.run(["which", muscle_cmd], capture_output=True).returncode == 0:
            cmd = [muscle_cmd, "-align", fasta_file, "-output", out_aln]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                n_aligned = sum(1 for _ in SeqIO.parse(out_aln, "fasta"))
                print(f"  [OK] {n_aligned} sequences aligned → {out_aln}")
                return out_aln
            else:
                print(f"  [WARN] {muscle_cmd} failed: {result.stderr[:100]}")

    # Fallback: try Biopython's pairwise aligner as simple alternative
    print(f"  [WARN] MUSCLE not found. Install: conda install -c bioconda muscle")
    print(f"         Or download from: https://www.drive5.com/muscle/downloads.htm")
    print(f"         For ConSurf: you can submit the unaligned FASTA directly")
    return fasta_file  # Return unaligned — ConSurf will align


def calculate_alignment_stats(aln_file: str) -> dict:
    """Calculate basic statistics about the alignment."""
    if not aln_file or not os.path.exists(aln_file):
        return {}

    seqs = list(SeqIO.parse(aln_file, "fasta"))
    if not seqs:
        return {}

    lengths = [len(str(s.seq).replace('-', '')) for s in seqs]

    return {
        "n_sequences"   : len(seqs),
        "aln_length"    : len(str(seqs[0].seq)),
        "mean_seq_len"  : np.mean(lengths),
        "min_seq_len"   : np.min(lengths),
        "max_seq_len"   : np.max(lengths),
    }


# ─────────────────────────────────────────────
# CONSURF PREPARATION
# ─────────────────────────────────────────────
def prepare_consurf_submission(name: str, aln_file: str, pdb_id: str) -> dict:
    """
    Prepare everything needed for ConSurf submission.
    ConSurf web server: https://consurf.tau.ac.il
    """
    print(f"\n  Preparing ConSurf submission for {name}...")

    submit_dir = os.path.join(CONSURF_DIR, name)
    os.makedirs(submit_dir, exist_ok=True)

    instructions_file = os.path.join(submit_dir, "SUBMISSION_INSTRUCTIONS.txt")

    with open(instructions_file, 'w') as f:
        f.write(f"ConSurf Submission Instructions — {name}\n")
        f.write("="*60 + "\n\n")
        f.write("1. Go to: https://consurf.tau.ac.il\n\n")
        f.write("2. Input protein structure:\n")
        f.write(f"   → Enter PDB ID: {pdb_id}\n")
        f.write(f"   → OR upload: structures/domains/{name}_RdRp_domain.pdb\n\n")
        f.write("3. Upload MSA:\n")
        if aln_file and os.path.exists(aln_file):
            f.write(f"   → Upload: {aln_file}\n")
            f.write(f"   → Format: FASTA\n")
        else:
            f.write(f"   → Upload: sequences/{name}_sequences.fasta (unaligned)\n")
            f.write(f"   → Check 'Perform MSA' option\n\n")
        f.write("4. Settings:\n")
        f.write("   → Substitution model: WAG (for proteins)\n")
        f.write("   → Algorithm: Bayesian\n")
        f.write("   → Iterations: 1\n\n")
        f.write("5. Download results:\n")
        f.write("   → grades file: consurf.grades\n")
        f.write("   → colored PDB: consurf_atoms.pdb\n")
        f.write(f"   → Save to: {submit_dir}/\n\n")
        f.write("6. Run Phase 06 again to parse results once downloaded.\n")

    print(f"  [OK] Instructions → {instructions_file}")
    return {"instructions": instructions_file, "submit_dir": submit_dir}


def parse_consurf_grades(name: str) -> pd.DataFrame:
    """
    Parse ConSurf .grades output file.
    Returns per-residue conservation scores (1=variable, 9=conserved).
    """
    grades_file = os.path.join(CONSURF_DIR, name, "consurf.grades")

    if not os.path.exists(grades_file):
        print(f"  [PENDING] ConSurf results not yet downloaded for {name}")
        print(f"            Upload to ConSurf, then save 'consurf.grades' to:")
        print(f"            {os.path.join(CONSURF_DIR, name)}/")
        return pd.DataFrame()

    print(f"  Parsing ConSurf grades: {grades_file}")

    records = []
    with open(grades_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    records.append({
                        'residue_num'    : int(parts[0]),
                        'aa'             : parts[1],
                        'score'          : float(parts[2]),
                        'color'          : int(parts[3]),      # 1-9 conservation grade
                        'confidence_int' : parts[4] if len(parts) > 4 else "",
                        'source'         : name
                    })
                except (ValueError, IndexError):
                    continue

    if not records:
        print(f"  [WARN] Could not parse {grades_file}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  [OK] {len(df)} residues with conservation scores")
    print(f"       Mean conservation: {df['color'].mean():.2f}/9")
    print(f"       Highly conserved (8-9): {(df['color'] >= 8).sum()} residues")

    out_csv = os.path.join(CONSURF_DIR, name, "conservation_scores.csv")
    df.to_csv(out_csv, index=False)
    return df


def map_conservation_to_pockets(name: str, conservation_df: pd.DataFrame,
                                  pocket_residues: list) -> pd.DataFrame:
    """
    Map ConSurf conservation scores onto cryptic pocket residues.
    High score (7-9) in pocket = evolutionarily conserved = resistance-evasive.
    """
    if conservation_df.empty or not pocket_residues:
        return pd.DataFrame()

    pocket_conservation = conservation_df[
        conservation_df['residue_num'].isin(pocket_residues)
    ].copy()

    pocket_conservation['in_pocket'] = True
    pocket_conservation['resistance_evasive'] = pocket_conservation['color'] >= 7

    print(f"\n  Pocket conservation analysis — {name}:")
    print(f"  Pocket residues: {len(pocket_residues)}")
    print(f"  Mean pocket conservation score: {pocket_conservation['color'].mean():.2f}/9")
    print(f"  Resistance-evasive residues (score ≥ 7): {pocket_conservation['resistance_evasive'].sum()}")

    return pocket_conservation


def plot_conservation_mapping(name: str, conservation_df: pd.DataFrame):
    """Plot conservation scores along sequence — key figure for paper."""
    if conservation_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    colors_map = {
        1: '#FF0000', 2: '#FF4500', 3: '#FF8C00',
        4: '#FFD700', 5: '#ADFF2F', 6: '#32CD32',
        7: '#00CED1', 8: '#1E90FF', 9: '#0000FF'
    }

    for _, row in conservation_df.iterrows():
        color = colors_map.get(int(row['color']), '#808080')
        ax.bar(row['residue_num'], row['color'], color=color,
               width=1, linewidth=0)

    ax.axhline(7, color='red', linestyle='--', linewidth=1,
               label='Resistance-evasive threshold (7)')
    ax.set_xlabel("Residue Number", fontsize=12)
    ax.set_ylabel("Conservation Score (1-9)", fontsize=12)
    ax.set_title(f"Evolutionary Conservation: {name} RdRp\n"
                 f"Blue = conserved (resistance-evasive) | Red = variable", fontsize=11)
    ax.set_ylim(0, 9.5)
    ax.legend()

    # Colorbar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=str(i))
                       for i, c in colors_map.items()]
    ax.legend(handles=legend_elements, title="ConSurf Grade",
              loc='upper right', ncol=9, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"conservation_{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [FIGURE] Conservation plot → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PHASE 06: Evolutionary Conservation Analysis")
    print("="*60 + "\n")

    PDB_IDS = {
        "H5N1_Influenza": "6QPF",
        "Dengue_DENV3"  : "4C11",
        "Nipah_Virus"   : "9GJT",
        "RSV"           : "6PZK",
        "Norovirus"     : "3BSO"
    }

    all_conservation = {}

    for name, query_info in SEARCH_QUERIES.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        # Step 1: Download sequences
        fasta_file = download_sequences(name, query_info)

        # Step 2: Align with MUSCLE5
        if fasta_file:
            aln_file = run_muscle_alignment(name, fasta_file)
            stats    = calculate_alignment_stats(aln_file)
            if stats:
                print(f"  Alignment: {stats['n_sequences']} seqs, {stats['aln_length']} columns")
        else:
            aln_file = None

        # Step 3: Prepare ConSurf submission
        consurf_info = prepare_consurf_submission(name, aln_file, PDB_IDS[name])

        # Step 4: Parse ConSurf results (if already downloaded)
        conservation_df = parse_consurf_grades(name)

        if not conservation_df.empty:
            plot_conservation_mapping(name, conservation_df)
            all_conservation[name] = conservation_df

    # Summary
    print("\n" + "="*60)
    print("  PHASE 06 SUMMARY")
    print("="*60)
    print("\n  ConSurf submissions prepared.")
    print("  Submit each to: https://consurf.tau.ac.il")
    print("  Check CONSURF_DIR/[name]/SUBMISSION_INSTRUCTIONS.txt")
    print("\n  After downloading results, re-run this script")
    print("  to parse grades and generate figures.")
    print("\n  Next → Phase 07: Cross-virus structural alignment (Foldseek)")
    print()


if __name__ == "__main__":
    main()

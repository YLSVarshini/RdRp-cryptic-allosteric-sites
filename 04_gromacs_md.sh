#!/bin/bash
# ============================================================
# PHASE 04: GROMACS Molecular Dynamics Pipeline
# Runs for ALL 5 RdRp structures
# ============================================================
# USAGE:
#   ./04_gromacs_md.sh                    # Run all 5 in sequence
#   ./04_gromacs_md.sh H5N1_Influenza     # Run just one
#   Submit each as a separate HPC job for parallel execution
#
# REQUIREMENTS:
#   - GROMACS 2021+ installed (gmx command available)
#   - ~50 GB disk space per simulation
#   - GPU strongly recommended (add -gpu_id 0 to mdrun calls)
#   - HPC cluster recommended for production runs
# ============================================================

set -e  # Exit on error

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FORCEFIELD="amber99sb-ildn"   # Validated for viral polymerases
WATER_MODEL="tip3p"            # Compatible with amber99sb-ildn
BOX_TYPE="dodecahedron"        # Most efficient box shape for globular proteins
BOX_DIST="1.0"                 # nm — minimum distance from protein to box edge
ION_CONC="0.15"               # mol/L — physiological NaCl concentration
N_THREADS=8                    # CPU threads (adjust to your machine)
GPU_ACCEL=""           # GPU acceleration flag (remove if no GPU)

# Directories
STRUCT_DIR="structures/prepared"
MD_OUT_DIR="md"
MDP_DIR="md"

# ─────────────────────────────────────────────
# STRUCTURES TO SIMULATE
# ─────────────────────────────────────────────
declare -A STRUCTURES
STRUCTURES["H5N1_Influenza"]="H5N1_Influenza_prepared.pdb"
STRUCTURES["Dengue_DENV3"]="Dengue_DENV3_prepared.pdb"
STRUCTURES["Nipah_Virus"]="Nipah_Virus_prepared.pdb"
STRUCTURES["RSV"]="RSV_prepared.pdb"
STRUCTURES["Norovirus"]="Norovirus_prepared.pdb"

# ─────────────────────────────────────────────
# CHECK GROMACS IS AVAILABLE
# ─────────────────────────────────────────────
if ! command -v gmx &> /dev/null; then
    echo "ERROR: GROMACS (gmx) not found."
    echo "Install: conda install -c bioconda gromacs"
    echo "      or: sudo apt install gromacs"
    exit 1
fi

GROMACS_VERSION=$(gmx --version 2>&1 | grep "GROMACS version" | awk '{print $3}')
echo "=========================================="
echo " GROMACS MD Pipeline — RdRp Study"
echo " GROMACS version: ${GROMACS_VERSION}"
echo " Force field: ${FORCEFIELD}"
echo " Water model: ${WATER_MODEL}"
echo "=========================================="

# ─────────────────────────────────────────────
# FUNCTION: run_md_for_structure
# ─────────────────────────────────────────────
run_md_for_structure() {
    local NAME=$1
    local PDB_FILE="${STRUCT_DIR}/${STRUCTURES[$NAME]}"
    local WORK_DIR="${MD_OUT_DIR}/${NAME}"

    echo ""
    echo "=========================================="
    echo " Processing: ${NAME}"
    echo " Input: ${PDB_FILE}"
    echo "=========================================="

    # Check input file exists
    if [ ! -f "${PDB_FILE}" ]; then
        echo "ERROR: Input file not found: ${PDB_FILE}"
        echo "Run Phase 03 first."
        return 1
    fi

    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    # ─────────────────────────────────────────
    # STEP 1: Generate topology
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 1] Generating topology with pdb2gmx..."
    echo "         Force field: ${FORCEFIELD}"
    echo "         Water: ${WATER_MODEL}"

    # Interactive: select force field 6 (AMBER99SB-ILDN) and water model 1 (TIP3P)
    # The echo pipe auto-selects these
    echo "6
1" | gmx pdb2gmx \
        -f "../../${PDB_FILE}" \
        -o protein_processed.gro \
        -p topol.top \
        -i posre.itp \
        -ff "${FORCEFIELD}" \
        -water "${WATER_MODEL}" \
        -ignh \
        2>&1 | tee pdb2gmx.log

    echo "[STEP 1] Done. Topology: topol.top"

    # ─────────────────────────────────────────
    # STEP 2: Define simulation box
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 2] Defining simulation box..."
    echo "         Type: ${BOX_TYPE}, distance: ${BOX_DIST} nm"

    gmx editconf \
        -f protein_processed.gro \
        -o protein_newbox.gro \
        -c \
        -d "${BOX_DIST}" \
        -bt "${BOX_TYPE}" \
        2>&1 | tee editconf.log

    echo "[STEP 2] Done."

    # ─────────────────────────────────────────
    # STEP 3: Solvate
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 3] Solvating with TIP3P water..."

    gmx solvate \
        -cp protein_newbox.gro \
        -cs spc216.gro \
        -o protein_solv.gro \
        -p topol.top \
        2>&1 | tee solvate.log

    echo "[STEP 3] Done."

    # ─────────────────────────────────────────
    # STEP 4: Add ions (neutralise + 0.15 M NaCl)
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 4] Adding ions (0.15 M NaCl, physiological)..."

    # Generate tpr for ion addition
    gmx grompp \
        -f "../../${MDP_DIR}/em.mdp" \
        -c protein_solv.gro \
        -p topol.top \
        -o ions.tpr \
        -maxwarn 5 \
        2>&1 | tee grompp_ions.log

    # Add ions — echo "13" selects SOL (water) group to replace
    echo "13" | gmx genion \
        -s ions.tpr \
        -o protein_ions.gro \
        -p topol.top \
        -pname NA \
        -nname CL \
        -neutral \
        -conc "${ION_CONC}" \
        2>&1 | tee genion.log

    echo "[STEP 4] Done."

    # ─────────────────────────────────────────
    # STEP 5: Energy Minimisation
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 5] Energy minimisation..."
    echo "         Target: Fmax < 1000 kJ/mol/nm"

    gmx grompp \
        -f "../../${MDP_DIR}/em.mdp" \
        -c protein_ions.gro \
        -p topol.top \
        -o em.tpr \
        -maxwarn 5 \
        2>&1 | tee grompp_em.log

    gmx mdrun \
        -v \
        -deffnm em \
        -ntmpi 1 \
        -ntomp "${N_THREADS}" \
        ${GPU_ACCEL} \
        2>&1 | tee mdrun_em.log

    # Check convergence
    FMAX=$(grep "Maximum force" em.log 2>/dev/null | tail -1 | awk '{print $4}')
    echo "[STEP 5] Done. Final Fmax: ${FMAX} kJ/mol/nm"

    # ─────────────────────────────────────────
    # STEP 6: NVT Equilibration (1 ns)
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 6] NVT equilibration (1 ns, 300 K)..."

    gmx grompp \
        -f "../../${MDP_DIR}/nvt.mdp" \
        -c em.gro \
        -r em.gro \
        -p topol.top \
        -o nvt.tpr \
        -maxwarn 5 \
        2>&1 | tee grompp_nvt.log

    gmx mdrun \
        -v \
        -deffnm nvt \
        -ntmpi 1 \
        -ntomp "${N_THREADS}" \
        ${GPU_ACCEL} \
        2>&1 | tee mdrun_nvt.log

    echo "[STEP 6] Done."

    # ─────────────────────────────────────────
    # STEP 7: NPT Equilibration (1 ns)
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 7] NPT equilibration (1 ns, 300 K, 1 bar)..."

    gmx grompp \
        -f "../../${MDP_DIR}/npt.mdp" \
        -c nvt.gro \
        -r nvt.gro \
        -t nvt.cpt \
        -p topol.top \
        -o npt.tpr \
        -maxwarn 5 \
        2>&1 | tee grompp_npt.log

    gmx mdrun \
        -v \
        -deffnm npt \
        -ntmpi 1 \
        -ntomp "${N_THREADS}" \
        ${GPU_ACCEL} \
        2>&1 | tee mdrun_npt.log

    echo "[STEP 7] Done."

    # ─────────────────────────────────────────
    # STEP 8: Production MD (200 ns)
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 8] Production MD (200 ns)..."
    echo "         This is the long step — submit to HPC for GPU acceleration"
    echo "         Estimated time: 12-48 hrs on GPU node per structure"

    gmx grompp \
        -f "../../${MDP_DIR}/md_prod.mdp" \
        -c npt.gro \
        -t npt.cpt \
        -p topol.top \
        -o md_prod.tpr \
        -maxwarn 5 \
        2>&1 | tee grompp_prod.log

    gmx mdrun \
        -v \
        -deffnm md_prod \
        -ntmpi 1 \
        -ntomp "${N_THREADS}" \
        ${GPU_ACCEL} \
         \
        2>&1 | tee mdrun_prod.log

    echo "[STEP 8] Production MD complete."

    # ─────────────────────────────────────────
    # STEP 9: Post-processing
    # ─────────────────────────────────────────
    echo ""
    echo "[STEP 9] Post-processing trajectory..."

    # Fix PBC (periodic boundary conditions) — centre protein in box
    echo "1 0" | gmx trjconv \
        -s md_prod.tpr \
        -f md_prod.xtc \
        -o md_prod_noPBC.xtc \
        -pbc mol \
        -center \
        2>&1 | tee trjconv.log

    # Calculate RMSD (backbone stability)
    echo "4 4" | gmx rms \
        -s md_prod.tpr \
        -f md_prod_noPBC.xtc \
        -o rmsd_backbone.xvg \
        -tu ns \
        2>&1 | tee rmsd.log

    # Calculate RMSF (per-residue flexibility — key for cryptic pockets)
    echo "4" | gmx rmsf \
        -s md_prod.tpr \
        -f md_prod_noPBC.xtc \
        -o rmsf_residues.xvg \
        -res \
        2>&1 | tee rmsf.log

    echo "[STEP 9] Done. RMSD → rmsd_backbone.xvg | RMSF → rmsf_residues.xvg"

    cd ../..

    echo ""
    echo "=========================================="
    echo " COMPLETE: ${NAME}"
    echo " Output directory: ${WORK_DIR}/"
    echo " Trajectory: ${WORK_DIR}/md_prod_noPBC.xtc"
    echo "=========================================="
}

# ─────────────────────────────────────────────
# MAIN: Run structures
# ─────────────────────────────────────────────
if [ $# -eq 1 ]; then
    # Run just the specified structure
    if [[ -v "STRUCTURES[$1]" ]]; then
        run_md_for_structure "$1"
    else
        echo "ERROR: Unknown structure '$1'"
        echo "Available: ${!STRUCTURES[@]}"
        exit 1
    fi
else
    # Run all structures in sequence
    # For HPC: submit each as a separate job
    echo ""
    echo "Running all 5 structures sequentially."
    echo "For HPC: submit each separately for parallel execution."
    echo ""
    for NAME in "${!STRUCTURES[@]}"; do
        run_md_for_structure "${NAME}"
    done
fi

echo ""
echo "=========================================="
echo " ALL MD SIMULATIONS COMPLETE"
echo " Next → Phase 05: fpocket Cryptic Pocket Detection"
echo "=========================================="

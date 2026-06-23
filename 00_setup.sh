#!/bin/bash
# ============================================================
# RdRp Cryptic Allosteric Sites Pipeline - Environment Setup
# Run this ONCE before anything else
# ============================================================

echo "=========================================="
echo " Setting up RdRp Pipeline Environment"
echo "=========================================="

# Core scientific Python
pip install biopython mdanalysis numpy pandas matplotlib seaborn scipy scikit-learn joblib tqdm requests

# Cheminformatics
pip install rdkit-pypi

# Structure handling
pip install pdbfixer

# Plotting
pip install plotly kaleido

# If on Google Colab, also run:
# !apt-get install -y gromacs fpocket

echo ""
echo "=========================================="
echo " External tools you must install manually:"
echo "=========================================="
echo " 1. GROMACS      -> https://www.gromacs.org/Downloads"
echo "    Ubuntu:  sudo apt install gromacs"
echo "    Conda:   conda install -c bioconda gromacs"
echo ""
echo " 2. fpocket      -> https://github.com/Discngine/fpocket"
echo "    Ubuntu:  sudo apt install fpocket"
echo "    Or:      git clone + make install"
echo ""
echo " 3. AutoDock Vina -> https://vina.scripps.edu/downloads/"
echo "    Ubuntu:  sudo apt install autodock-vina"
echo ""
echo " 4. PyMOL        -> conda install -c conda-forge pymol-open-source"
echo ""
echo " 5. Foldseek     -> https://github.com/steineggerlab/foldseek"
echo "    Or use web: https://search.foldseek.com"
echo ""
echo "Setup complete. Check above for any pip errors."

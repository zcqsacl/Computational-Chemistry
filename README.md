# Notes on the HF_SCF_Minimal_Solver
This script derives an algorithm for solving the Roothaan-Hall equations for a general molecule (uses water as an example) 
in basic Python code. The final converged answer for the Hartree-Fock energy of water is -74.34 Hartrees, which is in close 
agreement with literature values of -74.96, likely limited by the choice of basis set. N.B. I used an external computer 
(RunPod H100 PCLe) to compute the electron repulsion 4-index integral, since it would otherwise take ~5hrs on Google Colab 
where I originally wrote the script. Much of the theory is taken from my UCL CHEM0028 notes.

# j_and_k.py
This was solely used with RunPod to calculate the 4-index electron resonance integral matrix as described above.

# DFT_SCF_Minimal_Solver
Sharing some blocks of code with the HF solver, this is a work in progress.

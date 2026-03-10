# neural.ipynb
This model takes smile strings of small organic molecules and tokenises them using the bert-base-case tokeniser, which vectorises each symbol into the smile string into a 768-dimensional vector.
These vectors are then passed through 40 x [Attention + MLP] layers, with a final MLP head, to predict the 1st excited state energy of the molecule.
It was trained on 20,000 molecule smile strings and their energies via the Kaggle data set ‘qm8’, and sent to a HPC (runpod.io RTX5090 GPU).
It is validated on 1787 molecules as follows, with a MAE of +/-0.384 eV:
<img width="884" height="884" alt="image" src="https://github.com/user-attachments/assets/1adb0ddd-36b1-4d2f-884f-79d0d3fc5250" />


# Notes on the HF_SCF_Minimal_Solver
This script derives an algorithm for solving the Roothaan-Hall equations for a general molecule (uses water as an example) 
in basic Python code. The final converged answer for the Hartree-Fock energy of water is -74.34 Hartrees, which is in close 
agreement with literature values of -74.96, likely limited by the choice of basis set. N.B. I used an external computer 
(RunPod H100 PCLe) to compute the electron repulsion 4-index integral, since it would otherwise take ~5hrs on Google Colab 
where I originally wrote the script. Much of the theory is taken from my UCL CHEM0028 notes.

# j_and_k.py
This was solely used with RunPod to calculate the 4-index electron resonance integral matrix as described above.

# prediction TVJK.ipynb
Basic neural network which learns inter-orbital potentials between common organic atoms, by separately learning T, V, J and K by inter-atomic distance.
Could be used to provide rapid first guesses to molecular energy, though doesn't account for four-fold-orbital J and K interactions.

# DFT_SCF_Minimal_Solver
[Unfinished]

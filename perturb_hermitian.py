from canonical import *

import numpy as np
from scipy.linalg import schur
from bqskit.utils.math import pauli_expansion
from scipy.stats import unitary_group

def perturbed_unitary_log_no_i(unitary : np.array) -> np.array:
    U = unitary if unitary[0,0] >= 0 else -unitary
    D, Q = schur(U) # Equivalent to eigendecomposition
    D = np.diag(D) 
    D = D / np.abs(D)
    Theta = np.diag( np.imag(np.log(D)) % (2*np.pi) )

    # Perturbation
    perturbation = np.diag(2*np.pi * np.random.randint(0, 4, (np.shape(D))))

    Theta += perturbation

    H = Q @ Theta @ Q.conj().T
    return 0.5 * H + 0.5 * H.conj().T

if __name__ == '__main__':

    num_qubits = 3
    num_unitaries = 10

    # Create unitaries
    unitaries = [unitary_group.rvs(2**num_qubits) for _ in range(num_unitaries)]
    # Find canonical unitaries
    canonical_unitaries = [to_canonical_unitary(u) for u in unitaries]

    # Do eigendecomposition
    # Get angles for Theta = QH @ H @ Q
    # Randomly add 2pi diagonal
    perturbed_Hs = [perturbed_unitary_log_no_i(u) for u in unitaries]

    # Convert to perturbed Hermitians to unitaries 
    perturbed_paulis = [pauli_expansion(H) for H in perturbed_Hs]
    # Get canonical form of perturbed unitaries
    canonical_perturbed_unitaries = [pauli_to_unitary(p) for p in perturbed_paulis]

    for base, perturbed in zip(canonical_unitaries, canonical_perturbed_unitaries):
        assert_distance_unitaries([base, perturbed])
        assert_closeness_unitaries([base, perturbed])

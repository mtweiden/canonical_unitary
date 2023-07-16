import numpy as np
from scipy.linalg import expm, schur
from bqskit.utils.math import unitary_log_no_i, pauli_expansion
from bqskit.utils.math import dot_product, PauliMatrices
from bqskit.qis.unitary import UnitaryMatrix
from scipy.stats import unitary_group


def random_phase():
    phase = 2*np.pi * np.random.rand()
    return np.exp(1j * phase)

def better_print(array : np.array) -> None:
    print(np.round(array, 3))

def to_canonical_unitary(unitary : np.array) -> np.array:
    """
    Convert a unitary to a special unitary, eliminating the global phase factor.
    Ensure that unitaries that differ in global phase by a root of unitary are
    mapped to the same matrix.

    Arguments:
        unitary (np.array): A unitary matrix.
    
    Returns:
        (np.array): A canonical unitary matrix in the speical unitary group.
    """
    determinant = np.linalg.det(unitary)
    dimension = len(unitary)
    global_phase = np.angle(determinant) / dimension
    global_phase = global_phase % (2 * np.pi / dimension)
    global_phase_factor = np.exp(-1j * global_phase)
    special_unitary = global_phase_factor * unitary
    # Standardize speical unitary to account for exp(-i2pi/N) differences
    first_row_mags = np.linalg.norm(special_unitary[0,:], ord=2)
    index = np.argmax(first_row_mags)
    std_phase = np.angle(special_unitary[0,index])
    correction_phase = 0 - std_phase
    std_correction = np.exp(1j * correction_phase)
    return std_correction * special_unitary

def to_SU(unitary : np.array) -> np.array:
    """
    Convert a unitary to a special unitary, ensuring det(unitary) = 1

    Arguments:
        unitary (np.array): A unitary matrix.
    
    Returns:
        (np.array): A unitary matrix in the speical unitary group.
    """
    determinant = np.linalg.det(unitary)
    dimension = len(unitary)
    global_phase = np.angle(determinant) / dimension
    global_phase = global_phase % (2 * np.pi / dimension)
    global_phase_factor = np.exp(-1j * global_phase)
    special_unitary = global_phase_factor * unitary
    return special_unitary

def test_U_to_SU() -> None:
    for size in [2,4,8,16]:
        for _ in range(100):
            random_U = unitary_group.rvs(size)
            SU = to_SU(random_U)
            check_unitary = UnitaryMatrix(SU)
            assert np.allclose(np.linalg.det(SU), 1)
            assert check_unitary.get_distance_from(random_U) <= 1e-7

def std_unitary_log_no_i(unitary : np.array) -> np.array:
    # NOTE: Function is redundant if unitary is already canonical
    U = unitary if unitary[0,0] >= 0 else -unitary
    D, Q = schur(U) # Equivalent to eigendecomposition
    D = np.diag(D) 
    D = D / np.abs(D)
    Theta = np.diag( np.imag(np.log(D)) % (2*np.pi) )
    H = Q @ Theta @ Q.conj().T
    return 0.5 * H + 0.5 * H.conj().T

def pauli_to_unitary(pauli_vector : np.array) -> np.array:
    num_qubits = int(np.log(len(pauli_vector))/np.log(4))
    i_alpha_sigma = 1j * dot_product(pauli_vector, PauliMatrices(num_qubits))
    unitary = expm(i_alpha_sigma)
    return to_canonical_unitary(unitary)

def unitary_to_pauli(unitary : np.array) -> np.array:
    canonical_unitary = to_canonical_unitary(unitary)
    H = unitary_log_no_i(canonical_unitary)
    return pauli_expansion(H)

def assert_closeness_paulis(list_of_paulis : list[np.array]) -> None:
    for a in list_of_paulis:
        for b in list_of_paulis:
            assert np.allclose(a, b)

def assert_closeness_unitaries(list_of_unitaries : list[np.array]) -> None:
    for a in list_of_unitaries:
        for b in list_of_unitaries:
            assert np.allclose(a, b)

def assert_distance_unitaries(list_of_unitaries : list[np.array]) -> None:
    for a in list_of_unitaries:
        for b in list_of_unitaries:
            dist = UnitaryMatrix(a).get_distance_from(b)
            assert dist <= 1e-6

def test_canonical_pauli():
    num_phases = 4
    for num_q in [2,3,4,5]:
        for _ in range(16):
            pauli = 2*np.pi * np.random.random((4**num_q,))
            u_from_random = pauli_to_unitary(pauli)
            shifted_randoms = [
                random_phase() * u_from_random for _ in range(num_phases)
            ]
            canonical_paulis = [
                unitary_to_pauli(sru) for sru in shifted_randoms
            ]
            assert_closeness_paulis(canonical_paulis)

def fast_pauli(unitary : np.array) -> np.array:
    norm = 1/(np.sqrt(8))
    PauliMatrices(3)

if __name__ == '__main__':
    num_qubits = 4
    #np.random.seed(100)
    num_unitaries = 25
    base_unitary = unitary_group.rvs(2**num_qubits)
    base_phases = [
        np.exp(-1j * 1*np.pi/4), np.exp(-1j * 3*np.pi/4), 
        np.exp(-1j * 5*np.pi/4), np.exp(-1j * 7*np.pi/4), 
    ]
    random_phases = [random_phase() for _ in range(num_unitaries-len(base_phases))]

    global_phases = base_phases + random_phases
    shifted_unitaries = [phase * base_unitary for phase in global_phases]

    canonical_paulis = [unitary_to_pauli(u) for u in shifted_unitaries]
    assert_closeness_paulis(canonical_paulis)

    canonical_unitaries = [to_canonical_unitary(u) for u in shifted_unitaries]
    assert_closeness_unitaries(canonical_unitaries)
    assert_distance_unitaries(canonical_unitaries)

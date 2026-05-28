# ============================================================
# DVQA para o Desafio de Mobilidade
# Versão com subcircuitos construídos em PennyLane
# alpha=lambda, beta=1-lambda
# Tensor C em MPS/TT com bond dimension chi
# ============================================================
#
# Ideia:
#
#   |phi> = (U0 ⊗ U1 ⊗ U2) |C_chi>
#
# onde:
#   - U0, U1, U2 são subcircuitos PennyLane de 3 qubits;
#   - cada Uk é um brickwall RX, RY, RZ;
#   - |C_chi> é um estado virtual global em MPS/TT;
#   - chi controla a bond dimension do tensor clássico.
#
# A diferença desta versão:
#   Os subcircuitos locais são construídos em PennyLane.
#   Depois usamos qml.matrix(...) para obter a matriz unitária Uk
#   e reconstruir o estado global via contração tensorial.
#
# ============================================================

import numpy as np
import torch
import matplotlib.pyplot as plt

from ket.pytorch import KetUnitaryBridge
from ket import RX, RY, RZ, CNOT

torch.set_default_dtype(torch.float64)

# ============================================================
# 1. Dados do problema
# ============================================================

names = ["Hub", "P1", "P2", "P3"]

tempo = np.array([
    [0, 15, 10, 25],
    [15, 0, 15, 10],
    [10, 15, 0, 15],
    [25, 10, 15, 0],
], dtype=float)

carbono = np.array([
    [0, 10, 30, 15],
    [10, 0, 10, 25],
    [30, 10, 0, 10],
    [15, 25, 10, 0],
], dtype=float)

tempo_norm = tempo / tempo[tempo > 0].max()
carbono_norm = carbono / carbono[carbono > 0].max()

# 9 qubits = 3 pontos x 3 posições.
n_points = 3
n_pos = 3
n_qubits = n_points * n_pos

# DVQA: 3 subcircuitos de 3 qubits.
K = 3
sub_n = 3
sub_dim = 2 ** sub_n
global_dim = 2 ** n_qubits

seeds = [0, 1, 2, 3, 4]


epochs= 500
LAYERS = 5
LR = 0.01

# ============================================================
# 2. Hamiltoniano diagonal
# ============================================================

def q(i, t):
    """
    Índice da variável x_{i,t}.
    i = 0,1,2 representa P1,P2,P3.
    t = 0,1,2 representa posição da rota.
    """
    return t * n_points + i


def bits_to_x(bits):
    x = np.zeros((n_points, n_pos), dtype=int)

    for i in range(n_points):
        for t in range(n_pos):
            x[i, t] = int(bits[q(i, t)])

    return x


def decode_route(bits):
    """
    Decodifica:
        Hub -> posição 0 -> posição 1 -> posição 2 -> Hub
    """
    x = bits_to_x(bits)
    route = [0]

    for t in range(n_pos):
        locs = np.where(x[:, t] == 1)[0]

        if len(locs) == 1:
            route.append(int(locs[0]) + 1)
        else:
            route.append(None)

    route.append(0)
    return route


def is_valid(bits):
    x = bits_to_x(bits)

    each_point_once = np.all(x.sum(axis=1) == 1)
    one_point_per_pos = np.all(x.sum(axis=0) == 1)

    return bool(each_point_once and one_point_per_pos)


def route_time_carbon(route):
    if None in route:
        return np.nan, np.nan

    total_time = 0.0
    total_carbon = 0.0

    for a, b in zip(route[:-1], route[1:]):
        total_time += tempo[a, b]
        total_carbon += carbono[a, b]

    return total_time, total_carbon


def energy_bits(bits, lam, A):
    """
    alpha = lambda
    beta  = 1 - lambda

    H = lambda H_carbono + (1-lambda) H_tempo + A H_restr.
    """
    alpha = lam
    beta = 1.0 - lam

    x = bits_to_x(bits)
    route = decode_route(bits)

    h_carbon = 0.0
    h_time = 0.0

    if None not in route:
        for a, b in zip(route[:-1], route[1:]):
            h_carbon += carbono_norm[a, b]
            h_time += tempo_norm[a, b]

    p_point = 0.0
    for i in range(n_points):
        p_point += (np.sum(x[i, :]) - 1.0) ** 2

    p_pos = 0.0
    for t in range(n_pos):
        p_pos += (np.sum(x[:, t]) - 1.0) ** 2

    return alpha * h_carbon + beta * h_time + A * (p_point + p_pos)


def build_energy_vector(lam, A):
    energies = np.zeros(global_dim)

    for s in range(global_dim):
        bits = np.array(list(np.binary_repr(s, width=n_qubits)), dtype=int)
        energies[s] = energy_bits(bits, lam, A)

    return torch.tensor(energies, dtype=torch.float64)


def brute_force_best_valid(lam, A):
    best_e = np.inf
    best_bits = None
    best_route = None

    for s in range(global_dim):
        bits = np.array(list(np.binary_repr(s, width=n_qubits)), dtype=int)

        if not is_valid(bits):
            continue

        e = energy_bits(bits, lam, A)

        if e < best_e:
            best_e = e
            best_bits = bits
            best_route = decode_route(bits)

    ttot, ctot = route_time_carbon(best_route)

    return best_e, best_bits, best_route, ttot, ctot


# ============================================================
# 3. Subcircuito local em Ket usando PyTorch
# ============================================================

# ============================================================
# Opção 1 (Similar ao código do Cesar)
# Manter aqui para ver possibilidade de adaptar o Ket
# para rodar dessa forma
# ============================================================

# def ket_brickwall_ansatz(theta_flat_np, layers):
#     """
#     O Ansatz puro do Ket. Recebe os ângulos em 1D e reconstrói para 3D.
#     """
#     theta_np = theta_flat_np.reshape(layers, 3, 3)
    
#     def circuito(q):
#         for l in range(layers):
#             for j, wire in enumerate([0, 1, 2]):
#                 RX(theta_np[l, j, 0], q[wire])
#                 RY(theta_np[l, j, 1], q[wire])
#                 RZ(theta_np[l, j, 2], q[wire])
            
#             # Brickwall local:
#             CNOT(q[0], q[1])
#             CNOT(q[1], q[2])
            
#     return circuito


# def local_unitary_from_ket(theta_sub_tensor):
#     """
#     Constrói a matriz 8x8 chamando a Ponte.
#     """
#     layers = theta_sub_tensor.shape[0]
#     theta_flat = theta_sub_tensor.reshape(-1)
    
#     # Usamos uma função anônima (lambda) para injetar o 'layers'
#     # sem que o PyTorch perceba!
#     ansatz_injetado = lambda t: ket_brickwall_ansatz(t, layers)

#     # Chama a ponte nativa
#     U = KetUnitaryBridge.apply(theta_flat, ansatz_injetado)
    
#     return U.to(torch.complex128)

# ============================================================
# Opção 2
# Ansatz criado dentro da unitária local do Ket
# ============================================================

def local_unitary_from_ket(theta_sub_tensor):
    """
    Constrói a matriz 8x8 do subcircuito local usando nossa PONTE Ket-PyTorch.
    theta_sub_tensor shape original: (layers, 3, 3)
    """
    layers = theta_sub_tensor.shape[0]
    
    # 1. Achatamos o tensor para 1D para a nossa ponte iterar os gradientes corretamente
    theta_flat = theta_sub_tensor.reshape(-1)
    
    # 2. Criamos o "Closure" do Ansatz do Ket
    def ket_ansatz(theta_flat_np):
        # Reconstrói o formato 3D dentro do simulador
        theta_np = theta_flat_np.reshape(layers, 3, 3)
        
        def circuito(q):
            for l in range(layers):
                for j, wire in enumerate([0, 1, 2]):
                    RX(theta_np[l, j, 0], q[wire])
                    RY(theta_np[l, j, 1], q[wire])
                    RZ(theta_np[l, j, 2], q[wire])
                
                # Brickwall local:
                CNOT(q[0], q[1])
                CNOT(q[1], q[2])
                
        return circuito

    # 3. CHAMA A PONTE (A mágica acontece aqui!)
    U = KetUnitaryBridge.apply(theta_flat, ket_ansatz)
    
    # Garante dtype complexo compatível com as contrações clássicas
    return U.to(torch.complex128)


# ============================================================
# 4. Tensor C em MPS/TT com bond dimension chi
# ============================================================

def build_C_from_mps(G0_re, G0_im, G1_re, G1_im, G2_re, G2_im):
    """
    C[a,b,c] = sum_{r1,r2} G0[a,r1] G1[r1,b,r2] G2[r2,c]
    """
    G0 = G0_re + 1j * G0_im
    G1 = G1_re + 1j * G1_im
    G2 = G2_re + 1j * G2_im

    C = torch.einsum("ar,rbs,sc->abc", G0, G1, G2)

    C = C / torch.sqrt(torch.sum(torch.abs(C) ** 2) + 1e-12)

    return C


def global_state(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im):
    """
    Reconstrói o estado global:

        |phi> = (U0 ⊗ U1 ⊗ U2) |C_chi>

    onde U0, U1 e U2 vêm de subcircuitos PennyLane.
    """
    C = build_C_from_mps(G0_re, G0_im, G1_re, G1_im, G2_re, G2_im)

    U0 = local_unitary_from_ket(theta[0])
    U1 = local_unitary_from_ket(theta[1])
    U2 = local_unitary_from_ket(theta[2])

    phi = torch.einsum("ai,bj,ck,ijk->abc", U0, U1, U2, C)

    psi = phi.reshape(-1)
    psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) + 1e-12)

    return psi


def expected_energy(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im, energies):
    psi = global_state(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im)
    probs = torch.abs(psi) ** 2
    return torch.sum(probs.real * energies)


# ============================================================
# 5. Treino DVQA
# ============================================================

def train_once(lam, A, chi, seed=0, layers=5, epochs=epochs, lr=0.01, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    energies = build_energy_vector(lam, A)

    # theta: 3 subcircuitos, layers camadas, 3 qubits, RX/RY/RZ.
    theta = torch.nn.Parameter(
        0.10 * torch.randn(K, layers, sub_n, 3, dtype=torch.float64)
    )

    # MPS do tensor C_chi.
    G0_re = torch.nn.Parameter(0.10 * torch.randn(sub_dim, chi, dtype=torch.float64))
    G0_im = torch.nn.Parameter(0.10 * torch.randn(sub_dim, chi, dtype=torch.float64))

    G1_re = torch.nn.Parameter(0.10 * torch.randn(chi, sub_dim, chi, dtype=torch.float64))
    G1_im = torch.nn.Parameter(0.10 * torch.randn(chi, sub_dim, chi, dtype=torch.float64))

    G2_re = torch.nn.Parameter(0.10 * torch.randn(chi, sub_dim, dtype=torch.float64))
    G2_im = torch.nn.Parameter(0.10 * torch.randn(chi, sub_dim, dtype=torch.float64))

    params = [theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im]
    opt = torch.optim.Adam(params, lr=lr)

    loss_hist = []

    for ep in range(epochs):
        opt.zero_grad()

        loss = expected_energy(
            theta,
            G0_re, G0_im,
            G1_re, G1_im,
            G2_re, G2_im,
            energies,
        )

        loss.backward()
        opt.step()

        loss_value = float(loss.detach())
        loss_hist.append(loss_value)

        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"epoch {ep:04d} | loss = {loss_value:.6f}")

    with torch.no_grad():
        psi = global_state(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im)
        probs = (torch.abs(psi) ** 2).real.cpu().numpy()

    return {
        "loss": loss_hist[-1],
        "loss_hist": loss_hist,
        "probs": probs,
    }


# ============================================================
# 6. Leitura das probabilidades
# ============================================================

def top_probability_validity(probs, top_k=20):
    idxs = np.argsort(probs)[::-1][:top_k]

    valid_count = 0
    invalid_count = 0
    valid_mass = 0.0
    invalid_mass = 0.0

    for idx in idxs:
        bits = np.array(list(np.binary_repr(int(idx), width=n_qubits)), dtype=int)

        if is_valid(bits):
            valid_count += 1
            valid_mass += probs[idx]
        else:
            invalid_count += 1
            invalid_mass += probs[idx]

    total_mass = valid_mass + invalid_mass

    if total_mass > 0:
        valid_ratio = valid_mass / total_mass
        invalid_ratio = invalid_mass / total_mass
    else:
        valid_ratio = np.nan
        invalid_ratio = np.nan

    return valid_count, invalid_count, valid_ratio, invalid_ratio


def best_valid_from_probs(probs):
    for idx in np.argsort(probs)[::-1]:
        bits = np.array(list(np.binary_repr(int(idx), width=n_qubits)), dtype=int)

        if is_valid(bits):
            route = decode_route(bits)
            ttot, ctot = route_time_carbon(route)

            return int(idx), bits, route, probs[idx], ttot, ctot

    return None, None, None, None, None, None


# ============================================================
# 7. Desenho textual do circuito PennyLane
# ============================================================

# def print_example_circuit(layers=5):
#     """
#     Mostra o subcircuito local de 3 qubits.
#     """
#     dev = qml.device("default.qubit", wires=3)

#     @qml.qnode(dev)
#     def example_qnode(params):
#         pennylane_brickwall_ansatz(params, wires=(0, 1, 2))
#         return qml.state()

#     params = np.zeros((layers, 3, 3))

#     print("\n=== Subcircuito local PennyLane ===")
#     print(qml.draw(example_qnode)(params))

def print_example_circuit(layers=5):
    print("\n============================================================")
    print("ARQUITETURA HÍBRIDA: PyTorch + Ket")
    print(f"-> Simulando subcircuitos locais com {layers} camadas (RX, RY, RZ + CNOT)")
    print("-> Gradientes quânticos (Parameter-Shift) rastreados via Autograd")
    print("============================================================\n")

# ============================================================
# 8. Experimento A: variar A
# ============================================================

print_example_circuit(layers=LAYERS)

A_values = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

lam_for_A = 0.5
chi_for_A = 2

invalid_counts = []
invalid_mass_ratios = []
valid_mass_ratios = []

print("\n=== Experimento A: variando penalidade A ===")
print("lambda fixo =", lam_for_A)
print("chi fixo    =", chi_for_A)

for A in A_values:
    out = train_once(
        lam=lam_for_A,
        A=A,
        chi=chi_for_A,
        seed=0,
        layers=LAYERS,
        epochs=epochs,
        lr=LR,
        verbose=False,
    )

    vc, ic, vr, ir = top_probability_validity(out["probs"], top_k=10)

    invalid_counts.append(ic)
    valid_mass_ratios.append(vr)
    invalid_mass_ratios.append(ir)

    print(
        f"A={A:5.2f} | inválidas top20 = {ic:2d}/20 | "
        f"massa inválida top20 = {ir:.3f}"
    )

plt.figure(figsize=(7, 4))
plt.plot(A_values, invalid_counts, marker="o")
plt.xlabel("Penalidade A")
plt.ylabel("Número de bitstrings inválidas no top 20")
plt.title("Rotas inválidas em função de A")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("A_invalid_count.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(A_values, valid_mass_ratios, marker="o", label="massa válida")
plt.plot(A_values, invalid_mass_ratios, marker="o", label="massa inválida")
plt.xlabel("Penalidade A")
plt.ylabel("Fração de probabilidade dentro do top 20")
plt.title("Probabilidade válida vs inválida em função de A")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("A_valid_invalid_mass.png", dpi=300)
plt.show()

A_best = 8.0


# ============================================================
# 9. Experimento B: curva lambda
# ============================================================

lambda_values = np.linspace(0.0, 1.0, 6)
chi_for_lambda = 2

lambda_energy_mean = []
lambda_energy_std = []
lambda_time_mean = []
lambda_carbon_mean = []
lambda_prob_mean = []
lambda_invalid_mean = []

print("\n=== Experimento B: variando lambda ===")
print("alpha=lambda, beta=1-lambda")
print("A fixo   =", A_best)
print("chi fixo =", chi_for_lambda)

for lam in lambda_values:
    losses = []
    times = []
    carbons = []
    probs_valid = []
    invalids = []
    route_texts = []

    for seed in seeds:
        out = train_once(
            lam=float(lam),
            A=A_best,
            chi=chi_for_lambda,
            seed=seed,
            layers=LAYERS,
            epochs=epochs,
            lr=LR,
            verbose=False,
        )

        losses.append(out["loss"])

        vc, ic, vr, ir = top_probability_validity(out["probs"], top_k=20)
        invalids.append(ic)

        idx, bits, route, prob, ttot, ctot = best_valid_from_probs(out["probs"])
        times.append(ttot)
        carbons.append(ctot)
        probs_valid.append(prob)
        route_texts.append(" -> ".join(names[i] for i in route))

    lambda_energy_mean.append(np.mean(losses))
    lambda_energy_std.append(np.std(losses))
    lambda_time_mean.append(np.mean(times))
    lambda_carbon_mean.append(np.mean(carbons))
    lambda_prob_mean.append(np.mean(probs_valid))
    lambda_invalid_mean.append(np.mean(invalids))

    print(
        f"lambda={lam:.2f} | "
        f"E média={np.mean(losses):.4f} | "
        f"tempo médio={np.mean(times):.1f} | "
        f"carbono médio={np.mean(carbons):.1f} | "
        f"prob válida média={np.mean(probs_valid):.3f} | "
        f"rota seed0={route_texts[0]}"
    )

lambda_energy_mean = np.array(lambda_energy_mean)
lambda_energy_std = np.array(lambda_energy_std)
lambda_time_mean = np.array(lambda_time_mean)
lambda_carbon_mean = np.array(lambda_carbon_mean)
lambda_prob_mean = np.array(lambda_prob_mean)
lambda_invalid_mean = np.array(lambda_invalid_mean)

plt.figure(figsize=(7, 4))
plt.errorbar(
    lambda_values,
    lambda_energy_mean,
    yerr=lambda_energy_std,
    marker="o",
    capsize=4,
    linewidth=2,
)
plt.xlabel(r"$\lambda$  |  $\alpha=\lambda$, $\beta=1-\lambda$")
plt.ylabel("Energia final média")
plt.title("Energia variacional final em função da preferência")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lambda_energy.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(lambda_values, lambda_time_mean, marker="o")
plt.xlabel(r"$\lambda$  |  0 = tempo, 1 = carbono")
plt.ylabel("Tempo médio da rota válida")
plt.title("Tempo da rota em função da preferência")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lambda_time.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(lambda_values, lambda_carbon_mean, marker="o")
plt.xlabel(r"$\lambda$  |  0 = tempo, 1 = carbono")
plt.ylabel("Carbono médio da rota válida")
plt.title("Carbono da rota em função da preferência")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lambda_carbon.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(lambda_values, lambda_prob_mean, marker="o")
plt.xlabel(r"$\lambda$")
plt.ylabel("Probabilidade média")
plt.title("Probabilidade da rota válida mais provável")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lambda_prob.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(lambda_values, lambda_invalid_mean, marker="o")
plt.xlabel(r"$\lambda$")
plt.ylabel("Número médio de inválidas no top 20")
plt.title("Inválidas no top 20 em função da preferência")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lambda_invalids.png", dpi=300)
plt.show()

# Escolha manual: compromisso equilibrado.
lambda_best = 0.5

print("\nLambda escolhido:")
print("lambda =", lambda_best)
print("alpha  =", lambda_best)
print("beta   =", 1.0 - lambda_best)


# ============================================================
# 10. Experimento C: variar chi
# ============================================================

chi_values = [1, 2, 3, 5, 7, 11]

chi_energy_mean = []
chi_energy_std = []
chi_prob_mean = []

print("\n=== Experimento C: variando chi ===")
for chi in chi_values:
    losses = []
    probs_valid = []

    for seed in seeds:
        out = train_once(
            lam=lambda_best,
            A=A_best,
            chi=chi,
            seed=seed,
            layers=LAYERS,
            epochs=epochs,
            lr=LR,
            verbose=False,
        )

        losses.append(out["loss"])

        idx, bits, route, prob, ttot, ctot = best_valid_from_probs(out["probs"])
        probs_valid.append(prob)

    chi_energy_mean.append(np.mean(losses))
    chi_energy_std.append(np.std(losses))
    chi_prob_mean.append(np.mean(probs_valid))

    print(
        f"chi={chi:2d} | "
        f"E média={np.mean(losses):.4f} ± {np.std(losses):.4f} | "
        f"prob válida média={np.mean(probs_valid):.3f}"
    )

chi_energy_mean = np.array(chi_energy_mean)
chi_energy_std = np.array(chi_energy_std)

plt.figure(figsize=(7, 4))
plt.errorbar(
    chi_values,
    chi_energy_mean,
    yerr=chi_energy_std,
    marker="o",
    capsize=4,
    linewidth=2,
)
plt.xlabel("Bond dimension chi")
plt.ylabel("Energia final média")
plt.title("Resultado final em função de chi - média sobre 5 seeds")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chi_energy.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(chi_values, chi_prob_mean, marker="o")
plt.xlabel("Bond dimension chi")
plt.ylabel("Probabilidade média da melhor rota válida")
plt.title("Probabilidade da rota válida em função de chi")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chi_prob.png", dpi=300)
plt.show()

best_chi = chi_values[int(np.argmin(chi_energy_mean))]

print("\nChi escolhido:")
print("chi =", best_chi)


# ============================================================
# 11. Treino final
# ============================================================

print("\n=== Treino final ===")
print("lambda =", lambda_best)
print("A      =", A_best)
print("chi    =", best_chi)

final = train_once(
    lam=lambda_best,
    A=A_best,
    chi=best_chi,
    seed=0,
    layers=LAYERS,
    epochs=epochs,
    lr=LR,
    verbose=True,
)

probs = final["probs"]

idx, bits, route, prob, ttot, ctot = best_valid_from_probs(probs)

print("\n=== Melhor rota válida encontrada ===")
print("bitstring:", "".join(map(str, bits)))
print("probabilidade:", prob)
print("rota:", " -> ".join(names[i] for i in route))
print("tempo:", ttot, "min")
print("carbono:", ctot, "kg")
print("lambda:", lambda_best, "| alpha:", lambda_best, "| beta:", 1.0 - lambda_best)
print("A:", A_best, "| chi:", best_chi)

e_exact, bits_exact, route_exact, t_exact, c_exact = brute_force_best_valid(lambda_best, A_best)

print("\n=== Melhor rota válida clássica para este lambda ===")
print("bitstring:", "".join(map(str, bits_exact)))
print("energia:", e_exact)
print("rota:", " -> ".join(names[i] for i in route_exact))
print("tempo:", t_exact, "min")
print("carbono:", c_exact, "kg")


# ============================================================
# 12. Plots finais
# ============================================================

plt.figure(figsize=(7, 4))
plt.plot(final["loss_hist"], linewidth=2)
plt.xlabel("Época")
plt.ylabel("Energia esperada")
plt.title("Loss do treino final")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_final.png", dpi=300)
plt.show()

top_k = 20
top_idx = np.argsort(probs)[::-1][:top_k]

labels = []
values = []

for idx_top in top_idx:
    b = np.array(list(np.binary_repr(int(idx_top), width=n_qubits)), dtype=int)
    bitstring = "".join(map(str, b))

    if is_valid(b):
        r = decode_route(b)
        route_txt = "->".join(names[i] for i in r)
        label = bitstring + "\n" + route_txt
    else:
        label = bitstring + "\ninválida"

    labels.append(label)
    values.append(probs[idx_top])

plt.figure(figsize=(13, 5))
plt.bar(range(top_k), values)
plt.xticks(range(top_k), labels, rotation=60, ha="right", fontsize=8)
plt.ylabel("Probabilidade")
plt.title("Probabilidades finais das top bitstrings - 9 qubits")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("top_bitstrings.png", dpi=300)
plt.show()

coords = {
    0: (0.0, 0.0),
    1: (1.0, 1.0),
    2: (1.0, -1.0),
    3: (-1.0, 0.0),
}

xs = [coords[i][0] for i in route]
ys = [coords[i][1] for i in route]

plt.figure(figsize=(5, 5))

for i in range(4):
    x, y = coords[i]
    plt.scatter(x, y, s=180)
    plt.text(x + 0.04, y + 0.04, names[i], fontsize=12)

plt.plot(xs, ys, marker="o", linewidth=2)

for step, loc in enumerate(route[:-1]):
    x, y = coords[loc]
    plt.text(x - 0.12, y - 0.12, f"{step}", fontsize=11)

plt.title("Rota final: " + " -> ".join(names[i] for i in route))
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rota_final.png", dpi=300)
plt.show()

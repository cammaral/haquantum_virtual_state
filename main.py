# ============================================================
# DVQA simples para o Desafio de Mobilidade
# Com tensor C em formato MPS/TT e variação de chi
# ============================================================
#
# Este script faz:
#
# 1) Quebra o problema em 3 subcircuitos de brickwall RX, RY, RZ.
#    Cada subcircuito tem 3 qubits.
#
# 2) Reconstrói um estado global:
#
#       |phi> = (U0 ⊗ U1 ⊗ U2) |C_chi>
#
#    onde C_chi é um tensor clássico em formato MPS/TT:
#
#       C[a,b,c] = sum_{r1,r2} G0[a,r1] G1[r1,b,r2] G2[r2,c]
#
#    O parâmetro chi é a bond dimension.
#
# 3) Calcula o valor esperado:
#
#       <H> = sum_z p(z) H(z)
#
# 4) Faz um grid simples em alpha e beta com 5 seeds.
#
# 5) Escolhe o melhor alpha e beta pelo menor valor médio de energia.
#
# 6) Com esse melhor alpha e beta, treina para diferentes chi e plota:
#
#       chi  x  energia final média sobre 5 seeds
#
# 7) Faz um treino final com o melhor chi e plota as probabilidades finais.
#
# ============================================================

import numpy as np
import torch
import matplotlib.pyplot as plt

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

# Usamos 9 qubits:
# 3 pontos de entrega x 3 posições na rota.
n_points = 3
n_pos = 3
n_qubits = n_points * n_pos

# DVQA:
# 3 subcircuitos de 3 qubits.
K = 3
sub_n = 3
sub_dim = 2 ** sub_n
global_dim = 2 ** n_qubits

# ============================================================
# 2. Hamiltoniano diagonal do problema
# ============================================================

def q(i, t):
    """
    Índice do qubit da variável x_{i,t}.

    i = 0,1,2 representa P1,P2,P3.
    t = 0,1,2 representa a posição na rota.

    Organização:
        t=0: q0,q1,q2
        t=1: q3,q4,q5
        t=2: q6,q7,q8
    """
    return t * n_points + i


def bits_to_x(bits):
    """
    Transforma a bitstring de 9 bits em matriz x[i,t].
    """
    x = np.zeros((n_points, n_pos), dtype=int)

    for i in range(n_points):
        for t in range(n_pos):
            x[i, t] = int(bits[q(i, t)])

    return x


def decode_route(bits):
    """
    Converte a bitstring em rota:

        Hub -> ... -> ... -> ... -> Hub

    Se alguma posição tiver zero ou mais de um ponto,
    aparece None naquela posição.
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
    """
    Rota válida:
    - cada ponto aparece exatamente uma vez;
    - cada posição tem exatamente um ponto.
    """
    x = bits_to_x(bits)

    each_point_once = np.all(x.sum(axis=1) == 1)
    one_point_per_pos = np.all(x.sum(axis=0) == 1)

    return bool(each_point_once and one_point_per_pos)


def route_time_carbon(route):
    """
    Calcula tempo e carbono reais da rota.
    """
    if None in route:
        return np.nan, np.nan

    total_time = 0.0
    total_carbon = 0.0

    for a, b in zip(route[:-1], route[1:]):
        total_time += tempo[a, b]
        total_carbon += carbono[a, b]

    return total_time, total_carbon


def energy_bits(bits, alpha, beta, A):
    """
    Energia de uma bitstring:

        H = alpha H_carbono + beta H_tempo + A H_restricoes

    As restrições são one-hot:
    - cada ponto aparece uma vez;
    - cada posição recebe um ponto.
    """
    x = bits_to_x(bits)
    route = decode_route(bits)

    h_carbon = 0.0
    h_time = 0.0

    # Se a posição é decodificável, calculamos o custo do caminho.
    if None not in route:
        for a, b in zip(route[:-1], route[1:]):
            h_carbon += carbono_norm[a, b]
            h_time += tempo_norm[a, b]

    # Penalidade: cada ponto aparece uma vez.
    p_point = 0.0
    for i in range(n_points):
        p_point += (np.sum(x[i, :]) - 1.0) ** 2

    # Penalidade: cada posição recebe um ponto.
    p_pos = 0.0
    for t in range(n_pos):
        p_pos += (np.sum(x[:, t]) - 1.0) ** 2

    return alpha * h_carbon + beta * h_time + A * (p_point + p_pos)


def build_energy_vector(alpha, beta, A):
    """
    Cria o vetor H(z) para todos os 2^9 estados.
    """
    energies = np.zeros(global_dim)

    for s in range(global_dim):
        bits = np.array(list(np.binary_repr(s, width=n_qubits)), dtype=int)
        energies[s] = energy_bits(bits, alpha, beta, A)

    return torch.tensor(energies, dtype=torch.float64)


# ============================================================
# 3. Portas RX, RY, RZ e subcircuito brickwall
# ============================================================

I2 = torch.eye(2, dtype=torch.complex128)

def RX(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    return torch.stack([
        torch.stack([c + 0j, -1j * s]),
        torch.stack([-1j * s, c + 0j]),
    ])


def RY(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    return torch.stack([
        torch.stack([c + 0j, -s + 0j]),
        torch.stack([s + 0j, c + 0j]),
    ])


def RZ(theta):
    zero = torch.tensor(0.0 + 0.0j, dtype=torch.complex128)

    return torch.stack([
        torch.stack([torch.exp(-0.5j * theta), zero]),
        torch.stack([zero, torch.exp(0.5j * theta)]),
    ])


def kron3(A, B, C):
    return torch.kron(torch.kron(A, B), C)


def one_qubit_gate_on_3q(G, wire):
    ops = []

    for w in range(3):
        if w == wire:
            ops.append(G)
        else:
            ops.append(I2)

    return kron3(ops[0], ops[1], ops[2])


def cnot_3q(control, target):
    """
    CNOT em 3 qubits como matriz 8x8.
    """
    U = torch.zeros((8, 8), dtype=torch.complex128)

    for s in range(8):
        bits = [int(b) for b in np.binary_repr(s, width=3)]

        if bits[control] == 1:
            bits[target] = 1 - bits[target]

        out = int("".join(str(b) for b in bits), 2)
        U[out, s] = 1.0 + 0.0j

    return U


CNOT_01 = cnot_3q(0, 1)
CNOT_12 = cnot_3q(1, 2)


def local_brickwall_unitary(theta_sub):
    """
    Unidade local Uk de 3 qubits.

    theta_sub shape:
        (layers, 3, 3)

    theta_sub[l,w,0] = RX
    theta_sub[l,w,1] = RY
    theta_sub[l,w,2] = RZ
    """
    U = torch.eye(8, dtype=torch.complex128)

    layers = theta_sub.shape[0]

    for l in range(layers):
        for w in range(3):
            U = one_qubit_gate_on_3q(RX(theta_sub[l, w, 0]), w) @ U
            U = one_qubit_gate_on_3q(RY(theta_sub[l, w, 1]), w) @ U
            U = one_qubit_gate_on_3q(RZ(theta_sub[l, w, 2]), w) @ U

        # Brickwall em 3 qubits.
        U = CNOT_01 @ U
        U = CNOT_12 @ U

    return U


# ============================================================
# 4. Tensor C em MPS/TT com bond dimension chi
# ============================================================

def build_C_from_mps(G0_re, G0_im, G1_re, G1_im, G2_re, G2_im):
    """
    Constrói C[a,b,c] a partir dos tensores MPS:

        G0[a,r1]
        G1[r1,b,r2]
        G2[r2,c]

    C[a,b,c] = sum_{r1,r2} G0[a,r1] G1[r1,b,r2] G2[r2,c]
    """
    G0 = G0_re + 1j * G0_im
    G1 = G1_re + 1j * G1_im
    G2 = G2_re + 1j * G2_im

    C = torch.einsum("ar,rbs,sc->abc", G0, G1, G2)

    # Normaliza C para representar um estado.
    C = C / torch.sqrt(torch.sum(torch.abs(C) ** 2) + 1e-12)

    return C


def global_state(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im):
    """
    Reconstrói:

        |phi> = (U0 ⊗ U1 ⊗ U2) |C_chi>

    sem montar a matriz 512 x 512.
    """
    C = build_C_from_mps(G0_re, G0_im, G1_re, G1_im, G2_re, G2_im)

    U0 = local_brickwall_unitary(theta[0])
    U1 = local_brickwall_unitary(theta[1])
    U2 = local_brickwall_unitary(theta[2])

    # phi[a,b,c] = sum_{i,j,k} U0[a,i] U1[b,j] U2[c,k] C[i,j,k]
    phi = torch.einsum("ai,bj,ck,ijk->abc", U0, U1, U2, C)

    psi = phi.reshape(-1)
    psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) + 1e-12)

    return psi


def expected_energy(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im, energies):
    """
    <H> = sum_z |psi_z|^2 H_z.
    """
    psi = global_state(theta, G0_re, G0_im, G1_re, G1_im, G2_re, G2_im)
    probs = torch.abs(psi) ** 2
    return torch.sum(probs.real * energies)


# ============================================================
# 5. Treino DVQA para um dado chi
# ============================================================

def train_once(alpha, beta, A, chi, seed=0, layers=5, epochs=300, lr=0.01, verbose=False):
    """
    Treina uma vez o DVQA para alpha, beta, A e chi fixos.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    energies = build_energy_vector(alpha, beta, A)

    # Parâmetros quânticos locais:
    # K subcircuitos, layers camadas, 3 qubits, 3 rotações.
    theta = torch.nn.Parameter(
        0.10 * torch.randn(K, layers, sub_n, 3, dtype=torch.float64)
    )

    # Tensores MPS/TT de C.
    #
    # G0: (8, chi)
    # G1: (chi, 8, chi)
    # G2: (chi, 8)
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
# 6. Métricas e leitura das soluções
# ============================================================

def top_probability_validity(probs, top_k=20):
    """
    Conta quantas das top_k bitstrings mais prováveis são inválidas.
    Também mede a massa de probabilidade inválida dentro do top_k.
    """
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
    """
    Retorna a bitstring válida mais provável.
    """
    for idx in np.argsort(probs)[::-1]:
        bits = np.array(list(np.binary_repr(int(idx), width=n_qubits)), dtype=int)

        if is_valid(bits):
            route = decode_route(bits)
            ttot, ctot = route_time_carbon(route)

            return int(idx), bits, route, probs[idx], ttot, ctot

    return None, None, None, None, None, None


# ============================================================
# 7. Experimento 1: variar A
# ============================================================

alpha_fixed = 1.0
beta_fixed = 0.6
chi_fixed = 2

A_values = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

invalid_counts = []
invalid_mass_ratios = []
valid_mass_ratios = []

print("\n=== Experimento 1: variando A ===")
for A in A_values:
    out = train_once(
        alpha=alpha_fixed,
        beta=beta_fixed,
        A=A,
        chi=chi_fixed,
        seed=0,
        layers=5,
        epochs=300,
        lr=0.01,
        verbose=False,
    )

    vc, ic, vr, ir = top_probability_validity(out["probs"], top_k=20)

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
plt.title("Validade das rotas em função de A")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("validity_vs_A.png", dpi=300)
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
plt.savefig("probability_mass_vs_A.png", dpi=300)
plt.show()


# ============================================================
# 8. Experimento 2: grid alpha x beta com 5 seeds
# ============================================================

A_best = 8.0
chi_for_grid = 2

alpha_grid = [0.4, 0.8, 1.2]
beta_grid = [0.4, 0.8, 1.2]
seeds = [0, 1, 2, 3, 4]

mean_energy = np.zeros((len(alpha_grid), len(beta_grid)))
std_energy = np.zeros_like(mean_energy)

print("\n=== Experimento 2: grid alpha x beta com 5 seeds ===")
for ia, alpha in enumerate(alpha_grid):
    for ib, beta in enumerate(beta_grid):
        vals = []

        for seed in seeds:
            out = train_once(
                alpha=alpha,
                beta=beta,
                A=A_best,
                chi=chi_for_grid,
                seed=seed,
                layers=5,
                epochs=300,
                lr=0.01,
                verbose=False,
            )
            vals.append(out["loss"])

        mean_energy[ia, ib] = np.mean(vals)
        std_energy[ia, ib] = np.std(vals)

        print(
            f"alpha={alpha:.2f}, beta={beta:.2f} | "
            f"energia média = {mean_energy[ia, ib]:.6f} ± {std_energy[ia, ib]:.6f}"
        )

plt.figure(figsize=(6, 5))
plt.imshow(mean_energy, origin="lower", aspect="auto")
plt.colorbar(label="Energia final média")

plt.xticks(range(len(beta_grid)), beta_grid)
plt.yticks(range(len(alpha_grid)), alpha_grid)

plt.xlabel("beta")
plt.ylabel("alpha")
plt.title("Heatmap alpha x beta: energia média sobre 5 seeds")

for ia in range(len(alpha_grid)):
    for ib in range(len(beta_grid)):
        plt.text(
            ib,
            ia,
            f"{mean_energy[ia, ib]:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

plt.tight_layout()
plt.savefig("alpha_beta_heatmap.png", dpi=300)
plt.show()

best_pos = np.unravel_index(np.argmin(mean_energy), mean_energy.shape)
best_alpha = alpha_grid[best_pos[0]]
best_beta = beta_grid[best_pos[1]]

print("\nMelhor par alpha,beta:")
print("alpha =", best_alpha)
print("beta  =", best_beta)
print("A     =", A_best)


# ============================================================
# 9. Experimento 3: chi x resultado final médio sobre 5 seeds
# ============================================================

chi_values = [1, 2, 3, 4, 6]

chi_mean_energy = []
chi_std_energy = []

print("\n=== Experimento 3: variando chi com melhor alpha,beta ===")
for chi in chi_values:
    vals = []

    for seed in seeds:
        out = train_once(
            alpha=best_alpha,
            beta=best_beta,
            A=A_best,
            chi=chi,
            seed=seed,
            layers=5,
            epochs=300,
            lr=0.01,
            verbose=False,
        )
        vals.append(out["loss"])

    mean_val = np.mean(vals)
    std_val = np.std(vals)

    chi_mean_energy.append(mean_val)
    chi_std_energy.append(std_val)

    print(f"chi={chi:2d} | energia final média = {mean_val:.6f} ± {std_val:.6f}")

chi_mean_energy = np.array(chi_mean_energy)
chi_std_energy = np.array(chi_std_energy)

plt.figure(figsize=(7, 4))
plt.errorbar(
    chi_values,
    chi_mean_energy,
    yerr=chi_std_energy,
    marker="o",
    capsize=4,
    linewidth=2,
)
plt.xlabel("Bond dimension chi")
plt.ylabel("Energia final média")
plt.title("Resultado final em função de chi - média sobre 5 seeds")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("energy_vs_chi.png", dpi=300)
plt.show()

best_chi = chi_values[int(np.argmin(chi_mean_energy))]

print("\nMelhor chi:")
print("chi =", best_chi)


# ============================================================
# 10. Treino final com melhor alpha, beta e chi
# ============================================================

print("\n=== Treino final com melhor alpha, beta e chi ===")
final = train_once(
    alpha=best_alpha,
    beta=best_beta,
    A=A_best,
    chi=best_chi,
    seed=0,
    layers=5,
    epochs=300,
    lr=0.01,
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

# Loss final.
plt.figure(figsize=(7, 4))
plt.plot(final["loss_hist"], linewidth=2)
plt.xlabel("Época")
plt.ylabel("Energia esperada")
plt.title("Loss do treino final")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("final_loss.png", dpi=300)
plt.show()

# Probabilidades finais.
top_k = 20
top_idx = np.argsort(probs)[::-1][:top_k]

labels = []
values = []

for idx in top_idx:
    b = np.array(list(np.binary_repr(int(idx), width=n_qubits)), dtype=int)
    bitstring = "".join(map(str, b))

    if is_valid(b):
        r = decode_route(b)
        route_txt = "->".join(names[i] for i in r)
        label = bitstring + "\n" + route_txt
    else:
        label = bitstring + "\ninválida"

    labels.append(label)
    values.append(probs[idx])

plt.figure(figsize=(13, 5))
plt.bar(range(top_k), values)
plt.xticks(range(top_k), labels, rotation=60, ha="right", fontsize=8)
plt.ylabel("Probabilidade")
plt.title("Probabilidades finais das top bitstrings - 9 qubits")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("final_probabilities.png", dpi=300)
plt.show()


# ============================================================
# 11. Plot simples da rota final
# ============================================================

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
plt.savefig("final_route.png", dpi=300)
plt.show()

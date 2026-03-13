# QuTiP
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# =====================================================================
# Exemplo: Oscilações de Rabi com decaimento (T1 e T2*)
# Um qubit impulsionado por campo clássico ressonante
# =====================================================================

# Parâmetros físicos (em unidades onde ħ = 1)
omega_0 = 0.0       # frequência do qubit (detuning = 0 → ressonante)
Omega = 2 * np.pi * 10   # frequência de Rabi (em MHz, por exemplo)
gamma1 = 2 * np.pi * 0.5  # taxa de relaxação (T1 = 1/gamma1 ≈ 2 μs)
gamma2 = 2 * np.pi * 1.0  # taxa de dephasing (T2* = 1/gamma2 ≈ 1 μs)

# Tempo de simulação
tlist = np.linspace(0.0, 0.5, 500)   # tempo em μs

# Operadores do qubit
sz = sigmaz()
sx = sigmax()
sm = sigmam()           # operador de abaixamento σ⁻
sp = sigma_plus()       # operador de elevação σ⁺

# Hamiltoniano (drive ressonante + termo de detuning)
H = -0.5 * omega_0 * sz + 0.5 * Omega * sx

# Colapso operators (dissipação)
c_ops = [
    np.sqrt(gamma1) * sm,           # relaxação energia (T1)
    np.sqrt(gamma2) * sz            # dephasing puro (T2*)
]

# Estado inicial: |0⟩ (ground state)
psi0 = basis(2, 0)

# Evolução temporal usando master equation (Lindblad)
result = mesolve(H, psi0, tlist, c_ops, [sx, sz])

# Extrai as expectativas <σx> e <σz>
sx_exp = result.expect[0]
sz_exp = result.expect[1]

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
plt.plot(tlist, sx_exp, label=r'$\langle \sigma_x \rangle$', color='C0', lw=2)
plt.plot(tlist, sz_exp, label=r'$\langle \sigma_z \rangle$', color='C1', lw=2)
plt.xlabel('Tempo (μs)')
plt.ylabel('Valor esperado')
plt.title('Oscilações de Rabi com decaimento (T1 e T2*)')
plt.legend()
plt.grid(True, alpha=0.3)

# Diagrama de Bloch simplificado (últimos pontos)
b = Bloch(fig=plt.figure(figsize=(5,5)))
b.add_points([sx_exp[-1], 0, sz_exp[-1]], 'r')   # ponto final
b.add_vectors([sx_exp[-1], 0, sz_exp[-1]], 'r', alpha=0.7)
b.show()

plt.tight_layout()
plt.show()

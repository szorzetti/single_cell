from qutip import *
import numpy as np
import matplotlib.pyplot as plt

wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
#kappa = 0.005  # cavity dissipation rate
#gamma = 0.05  # atom dissipation rate
kappa=0.05
gamma=0.005
N = 10  # number of cavity fock states
n_th_a = 0.0  # temperature in frequency units
use_rwa = True

tlist = np.linspace(0, 25, 100)


# intial state
psi0 = tensor(basis(N, 0), basis(2, 1))  # start with an excited atom



# operators
a = tensor(destroy(N),qeye(2))
sm = tensor(qeye(N),destroy(2))

# Hamiltonian
#H = wc1 * a.dag() * a + wc2 * b.dag() * b + wa * sm.dag() * sm + g1 * (a.dag() * sm + a * sm.dag()) + g2 * (
   #         b.dag() * sm + b * sm.dag())  # + 20 * a.dag() * a * a.dag() * a

H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())

# The last term is the Kerr term

c_op_list = []

rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * a)

rate = kappa * n_th_a
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * a.dag())

rate = gamma
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sm)


output = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])



fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist, output.expect[0], label="$<n_1>$")
ax.plot(tlist, output.expect[1], label="Atom excited state")
ax.legend()
ax.set_xlabel('Time (arbitrary units)')
ax.set_ylabel('Occupation probability')
ax.set_title('Rabi oscillations | Two-cavity + one transmon | $g_1/g_2 = 1/3$');
filename = 'fig_single_cell_rabi.png'
fig.savefig(filename)


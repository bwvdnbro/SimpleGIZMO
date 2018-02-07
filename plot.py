import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl

# Computes the analytical solution of the Sod shock

# Parameters
gas_gamma = 5./3.
rho_L = 1.
rho_R = 0.125
v_L = 0.
v_R = 0.
P_L = 1.
P_R = 0.1

time = 0.1

N = 1000  # Number of points

c_L = np.sqrt(gas_gamma * P_L / rho_L)
c_R = np.sqrt(gas_gamma * P_R / rho_R)

# Helpful variables
Gama = (gas_gamma - 1.) / (gas_gamma + 1.)
beta = (gas_gamma - 1.) / (2. * gas_gamma)

# Characteristic function and its derivative, following Toro (2009)
def compute_f(P_3, P, c):
  u = P_3 / P
  if u > 1.:
    term1 = gas_gamma * ((gas_gamma + 1.) * u + gas_gamma - 1.)
    term2 = np.sqrt(2. / term1)
    fp = (u - 1.) * c * term2
    dfdp = c * term2 / P + \
           (u - 1.) * c / term2 * (-1. / term1**2) * gas_gamma * \
             (gas_gamma + 1.) / P
  else:
    fp = (u**beta - 1.) * (2. * c / (gas_gamma - 1.))
    dfdp = 2. * c / (gas_gamma - 1.) * beta * u**(beta - 1.) / P
  return (fp, dfdp)

# Solution of the Riemann problem following Toro (2009) 
def RiemannProblem(rho_L, P_L, v_L, rho_R, P_R, v_R):
  P_new = ((c_L + c_R + (v_L - v_R) * 0.5 * (gas_gamma - 1.)) / \
           (c_L / P_L**beta + c_R / P_R**beta))**(1. / beta)
  P_3 = 0.5 * (P_R + P_L)
  f_L = 1.
  while abs(P_3 - P_new) > 1.e-6:
    P_3 = P_new
    (f_L, dfdp_L) = compute_f(P_3, P_L, c_L)
    (f_R, dfdp_R) = compute_f(P_3, P_R, c_R)
    f = f_L + f_R + (v_R - v_L)
    df = dfdp_L + dfdp_R
    dp =  -f / df
    prnew = P_3 + dp
  v_3 = v_L - f_L
  return (P_new, v_3)

# Solve Riemann problem for post-shock region
(P_3, v_3) = RiemannProblem(rho_L, P_L, v_L, rho_R, P_R, v_R)

# Check direction of shocks and wave
shock_R = (P_3 > P_R)
shock_L = (P_3 > P_L)

# Velocity of shock front and and rarefaction wave
if shock_R:
  v_right = v_R + c_R**2 * (P_3 / P_R - 1.) / (gas_gamma * (v_3 - v_R))
else:
  v_right = c_R + 0.5 * (gas_gamma + 1.) * v_3 - 0.5 * (gas_gamma - 1.) * v_R

if shock_L:
  v_left = v_L + c_L**2 * (P_3 / p_L - 1.) / (gas_gamma * (v_3 - v_L))
else:
  v_left = c_L - 0.5 * (gas_gamma + 1.) * v_3 + 0.5 * (gas_gamma - 1.) *v_L

# Compute position of the transitions
x_23 = -abs(v_left) * time
if shock_L :
  x_12 = -abs(v_left) * time
else:
  x_12 = -(c_L - v_L) * time

x_34 = v_3 * time

x_45 = abs(v_right) * time
if shock_R:
  x_56 = abs(v_right) * time
else:
  x_56 = (c_R + v_R) * time

# Prepare arrays
delta_x = 0.5 / N
x_s = np.arange(-0.25, 0.25, delta_x)
rho_s = np.zeros(N)
P_s = np.zeros(N)
v_s = np.zeros(N)

# Compute solution in the different regions
for i in range(N):
  if x_s[i] <= x_12:
    rho_s[i] = rho_L
    P_s[i] = P_L
    v_s[i] = v_L
  if x_s[i] >= x_12 and x_s[i] < x_23:
    if shock_L:
      rho_s[i] = rho_L * (Gama + P_3 / P_L) / (1. + Gama * P_3 / P_L)
      P_s[i] = P_3
      v_s[i] = v_3
    else:
      rho_s[i] = rho_L * (Gama * (0. - x_s[i]) / (c_L * time) + \
                 Gama * v_L / c_L + (1. - Gama))**(2. / (gas_gamma - 1.))
      P_s[i] = P_L * (rho_s[i] / rho_L)**gas_gamma
      v_s[i] = (1. - Gama) * (c_L - (0. - x_s[i]) / time) + Gama * v_L
  if x_s[i] >= x_23 and x_s[i] < x_34:
    if shock_L:
      rho_s[i] = rho_L * (Gama + P_3 / P_L) / (1. + Gama * P_3 / p_L)
    else:
      rho_s[i] = rho_L * (P_3 / P_L)**(1. / gas_gamma)
    P_s[i] = P_3
    v_s[i] = v_3
  if x_s[i] >= x_34 and x_s[i] < x_45:
    if shock_R:
      rho_s[i] = rho_R * (Gama + P_3 / P_R) / (1. + Gama * P_3 / P_R)
    else:
      rho_s[i] = rho_R * (P_3 / P_R)**(1. / gas_gamma)
    P_s[i] = P_3
    v_s[i] = v_3
  if x_s[i] >= x_45 and x_s[i] < x_56:
    if shock_R:
      rho_s[i] = rho_R
      P_s[i] = P_R
      v_s[i] = v_R
    else:
      rho_s[i] = rho_R * (Gama * (x_s[i]) / (c_R * time) - \
                 Gama * v_R / c_R + (1. - Gama))**(2. / (gas_gamma - 1.))
      P_s[i] = p_R * (rho_s[i] / rho_R)**gas_gamma
      v_s[i] = (1. - Gama) * (-c_R - (-x_s[i]) / time) + Gama * v_R
  if x_s[i] >= x_56:
    rho_s[i] = rho_R
    P_s[i] = P_R
    v_s[i] = v_R

data = np.loadtxt("result.txt")

fig, ax = pl.subplots(3, 1, sharex = True, figsize = (6., 8.))

# density
ax[0].plot(x_s + 0.5, rho_s, "r-")
ax[0].plot(- x_s, rho_s, "r-")
ax[0].plot(1. - x_s, rho_s, "r-")
ax[0].plot(data[:,0], data[:,1], "k.")

# velocity
ax[1].plot(x_s + 0.5, v_s, "r-")
ax[1].plot(- x_s, -v_s, "r-")
ax[1].plot(1. - x_s, -v_s, "r-")
ax[1].plot(data[:,0], data[:,2], "k.")

# pressure
ax[2].plot(x_s + 0.5, P_s, "r-")
ax[2].plot(- x_s, P_s, "r-")
ax[2].plot(1. - x_s, P_s, "r-")
ax[2].plot(data[:,0], data[:,3], "k.")

# axis properties
ax[0].set_xlim(0., 1.)
ax[0].set_ylabel("density")
ax[1].set_ylabel("velocity")
ax[2].set_ylabel("pressure")
ax[2].set_xlabel("position")

pl.tight_layout()
pl.savefig("result.png")

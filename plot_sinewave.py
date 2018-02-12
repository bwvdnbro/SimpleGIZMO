import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl

amplitude = 1.

x = np.linspace(0., 1., 1000)
rho_x = 1.*np.exp(-0.5*amplitude/np.pi*np.cos(2.*np.pi*x))

data = np.loadtxt("result.txt")
data_noe = np.loadtxt("result_noenergy.txt")
data_mf = np.loadtxt("result_mass_flux.txt")

original_positions = np.arange(0.5 / len(data), 1., 1. / len(data))
original_positions_noe = np.arange(0.5 / len(data_noe), 1., 1. / len(data_noe))
original_positions_mf = np.arange(0.5 / len(data_mf), 1., 1. / len(data_mf))
displacement = np.abs(data[:,0] - original_positions)
displacement_noe = np.abs(data_noe[:,0] - original_positions_noe)
displacement_mf = np.abs(data_mf[:,0] - original_positions_mf)

fig, ax = pl.subplots(2, 2, sharex = True)

# density
ax[0][0].plot(data_noe[:,0], data_noe[:,1], "b.")
ax[0][0].plot(data[:,0], data[:,1], "k.")
ax[0][0].plot(data_mf[:,0], data_mf[:,1], "y.")
ax[0][0].plot(x, rho_x, "r-")

# velocity
ax[0][1].plot(data_noe[:,0], data_noe[:,2], "b.")
ax[0][1].plot(data_mf[:,0], data_mf[:,2], "y.")
ax[0][1].plot(data[:,0], data[:,2], "k.")
ax[0][1].plot(x, np.zeros(len(x)), "r-")

# pressure
ax[1][0].plot(data_noe[:,0], data_noe[:,3], "b.")
ax[1][0].plot(data_mf[:,0], data_mf[:,3], "y.")
ax[1][0].plot(data[:,0], data[:,3], "k.")
ax[1][0].plot(x, rho_x, "r-")

# particle movement
#ax[1][1].plot(data_noe[:,0], displacement_noe, "b.")
#ax[1][1].plot(data_mf[:,0], displacement_mf, "y.")
#ax[1][1].plot(data[:,0], displacement, "k.")
u = 1.5 * data[:,3] / data[:,1]
ax[1][1].plot(data[:,0], u, "k.")
ax[1][1].plot(x, 1.5 * (1. - 0.01 * abs(0.5 -x)), "r-")

# axis properties
ax[0][0].set_xlim(0., 1.)
ax[0][0].set_ylabel("density")
ax[0][1].set_ylabel("velocity")
ax[1][0].set_ylabel("pressure")
ax[1][1].set_ylabel("specific energy")
ax[1][0].set_xlabel("position")
ax[1][1].set_xlabel("position")

pl.tight_layout()
pl.savefig("result.png")

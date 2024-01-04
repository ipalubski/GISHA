This is Gizmo-ISolated-Halo-Analysis (GISHA); A set of tools for multivariate analysis of N-body simulations of isolated self-interacting dark matter (SIDM) halos.
GISHA easily extractrs information form any number of simulation snapshots, centers the halo using particle local densities determined with Smoothed-Particle Hydrodynamics (SPH),
calculates the density and velocity dispersion profiles, the expected scattering rate for a given SIDM model as well as the halo energies. <br>
<br>
Dependencies: numpy, scipy, pandas, csv, h5py, astropy, matplotlib <br>
<br>
Parameters:<br>
a: Inner radius for data binning<br>
b: Outer radius for data binning<br>
n: Number of bins<br>
sigma: interaction cross section normalization constant<br>
w: Yukawa velocity scale factor<br>
dt: time between snapshots in Gyr<br>
rho_s: NFW characteristic density of the initial halo<br>
rs: NFW characteristic radius of the initial halov<br>
<br>
Example use:<br>
import GISHA<br>
halo = GISHA.gisha(NSItable = False, write = False) # No scattering table used for analysis<br>
halo.analyze(name = "example_simulation_directory", a = 0.001, b = 30, n = 100, sigma = 10.0, w = 0, dt = 0.1, rho_s = 2.73e7, rs = 1.18)<br>
<br>
Example on-the-fly output includes the central density, halo energy, scattering rate as a fraction of the expected rate, velocity dispersion as well as the density profile of the halo:<br>

![Unknown-24](https://github.com/ipalubski/GISHA/assets/46392921/0a7fe08f-e48f-4d75-927c-6f35810eca3d)

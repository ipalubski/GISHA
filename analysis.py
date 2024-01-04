import h5py
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
import csv
import astropy.units as ut
from dataclasses import dataclass, fields, field
from typing import List
import warnings
from IPython.display import clear_output
warnings.filterwarnings('ignore')

class gisha():
    
    def __init__(self, Strip=False, CalcVel=True, Geth = True, DMRho = True, DMN = True, NSItable = True, Potential = True, DMRCM = False, M1 = False, M2 = True, M3 = False, Plot = True, write = True, e = 1000):
        self.Strip = Strip #remove particles outside some radius
        self.CalcVel = CalcVel #Calculate velocity variables: mean total, radial velocity per binning shell, and velocity dispersion
        self.Geth = Geth #Read the force softening values form snapshots
        self.DMRho = DMRho # Read the local density values from snapshots
        self.DMN = DMN # Read the interaction count from snaphots
        self.NSItable = NSItable #Read table of self interactions
        self.Potential = Potential #Read particles potential values from snapshots
        # Centering
        self.DMRCM = DMRCM # Center by 200 central recursive
        self.M1 = M1 # Recursive Center of shells
        self.M2 = M2 # Center by N>200 particles with highest local DM densities (works best, but only if SPH denisty values are saved from the simulation and present in the snapshot files. Requires self.DMRho = True)
        self.M3 = M3 # CM of entire halo
        if(self.DMRho == False):
            self.M2 = False
            self.M3 = True
            print("No local density values. Swtching to centering by halo's center of mass.")
        # Options
        self.Plot = Plot # Plot resultss on the fly
        self.write = write # Top-level switch for writing resultss into a csv file
        self.time_j = 0 # for looping through table of interactions
        self.e = e # Number of snapshots to analyse

    @dataclass
    class Interaction_table:
        """ Information to be read from the interaction table """
        dr: List[float] = field(default_factory=list)
        time: List[float] = field(default_factory=list)
        xi: List[float] = field(default_factory=list)
        yi: List[float] = field(default_factory=list)
        zi: List[float] = field(default_factory=list)

    @dataclass
    class Particle_data:
        """ Information to be read from the particles data in the snapshots """
        radpos: List[float] = field(default_factory=list)
        Velr: List[float] = field(default_factory=list)
        Velm: List[float] = field(default_factory=list)
        velx: List[float] = field(default_factory=list)
        vely: List[float] = field(default_factory=list)
        velz: List[float] = field(default_factory=list)
        NSI: List[float] = field(default_factory=list)
        DMR: List[float] = field(default_factory=list)
        hs: List[float] = field(default_factory=list)
        cm: List[float] = field(default_factory=list)
        pot: List[float] = field(default_factory=list)

    @dataclass
    class Gtcurve:
        """ Gravothermal solution from Outmezguine et al. 2022 for comparison. """
        rho_c: List[int] = field(default_factory=list)
        t_c: List[int] = field(default_factory=list)

    @dataclass
    class Snap:
        """Dataclass for keeping calculation results from a snapshot."""
        time_j: int = 0
        Nr: float = 0
        Nrc: float = 0
        corefit: float = 0
        denfit: float = 0
        energy: float = 0
        kinetic: float = 0
        potential: float = 0
        NSIcore = float = 0
        NSItotal = float = 0

        number_of_core_particles: int = 0
        rho_c_est: float = 0
        core_dx_r: float = 0
        vc0: float = 0
        core_energy: float = 0
        core_kinetic: float = 0
        core_potential: float = 0

    @dataclass
    class Bin_values:
        """ Binned data for a given snapshot """
        x: List[float] = field(default_factory=list) # Bin position form center
        density: List[float] = field(default_factory=list) # Binned density
        sigv: List[float] = field(default_factory=list) # Binned velocity dipersion
        sigma_den: List[float] = field(default_factory=list) # Poisson noise in binned density (used for fitting a core profile in the final calculation for central denisty)

    @dataclass
    class Pars():
        """ Dataclass holding general simulation parameters """
        name: List[str] = field(default_factory=list) # Simulation name
        Dir: List[str] = field(default_factory=list)    # Full path to simulation Dir
        snaps: List[str] = field(default_factory=list)  # Full list of paths to individual snapshots
        save_case_name: List[str] = field(default_factory=list)
        m: float = 0 # Particles mass
        a: float = 0 # Inner boundary for binning
        b: float = 0 # Outer boundary for binning
        n: int = 0   # Nuber of bins
        w: float = 0    # Velocity scale "Omega" for velocity dependent cross section. Choose w = 0 for constant cross sections.
        dt: float = 0   # Time between each simulation snapshot
        sigma: float = 0    # Interaction cross section normalization constant
        rho_s: float = 0    # Halo's characteristic NFW density
        rs: float = 0       # halo's characteristic NFW radius
        vmax: float = 0     #Vmax of halo
        G : float = 4.3*10**(-6)    #Gravitational constant
        number_of_snaps: int = 0    #Number of snapshots for this simulation
        Npart: int = 0              #Number of particles in the halo
    
    def initialize(self,name,a,b,n,sigma,w,dt,rho_s,rs): # should be a subclass of analyze
        """ This method initializes several dataclasses: pars, gt, results
        and populates the first two """
        pars = self.make_parameters(name,a,b,n,sigma,w,dt,rho_s,rs)
        results = self.make_results(pars)
        gt = self.initialize_gt()
        return pars, results, gt

    # def initialize_results(self,name,a,b,n,sigma,w,dt,rho_s,rs):
    #     """ Initialize the necessary dataclasses: simulation paremers and resultss arrays """
    #     pars = self.make_parameters(name,a,b,n,sigma,w,dt,rho_s,rs)
    #     results = self.make_results(pars)
    #     return pars, results

    def initialize_gt(self):
        """ Load the gravothermal curve """
        return self.GT()

    def make_parameters(self,case,a,b,n,sigma,w,dt,rho_s,rs):
        pars = self.Pars()
        pars.name = case
        pars.Dir = "/lustre/projects/palubski-group/"+case+"out"
        pars.snaps = sorted(glob.glob(os.path.join(pars.Dir, 'snap*')))
        pars.save_case_name="halo_data/"+pars.name+"_n100_M2_isofitE_test.txt"
        pars.a = a
        pars.b = b
        pars.n = n
        pars.sigma = sigma
        pars.w = w
        pars.dt = dt
        pars.rho_s = rho_s
        pars.rs = rs
        pars.vmax = 21 #1.64*rs*np.sqrt(2*4.3e-6*rho_s) #Vmax of a NFW halo
        pars.number_of_snaps = len(pars.snaps)

        return pars

    def GT(self):
        """ This is the gravothermal solution for central density evolution in a SIDM halo with a constant interaction cross section (from Outmezguine et al. 2022) """
        file = open('/lustre/home/ipalubski/t_vs_rhoc_units_n0.csv') # C = 0.6
        file2= open('/lustre/home/ipalubski/rhoc_vs_t_C_0_753.csv') # C = 0.753
        csvreader = csv.reader(file)
        csvreader2 = csv.reader(file2)
        header = []
        header = next(csvreader)
        header2 = []
        header2 = next(csvreader2)
        rho_c06 = []
        tan06 = []

        rsan = 3
        rhosan = 2e7
        vm = 45.9
        rhoc0 = 2.4*rhosan
        rc0 = 0.45*rsan
        vc0 = 0.64*vm
        sigma = 5
        w = 10**4
        
        for row in csvreader:
            tan06.append(float(row[0][:]))
            rho_c06.append(float(row[1][:])/rhosan)

        def t_c0(sigmam_0,rhoc0,vc0,w,K):
            a = 4./np.sqrt(np.pi)
            C = 0.6
            t_c0 = (2/(3*a * C * (sigmam_0*ut.cm**2/ut.g)*(rhoc0*ut.Msun/ut.kpc**3)*(vc0*ut.km/ut.s)*K(vc0,w))).to_value('Gyr')
            return t_c0
        def t_c04(sigmam_0,rhoc0,vc0,w,K):
            a = 4./np.sqrt(np.pi)
            C = 0.47
            t_c0 = (2/(3*a * C * (sigmam_0*ut.cm**2/ut.g)*(rhoc0*ut.Msun/ut.kpc**3)*(vc0*ut.km/ut.s)*K(vc0,w))).to_value('Gyr')
            return t_c0

        def t_c02(sigmam_0,rhoc0,vc0,w,K):
            a = np.sqrt(16/np.pi)
            C = 0.6
            t_c0 = (1/(a * (sigmam_0*ut.cm**2/ut.g)*(rhoc0*ut.Msun/ut.kpc**3)*(vc0*ut.km/ut.s)*K(vc0,w))).to_value('Gyr')
            return t_c0
        def t_c03(sigmam_0,rhoc0,vc0,w,K):
            a = np.sqrt(16/np.pi)
            C = 0.8
            t_c0 = (2/(3*a * C * (sigmam_0*ut.cm**2/ut.g)*(rhoc0*ut.Msun/ut.kpc**3)*(vc0*ut.km/ut.s)*K(vc0,w))).to_value('Gyr')
            return t_c0

        def K3(v,w): 
            s = (v/w)
            return 1.5/(1.5**0.68 + 1/(np.log(1+(((s**2)+1e-4)*0.303941)**0.74)/(5.92*((s**2*s**2)+1e-8)))**0.68)**1.4706 
        #*(1.98*10**(33))*(3.17*10**(16))**(-3)*10**(-10)/(3.17*10**(-8-9))
        #print(tsan)
        tt06 = np.zeros(len(tan06))
        tt062=np.zeros(len(tan06))
        tt063=np.zeros(len(tan06))
        tt064=np.zeros(len(tan06))
        for i,t in enumerate(tan06):
            tt06[i] = t/t_c0(sigma,rhoc0,vc0,w,K3)#-14/t_c0(sigma,rhoc0,vc0,w,K3)
            tt062[i]= t/t_c0(sigma,rhoc0*2/2.4,vc0,w,K3)#-14/t_c0(sigma,rhoc0*2/2.4,vc0,w,K3)
            tt063[i]= t/t_c03(sigma,rhoc0*2/2.4,vc0,w,K3)
            tt064[i]= t/t_c04(sigma,rhoc0*2/2.4,vc0,w,K3)
        gt = self.Gtcurve()
        gt.rho_c = rho_c06
        gt.t_c = tt06
        return gt 

    def make_results(self,pars):
        results = [None]*len(pars.snaps)
        for i in range(len(pars.snaps)):
            temp = self.Snap()
            results[i] = temp
        return results

    def analyze(self,name,a,b,n,sigma,w,dt,rho_s,rs):
        pars, results, gt = self.initialize(name,a,b,n,sigma,w,dt,rho_s,rs)
        t = self.get_time(pars) # Create a list of snapshot times
        # 2d arrays for final outputs of halo profiles
        densnap = np.zeros([pars.number_of_snaps,n-1])  # <- density
        sigsnap = np.zeros([pars.number_of_snaps,n-1])  # <- velocity dispersion
        bin_values = self.Bin_values() 
        self.getx(pars,bin_values)
        if self.e == 1000:
            self.e = pars.number_of_snaps
        if self.NSItable:
            interaction_table = self.read_nsi(pars.name)
        for i in range(pars.number_of_snaps):
            #start = time.time()
            #bin_values = self.Bin_values() 
            particle_data = self.get_data(pars.snaps[i],pars)
            self.countblognp(particle_data,pars,bin_values)
            self.rhomax(particle_data,results[i],pars)
            self.fit_density(bin_values,results[i].rho_c_est,results[i],pars)
            if self.Potential:
                self.calc_energy(particle_data,pars,results[i])
                self.calc_core_energy(particle_data,bin_values.x,pars,results,i)
            if self.DMRho:
                self.central_dispersion(particle_data.Velr,results[i].number_of_core_particles,results[i])
            else:
                self.fit_dispersion(bin_values,results[i])
            if (i > 0):
                if self.NSItable:
                    si_time_bin = self.si_bin(t,interaction_table)
                    check_for_core_si(si_time_bin,interaction_table,pars,i,particle_data.cm)
                    self.predictNSIcore(results,i,t,pars,bin_values)
                if self.DMN:
                    self.predictNSI(bin_values,particle_data,t,pars,results,i)
            #print("core size = ",results[i].corefit,"_core density = ",results[i].denfit)
            #print(bin_values.sigma_den)
            #end = time.time()
            #print("{0}s for this snap".format(end - start))
            clear_output(wait=True)
            if self.Plot:
                #if (i == 0):
                    #fig, ax = plt.subplots(5,1,figsize=(6,12))
                self.plot(gt,results,bin_values,t,i,pars)
            if self.write:
                if (i == int((self.e - 1))): #If on last snapshot
                    self.write_results(pars,results,bin_values)
    
    def calc_energy(self,particle_data,pars,results):
        results.energy = np.sum(1/2*pars.m*particle_data.Velm**2) + np.sum(particle_data.pot*pars.m)/2
        results.kinetic = np.sum(1/2*pars.m*particle_data.Velm**2)
        results.potential = np.sum(particle_data.pot*pars.m)/2
    
    def predictNSI(self,bin_values,particle_data,t,pars,results,i):
        """ Determine the expected scattering rate based on the density and disperion profile of the halo. Works for constant and Yukawa scattering. 
            This function integrates the scattering rate in each bin to determine the expected number of scattering events. 
        """
        integrand = 4*np.pi*(bin_values.x**2)*bin_values.density*bin_values.density*bin_values.sigv/np.sqrt(3)*pars.sigma*4/np.sqrt(np.pi)/pars.m
        if pars.w > 0:
            """
                Assumes a Yukawa cross section. 
            """
            if(pars.w > 500):
                print("Increase integral limit!")
            def vsig_integrand(v,w,loc_disp):
                SIG = 4*np.pi*pars.sigma*(1/(1+pow(v/pars.w,2)+g*pow(v/pars.w,2)))
                return SIG * v**3 * math.exp(-pow(v,2)/(4*pow(loc_disp,2)))
            vsig = np.zeros(pars.n-1) 

            for j in range(pars.n-1):
                if bin_values.sigv[j] == 0:
                    vsig[j] = 0
                    continue
                vsig[j],_ = integrate.quad(vsig_integrand,0.1,500,args=(w,bin_values.sigv[j]),full_output=0)
            integrand = 4*np.pi*(bin_values.x**2)*den*den*(vsig/(2*pars.m*pow(bin_values.sigv,3)))
        """ Remove any Nan and infinities in empty bins """
        integrand[np.isnan(integrand)] = 0
        non0ind = integrand != 0
        integrand = integrand[non0ind]
        x = bin_values.x
        x = x[non0ind]
        """ Integrate the scattering rate """
        integrand_int = InterpolatedUnivariateSpline(x,integrand,k=1) #use k=1
        integrand2_int= InterpolatedUnivariateSpline(x,integrand,k=1)
        
        Units = 1/(3.086*10**19)**3 / 10 * 1.989*10**33 # Convert units 
        dt = (t[i] - t[i-1]) * 10**9 * 3.154*10**7  # from a to x[-1] below 
        NSIp = dt * integrand_int.integral(x[0], x[-1]) * Units #predicted number

        """ Take the ratio of the number of scattering events in the simulation to the expected/predicted number calculated here. """
        results[i].Nr = np.sum(particle_data.NSI)/NSIp
        
    def central_dispersion(self,Velr,numcore,results):
        vc0 = np.sqrt(3)*np.std(Velr[0:int(numcore)]) #3d dispersion, central
        results.vc0 = vc0
    
    def predictNSIcore(results,i,t,pars,bin_values):
        if (i == 0):
            results[i].Nrcsnap = 0.5
        if (i > 0):
            integrand = 4*np.pi*(bin_values.x**2)*den*den*bin_values.sigv*pars.sigma*4/np.sqrt(np.pi)/m
            integrand[np.isnan(integrand)] = 0
            non0ind = integrand != 0
            integrand = integrand[non0ind]
            x = bin_values.x[non0ind]
            integrand_int = InterpolatedUnivariateSpline(x,integrand,k=1) #use k=1
            Units = 1/(3.086*10**19)**3 / 10 * 1.989*10**33# * 1.989*10**(30)
            dt = (t[i] - t[i-1]) * 10**9 * 3.154*10**7
            # from a to x[-1] below
            #NSIpsnap[i] = dt * integrand_int.integral(x[0], x[-1]) * Units  #j is the index for bin i for snapshot  x[-1]
            results[i].NSIcp = dt * integrand_int.integral(x[0], results[i].corefit) * Units #rs/6.0
            if NSItable:
                #culmulative rate
                #Nrcsnap.append(np.sum(NSIcore.copy())/np.sum(NSIpsnapcore[0:i+1].copy()))

                #instantenous rate
                results[i].Nrc = results[i].NSIcore/results[i].NSIcp

                #average rate
                #results[i].Nrc = np.nanmean( results[0:i+1].NSIcore / results[0:i+1].NSIcp )

    def get_time(self,pars):
        n = pars.number_of_snaps
        a = pars.dt * n
        return np.linspace(0,a,n+1)

    def si_bin(t,interaction_table,results):
        bin_ind = np.searchsorted(t,interaction_table.time,side="right")
        assert(len(bin_ind)==len(interaction_table.time)) 
        NSItotal = np.zeros(len(t))
        
        for i in range(len(t)):
            x = interaction_table.xi[bin_ind==i]
            y = interaction_table.yi[bin_ind==i]
            z = interaction_table.zi[bin_ind==i]
            loc = np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))
            assert(len(loc)==len(x))
            #loc_resolved = loc[loc > rs]
            #loc_resolved = loc_resolved[loc_resolved < 2.3]
            results[i].NSItotal = int(len(bin_ind[bin_ind == i]))
            
            #NSItotal[i] = int(len(loc_resolved))  #only consider interactions that happen in the symmetric region
        #print("SUM in sibin2 = {}".format(np.sum(NSItotal)))
        return bin_ind #Return the time bin indices of the scattering events

    def check_for_core_si(bin_ind,Nsir,pars,i,cm):
        """ This routine calculates the average distance between particles scattering in the halo's core """
        x = interaction_table.xi[bin_ind==i]
        y = interaction_table.yi[bin_ind==i]
        z = interaction_table.zi[bin_ind==i]
        loc = interaction_table.dr[bin_ind==i] # Particles separation
        dx = np.array([(calc_core_ave(x[j],y[j],z[j],loc[j],cm,pars.rcore)) for j in range(0,len(x),1)]) # Returns particle separations for the scattering events that took place within the core radius.
        results[i].core_dx_r = np.nanmean(dx)/pars.rcore # The average of the above
        dx_non_nan = ~np.isnan(dx)
        results[i].NSIcore = len(dx[dx_non_nan]) # The number of scattering event that occured in the core between this and the last snapshot.

    def calc_core_ave(x,y,z,loc,cm,rcore):
    
        x = x - cm[0]
        y = y - cm[1]
        z = z - cm[2]
        r = np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))

        if (r < rcore):
            return loc # returns single value
        else:
            return float('Nan')

    def read_nsi(self,name):
        interaction_table = self.Interaction_table()
        Nsirfile = "/lustre/home/ipalubski/simulations"+name+"/NSI.txt"
        Nsir = pd.read_csv(Nsirfile, delim_whitespace= True, error_bad_lines=True, header=None, keep_default_na=True ,na_values=' ') #error_bad_lines=False ,sep='\t'
        Nsir = Nsir.sort_values(by=[1]) # Sort interaction table by time
        Nsir = Nsir.reset_index(drop=True)
        pd.to_numeric(Nsir[0],errors='coerce') #.notnull()
        interaction_table.dr = Nsir[0] #Location of interaction from halo center
        interaction_table.time = Nsir[1] #Time of interaction
        interaction_table.xi = Nsir[2] #x position
        interaction_table.yi = Nsir[3] #y position
        interaction_table.zi = Nsir[4] #z position
        return interaction_table

    def calc_core_energy(self,particle_data,x,pars,results,i):
        core_part = particle_data.radpos[np.where(particle_data.radpos < results[i].corefit)]
        results[i].number_of_core_particles = len(core_part)
        results[i].core_energy = 1/2 * pars.m * np.sum(particle_data.Velm[0:len(core_part)]**2) + np.sum(particle_data.pot[0:len(core_part)]*pars.m/2)
        results[i].core_potential = np.sum(particle_data.pot[0:len(core_part)]*pars.m/2) 
        results[i].core_kinetic = 1/2 * pars.m * np.sum(particle_data.Velm[0:len(core_part)]**2)

    def fit_dispersion(self,bin_values,results):
        sigbin = bin_values.sigv
        den = bin_values.density
        x = bin_values.x
        sigbin0 = sigbin[np.nonzero(sigbin)]
        den0 = den[np.nonzero(den)]
        x0 = x[np.nonzero(sigbin)]
        core_idx = (np.abs(x0 - results.corefit)).argmin()
        vc0 = np.mean(sigbin0[0:core_idx])   
        results.vc0 = vc0
    
    def fit_func(self,x,rho0,r0):
        """ Core profile used for fitting the halo's central regions to determine the central density. """
        return rho0/(1+(x/r0)**2)**(3/2)

    def fit_density(self,bin_values,rho_c,results,pars):
        """ Determine the central density the core size by fitting a core profile: \rho =  """
        #den = bin_values.density
        #den0 = den[np.nonzero(den)]
        #x0 = bin_values.x[np.nonzero(den)]
        
        den0 = bin_values.density[np.nonzero(bin_values.density)]
        x0 = bin_values.x[np.nonzero(bin_values.density)]

        sigma_den0=np.array(bin_values.sigma_den)[np.nonzero(bin_values.density)]
        
        for i in range(len(den0)):
            j = len(den0) - i - 1
            if(den0[j] > pars.rho_s): # try rho_s / 2 or so
                xmax_ind = j
                break

        fitpar, _ = curve_fit(self.fit_func, x0[0:xmax_ind], den0[0:xmax_ind],bounds=([1*10**6,0.01],[10**12,50.0]),method='trf',sigma=sigma_den0[0:xmax_ind],absolute_sigma= True)
        #prof_from_fit = self.fit_func(x0[0:xmax_ind],fitpar[0],fitpar[1])

        results.corefit = fitpar[1]
        results.denfit = fitpar[0]

    def rhomax(self,particle_data,results,pars):
        """ Choose the number of particles to be used for our initial guess for the central density. 
        Generally 200 particles are used unless the mass reoslution is small in which case 50 particles are used.
        """
        if(pars.Npart == 500000):
            results.rho_c_est=np.mean(particle_data.DMR[0:200])
        elif(pars.Npart == 1000000):
            results.rho_c_est=np.mean(particle_data.DMR[0:200])
        elif(pars.Npart == 30000):
            results.rho_c_est=np.mean(particle_data.DMR[0:50])

    def getx(self,pars,bin_values):
        """ Create the radial bins using specified values of a,b,n """
        log_bins=np.logspace(np.log10(pars.a),np.log10(pars.b),pars.n)
        x = np.zeros(pars.n-1)
        for i in range(pars.n-1):
            x[i] = (log_bins[i]+log_bins[i+1])/2
        bin_values.x = x

    def countblognp(self,particle_data,pars,bin_values):
        """ Bin particle's position and velocity data. The former is used to calculate the density profile, while the latter is used for the velocity dispersion profile."""
        log_bins= np.logspace(np.log10(pars.a),np.log10(pars.b),pars.n)
        bin_ind = np.searchsorted(log_bins,particle_data.radpos,side="right")
        bin_values.sigma_den = [] #Zero out the noise data between the snapshots
        #velx_asarray = particle_data.velx
        #vely_asarray = particle_data.vely
        #velz_asarray = particle_data.velz
        
        #sig = np.zeros([pars.n-1])
        #den = np.zeros([pars.n-1])
        
        den = np.array([(self.calc_density(particle_data.radpos[bin_ind==i],i,log_bins,bin_values,pars)) for i in range(1,pars.n,1)])
        sig = np.array([(self.calc_dispersion(particle_data.velx[bin_ind==i],particle_data.vely[bin_ind==i],particle_data.velz[bin_ind==i])) for i in range(1,pars.n,1)])
        
        bin_values.density = den
        bin_values.sigv = sig
        
    def calc_density(self,r_i,i,log_bins,bin_values,pars):
        v=(4*np.pi)*((log_bins[i])**3-(log_bins[i-1])**3)/3 #volume of shell
        if(len(r_i) != 1):
            bin_values.sigma_den.append(2/np.sqrt(len(r_i))*len(r_i)*pars.m/v)
        if(len(r_i) == 1):
            bin_values.sigma_den.append(10000*len(r_i)*pars.m/v)
            return 0
        return len(r_i)*pars.m/v

    def calc_dispersion(self,vel_x,vel_y,vel_z):
        avex = np.mean(vel_x)
        avey = np.mean(vel_y)
        avez = np.mean(vel_z)
        vel_x = vel_x - avex
        vel_y = vel_y - avey
        vel_z = vel_z - avez
        vmag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        sig = np.std(np.array(vmag)) # 3d dispersion

        sig = np.sqrt(np.sum(vmag**2/(len(vmag)-1)/3))

        if (sig != sig):
            sig = 0
        return sig

    def get_data(self,curr_snap_path,pars):
        """ This fetches all the data from simulation snapshots and loads it into a dataframe: particle_data. """
        f = curr_snap_path
        h5 = h5py.File(f,"r",driver = None)
        dset=h5['PartType1']
        pars.Npart = len(h5['PartType1']['Masses'])
        if self.DMN:
            nsi = dset['NInteractions'][:]
        if self.DMRho:
            dmrho = dset['LocalDMDensity'][:]*1e10 #Convert to units of solar masses
        if self.Potential:
            pot = dset['Potential'][:]
        pos = dset['Coordinates'][:]
        mass = (dset['Masses'][:])*1e10
        pars.m = mass[0]
        if self.Geth:
            h = dset['AGS-Softening'][:]
        vel = dset['Velocities'][:]
        velx=np.array([vi[0] for vi in vel])
        vely=np.array([vi[1] for vi in vel])
        velz=np.array([vi[2] for vi in vel])
        h5.close()
        dset = 0; h5=0

        radpos_temp, tpos_sort, cm = self.getpos(pos,pars.m,pars.Npart,dmrho,f)
        VelRad, VelM = self.getVelM(radpos_temp,tpos_sort,vel)
            
        df=pd.DataFrame({'Position':radpos_temp.copy(),'Velocity':VelM.copy(),"Velrad":VelRad.copy(),
                     "velx":velx.copy(),"vely":vely.copy(),"velz":velz.copy()})#,"DMRho":dmrho.copy()})#,'Tpos':tpos})  
        if self.DMRho:
            df2= pd.DataFrame({"DMRho":dmrho.copy()})#,'NSI01':nsi01.copy()})
            df = pd.concat([df,df2],axis=1)
        if self.DMN:
            df2= pd.DataFrame({'NSI':nsi.copy()})#,'NSI01':nsi01.copy()})
            df = pd.concat([df,df2],axis=1)
        if self.Geth:
            df = pd.concat([df,pd.DataFrame({'AGS-Softening':h})],axis = 1)
        if self.Potential:
            df = pd.concat([df,pd.DataFrame({'Potential':pot})],axis = 1)
        
        if self.Strip:
            delete = []
            for i in range(len(radpos_temp)):
                if (radpos_temp[i]<a) or (radpos_temp[i]>b):
                    delete.append(i)
            df=df.drop(delete,axis=0)
        
        df=df.sort_values(by=['Position'])
        radpos=df['Position'].to_numpy(copy=True)
        
        if self.DMRho:
            DMR = df['DMRho'].to_list()
        if self.DMN:
            NSI = df['NSI'].to_numpy(copy=True,dtype=int)
            
        Velm = df['Velocity'].to_numpy(copy=True,dtype=float)
        Velr = df['Velrad'].to_numpy(copy=True,dtype=float)
        velx = df['velx'].to_numpy(copy=True,dtype=float)
        vely = df['vely'].to_numpy(copy=True,dtype=float)
        velz = df['velz'].to_numpy(copy=True,dtype=float)

        if self.Potential:
            pot = df['Potential'].to_numpy(copy=True,dtype=float)
        if self.Geth:
            hs = df['AGS-Softening'].to_numpy(copy=True,dtype=float)
            
        df2= pd.DataFrame()
        df= pd.DataFrame()

        particle_data=self.Particle_data()
        particle_data.radpos = radpos
        particle_data.Velr = Velr
        particle_data.Velm = Velm
        particle_data.velx = velx
        particle_data.vely = vely
        particle_data.velz = velz
        particle_data.NSI = NSI
        particle_data.DMR = DMR
        particle_data.hs = hs
        particle_data.cm = cm
        particle_data.pot = pot
        return particle_data

    def getpos(self,pos,m,Npart,dmrho,snap):
        posx=[]
        posy=[]
        posz=[]
        posxcm=[]
        posycm=[]
        poszcm=[]

        def center_particles(cm,pos_sort):
            tpos=[]
            tpos_sort = np.subtract(pos_sort,cm)
            assert(tpos_sort[0,0]==(pos_sort[0,0]-cm[0]))
            assert(tpos_sort[50,1]==(pos_sort[50,1]-cm[1]))
            assert(tpos_sort[500,2]==(pos_sort[500,2]-cm[2]))
            radpos = np.sqrt(tpos_sort[:,0]*tpos_sort[:,0]+tpos_sort[:,1]*tpos_sort[:,1]+tpos_sort[:,2]*tpos_sort[:,2])
            return radpos, tpos_sort

        if self.DMRCM:
            convergence = 2.0
            count = 0
            cm = [0, 0, 0]
            radpos = get_radpos(cm)
            df2 = pd.DataFrame({"radpos":radpos,"x":posx,"y":posy,"z":posz,"DMRho":dmrho.copy()})
            df2=df2.sort_values(by=['radpos'])

            cm = self.get_center(df2,Npart)
            cmtotal = cm
            tpos, radpos, tposx, tposy, tposz = center_particles(cm,pos)
            df2 = pd.DataFrame()

            while convergence > 1.1:
                df2 = pd.DataFrame({"radpos":radpos,"x":tposx,"y":tposy,"z":tposz,"DMRho":dmrho.copy()})
                #df2=df2.sort_values(by=['DMRho'])
                count += 1
                cm_last = cm
                cm = self.get_center(df2,Npart)
                cmtotal = [cmtotal[0]+cm[0],cmtotal[1]+cm[1],cmtotal[2]+cm[2]]
                tpos, radpos, tposx, tposy, tposz = center_particles(cm,tpos)
                d = [cm[0],cm[1],cm[2]]
                print(convergence,count)
                convergence = np.sqrt(np.dot(d,d))/np.sqrt(np.dot(cm_last,cm_last))
                df2 = pd.DataFrame()
            mass = m*np.ones(Npart)
            cmx2=np.sum(posxcm*mass[0:len(posxcm)]/sum(mass[0:len(posxcm)]))
            cmy2=np.sum(posycm*mass[0:len(posxcm)]/sum(mass[0:len(posxcm)]))
            cmz2=np.sum(poszcm*mass[0:len(posxcm)]/sum(mass[0:len(posxcm)]))
            cm=[cmx2,cmy2,cmz2]

        elif(self.M2):
            if(Npart == 500000):
                part_cent = 3333
            if(Npart == 30000):
                part_cent = 300
            if(Npart == 1000000):
                part_cent = 4500

            posx=[];posy=[];posz=[]
            for x,y,z in pos:
                posx.append(x)
                posy.append(y)
                posz.append(z)

            df3 = pd.DataFrame({"x":posx,"y":posy,"z":posz,"DMRho":dmrho.copy()})
            df3 = df3.sort_values(by=['DMRho'])

            x = df3['x'].to_list()
            y = df3['y'].to_list()
            z = df3['z'].to_list()
            # Center of mass of part_cent densest particles
            cmx=np.sum(x[-part_cent:])/(part_cent)
            cmy=np.sum(y[-part_cent:])/(part_cent)
            cmz=np.sum(z[-part_cent:])/(part_cent)
            cm=[cmx,cmy,cmz]
            radpos, tpos_sort = center_particles(cm,pos)

        elif M1:
            def get_central_part(pos,r):
                posx=[];posy=[];poss=[]
                posxcm=[];posycm=[];poszcm=[]
                poscm=[]
                for x,y,z in pos:
                    posx.append(x)
                    posy.append(y)
                    posz.append(z)
                    #print(len(x),len(y),len(z))
                    r_i = np.sqrt(x**2 + y**2 + z**2)
                    #r_i = [np.sqrt(x**2 + y**2 + z**2) for x,y,z in pos]
                    #poscm = []
                    if (r_i < r):
                        posxcm.append(x)
                        posycm.append(y)
                        poszcm.append(z)
                        poscm.append([x,y,z])
                return poscm, posxcm, posycm, poszcm

            r = 30
            poscm, posxcm, posycm, poszcm = get_central_part(pos,r)
            #print(len(poscm[0]),len(poscm[1]),len(poscm[2]),len(poscm[3]))
            #print(len(pos[0]),len(pos[1]),len(pos[2]),len(pos[3]))
            #poscm = pos
            while r > 0.1*rs:
                poscm, posxcm, posycm, poszcm = get_central_part(poscm,r) 
                mass = m*np.ones(len(poscm))
                assert(len(poscm) == len(posxcm))
                cmx=np.sum(posxcm*mass/sum(mass))
                cmy=np.sum(posycm*mass/sum(mass))
                cmz=np.sum(poszcm*mass/sum(mass))
                cm=[cmx,cmy,cmz]
                poscm, radpos, tposx, tposy, tposz = center_particles(cm,poscm)
                #print(len(poscm[0]))
                #tpos = get_central_part(tpos,r)
                r = r * 0.6
            #print(len(pos[0]),len(poscm[0]))
            radpos, tpos_sort = center_particles(cm,pos)

        return radpos, tpos_sort, cm
    
    def getVelM(self,radpos,tpos_sort,vel):
        xhat = np.vstack((np.divide(tpos_sort[:,0].T,radpos),
                          np.divide(tpos_sort[:,1].T,radpos),
                          np.divide(tpos_sort[:,2].T,radpos)))
        VelRad = np.sum(vel.T*xhat,0)
        VelM = np.sqrt(np.sum(vel*vel,1))
        return VelRad, VelM

    def get_center(df2,Npart):
        mass = m*np.ones(Npart)
        part_cent = int(np.floor(Npart * 0.006667))
        df2=df2.sort_values(by=['radpos'])
        
        #xm = df2['x'].to_list() #recently commented out
        #ym = df2['y'].to_list()
        #zm = df2['z'].to_list()
        
        cmx=np.sum(xm[0:part_cent])/(part_cent)
        cmy=np.sum(ym[0:part_cent])/(part_cent)
        cmz=np.sum(zm[0:part_cent])/(part_cent)
        cm=[cmx,cmy,cmz]
        return cm

    def get_radpos(self,cmz):
        tposx=[]
        tposy=[]
        tposz=[]
        tpos=[]
        for x,y,z in pos:
            tposx.append(x-cm[0])
            tposy.append(y-cm[1])
            tposz.append(z-cm[2])
        for i in range(len(posx)):
            tpos.append([tposx[i],tposy[i],tposz[i]])
        radpos=[]
        for i in range(len(tposx)):
            radpos.append(np.sqrt(tposx[i]**2+tposy[i]**2+tposz[i]**2))
        return radpos

    def write_results(self,pars,results,bin_values):
        """ vc0, number_of_core_particles, core_energy_snap, core_potential_snap ,core_kinetic_snap, \
        rcore_snap, dx_r_snap, rho200snap, rho200ccmsnap, denfit, corefitsnap, Nrsnap, \
        results.energy, rho300 """
        data_for_writing = np.zeros((14, pars.number_of_snaps))
        header2 = ""
        for k in range(pars.number_of_snaps): #loop over each snapshot
            for field in fields(results[k]): #loop over data types to write
                field_name = field.name
                print(field.name)
                deader2=header2+field.name+"\t"
                data_for_writing[:,k] = getattr(results[k], field_name)

        save_case_den="halo_data/"+pars.name+"_n100_M2_den.txt"
        save_case_sig="halo_data/"+pars.name+"_n100_M2_sig.txt"

        np.savetxt(save_case_den, np.matrix.round(bin_values.density,decimals=3))
        np.savetxt(save_case_sig, np.matrix.round(bin_values.sigv,decimals=3))

        # if NSItable:
        # full_data_array = np.stack([core_part_number_snap,vc0,core_energy_snap,
        #                             core_potential_snap,core_kinetic_snap,
        #                             rcore_snap,dx_r_snap,
        #                             rho200snap,rho200ccmsnap,
        #                             denfit,corefitsnap,
        #                             Nrsnap,
        #                             results.energy,rho300                                
        #                             ], axis=1)
        # else:
        #     full_data_array = np.stack([core_part_number_snap,vc0,
        #                                 core_energy_snap,
        #                                 core_potential_snap,core_kinetic_snap,
        #                                 rcore_snap,dx_r_snap,
        #                                 rho200snap,rho200ccmsnap,
        #                                 denfit,corefitsnap,
        #                                 Nrsnap,results.energy,
        #                                 rho300,NSItotal2
        #                                 ], axis=1)
        full_data_array = np.round(data_for_writing,decimals=3)
        np.savetxt(pars.save_case_name, full_data_array, delimiter="\t",
                    header=header2,
#                   "Corepartnum\tvc03d\t\
#Ecore\tEpotential\t\
#Ekinetic\trcore\t\
#dx_to_r\tden200\t\
#den200ccm\tden_fit\t\
#core_fit\tNr\tEnergy\tp300\tNSI",
                        #header="Energy\t",
                        comments='')

    def plot(self,gt,results,bin_values,t,i,pars):
        def get_t0(sigmam_0,rhos,rs,w):
            G = 4.3*10**(-6) 
            v0 = np.sqrt(4*np.pi*G*rhos*rs**2) # km/s
            a = 4./np.sqrt(np.pi)

            t_c0 = (1/(a*(rhos*ut.Msun/ut.kpc**3)*(v0*ut.km/ut.s)*(sigmam_0*ut.cm**2/ut.g))).to_value('Gyr')
            return t_c0

        ts = 1/get_t0(pars.sigma,pars.rho_s,pars.rs,w=10**4)

        x = bin_values.x

        density = np.zeros(pars.number_of_snaps)
        density_est = np.zeros(pars.number_of_snaps)
        energy = np.zeros(pars.number_of_snaps)
        dispersion = np.zeros(pars.number_of_snaps)
        rate = np.zeros(pars.number_of_snaps)

        for k in range(i): #loop over each snapshot so far
            density[k] = results[k].denfit
            density_est[k] = results[k].rho_c_est
            energy[k] = results[k].energy
            dispersion[k] = results[k].vc0
            rate[k] = results[k].Nr

        fig, ax = plt.subplots(5,1,figsize=(6,14))
        fig.tight_layout()
        ax[0].plot(gt.t_c[0:25000],gt.rho_c[0:25000],color='gray') # Central density
        if self.DMRho:
            #ax[0].scatter(t[i]*ts,results[i].rho_c_est/pars.rho_s,color="red",s=5)
        #ax[0].scatter(t[i]*ts,results[i].denfit/pars.rho_s,color="blue",s=5)
            ax[0].plot(t[0:i]*ts,density_est[0:i]/pars.rho_s,color="red")
        ax[0].plot(t[0:i]*ts,density[0:i]/pars.rho_s,color="blue")

        ax[0].set_yscale("log")
        ax[0].set_ylim(1,1000)
        ax[0].set_ylabel(r"$\rho / \rho_s$")
        ax[0].set_xlabel(r"$t / t_0$")
        
        #ax[1].plot(density[0:i+1])/rho_s,energy[0:i+1]/energy[0])
        ax[1].plot(t[0:i]*ts,energy[0:i]/energy[0])
        ax[1].set_ylabel(r"$E / E_0$")
        ax[1].set_xlabel(r"$t / t_0$")

        #ax[1].set_ylim(0,10**9)
        prof_from_fit = self.fit_func(x,results[i].denfit,results[i].corefit)
        den0 = bin_values.density[np.nonzero(bin_values.density)]
        x0 = x[np.nonzero(bin_values.density)]
        ax[4].plot(x0,den0,color="black")
        ax[4].plot(x,prof_from_fit)
        ax[4].set_xlim(0.01,27)
        ax[4].set_ylim(1e4,5e10)
        ax[4].set_yscale("log")
        ax[4].set_xscale("log")
        ax[4].set_ylabel(r"$\rho (M_{\odot}/kpc^3)$")
        ax[4].set_xlabel(r"$r$ (kpc)")

        ax[3].plot(t[0:i]*ts,dispersion[0:i]/pars.vmax/np.sqrt(3),color='red')
        #ax[3].set_xlim(0,500)
        ax[3].set_ylabel(r"$v_{c,0}$")
        ax[3].set_xlabel(r"$t / t_0$")

        ax[2].plot(t[0:i]*ts,rate[0:i],color='red')
        #ax[4].set_xlim(0,500)
        ax[2].set_ylabel(r"$N_{SI}/N_{exp}$")
        ax[2].set_xlabel(r"$t / t_0$")

        plt.show()
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        

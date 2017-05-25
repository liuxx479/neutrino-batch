###### generate parameter files for camb, ICs, gadget, and slurm
from scipy import *
from scipy import optimize
import os
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
import sys

machine = ['perseus','KNL','stampede1'][int(sys.argv[1])]
plane_thickness = 512/3.0###128 Mpc/h

if machine =='KNL':
    main_dir = '/work/02977/jialiu/neutrino-batch/'
    temp_dir = '/scratch/02977/jialiu/temp/'
    NgenIC_loc = '/work/02977/jialiu/PipelineJL/S-GenIC/N-GenIC'
    Gadget_loc = '/work/02977/jialiu/PipelineJL/Gadget-2.0.7/Gadget2/Gadget2_massive'
    mpicc = 'ibrun'
    Ncore, nnodes = 40, 17
    extracomments ='''#SBATCH -A TG-AST140041
#SBATCH -p normal

module load fftw2
module load gsl'''

elif machine =='stampede1':
    main_dir = '/work/02977/jialiu/neutrino-batch/'
    temp_dir = '/scratch/02977/jialiu/temp/'
    NgenIC_loc = '/work/02977/jialiu/PipelineJL/S-GenIC/N-GenIC'
    Gadget_loc = '/work/02977/jialiu/PipelineJL/Gadget-2.0.7-stampede1/Gadget2/Gadget2_massive'
    mpicc = 'ibrun'
    Ncore, nnodes = 45, 16
    extracomments ='''#SBATCH -A TG-AST140041
#SBATCH -p normal

module load fftw2
module load gsl'''

elif machine =='perseus':
    main_dir = '/tigress/jialiu/neutrino-batch/'
    temp_dir = '/tigress/jialiu/temp/'
    NgenIC_loc = '/tigress/jialiu/PipelineJL/S-GenIC/N-GenIC'
    Gadget_loc = '/tigress/jialiu/PipelineJL/Gadget-2.0.7/Gadget2/Gadget2_massive'
    mpicc = 'srun'
    Ncore, nnodes = 25, 28
    extracomments='''module load openmpi
module load fftw'''

#########################
os.system('mkdir -p %sparams'%(main_dir))
os.system('mkdir -p %scamb'%(main_dir))
os.system('mkdir -p %sjobs'%(main_dir))
os.system('mkdir -p %slogs'%(main_dir))


h = 0.7
ombh2 = 0.0223

###Planck 2015 parameters
omega_m = 0.2880
A_s9 = 2.142
M_nu = 0.1
omb=ombh2/h**2
#### delta m21^2=7.37e-5
#### |delta m^2| = 2.5e-3 (normal) 2.46e-3 (inverted)
d31N = 2.5e-3
d31I = 2.46e-3
d21 = 7.37e-5

m2fcn = lambda m1: sqrt(d21 + m1**2)
m3_NH = lambda m1: sqrt(d31N + 0.5*m2fcn(m1)**2 +0.5*m1**2)
m3_IH = lambda m1: sqrt(0.5*m2fcn(m1)**2 +0.5*m1**2 - d31I)
root_NH = lambda m1, M: M-(m1+m2fcn(m1)+m3_NH(m1))
root_IH = lambda m1, M: M-(m1+m2fcn(m1)+m3_IH(m1))

Mmin_NH = sqrt(d21)+sqrt(d31N+d21/2)
Mmin_IH = sqrt(d31I-0.5*d21) + sqrt(d31I+0.5*d21)
m1min_IH = sqrt(d31I-0.5*d21)

hubble = 70
w = -1
ns = 0.97
pivot_scalar              = 0.05
pivot_tensor              = 0.05

def neutrino_mass_calc (M, split=1):
    '''split = 1, 2, 3 for normal, inverted, degenerate
    '''    
    #print M
    if M==0:
        m1,m2,m3=zeros(3)
    else:
        if split == 1:
            m1=optimize.bisect(root_NH, 0, M, args=(M,))       
            m2=m2fcn(m1)
            m3=m3_NH(m1)
        elif split == 2:
            m1=optimize.bisect(root_IH, m1min_IH, M, args=(M,))
            m2=m2fcn(m1)
            m3=m3_IH(m1)
        elif split ==3:
            m1, m2, m3 = ones(3)*M/3.0
    return m1,m2,m3

def Mnu2Omeganu(M_nu, omega_m):
    mnu_arr = array(neutrino_mass_calc(M_nu)) * u.eV
    cosmo = FlatLambdaCDM(H0=h*100, Om0=omega_m, m_nu=mnu_arr)
    return cosmo.Onu0
    
def camb_gen(M_nu, omega_m, A_s9):
    print 'generating CAMB parameter files'
    '''M_nu: total mass of neutrinos in unit of eV
    A_s9 = A_s * 1e9
    modify omch2, omnuh2, scalar_amp(1)
    '''
    if M_nu==0:
        omnuh2=0
    else:
        omnuh2 = Mnu2Omeganu(M_nu, omega_m)*h**2
    omch2 = omega_m*h**2 - omnuh2 - ombh2
    filename = 'camb_mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    m1, m2, m3 = neutrino_mass_calc (M_nu)
    
    paramtext='''#Parameters for CAMB

#output_root is prefixed to output file names
output_root = %scamb/%s

get_scalar_cls = F
get_vector_cls = F
get_tensor_cls = F
get_transfer   = T
do_lensing     = F
do_nonlinear = 0
l_max_scalar      = 8000
k_eta_max_scalar  = 16000
l_max_tensor      = 1500
k_eta_max_tensor  = 3000
use_physical   = T
########## cosmological parameters (Planck 2015 TT,TE,EE+lowP+lensing+ext)
ombh2          = 0.02230
omch2          = %.5f ##### = 0.1188 Planck2015
omnuh2         = %.5f ##### = Mnu2Omeganu(M_nu, omega_m)eV 
omk            = 0
hubble         = 70
w              = -1
cs2_lam        = 1
temp_cmb           = 2.7255
helium_fraction    = 0.24

massless_neutrinos = 0.046
nu_mass_eigenstates = 3
massive_neutrinos  = 1 1 1
share_delta_neff = T
nu_mass_fractions = %.5f %.5f %.5f
nu_mass_degeneracies = 

initial_power_num         = 1
pivot_scalar              = 0.05
pivot_tensor              = 0.05
scalar_amp(1)             = %.4fe-9 ##### = 2.142e-9 Planck2015
scalar_spectral_index(1)  = 0.97
scalar_nrun(1)            = 0
scalar_nrunrun(1)         = 0
tensor_spectral_index(1)  = 0
tensor_nrun(1)            = 0
tensor_parameterization   = 1
initial_ratio(1)          = 1

reionization         = F
re_use_optical_depth = T
re_optical_depth     = 0.09
re_redshift          = 11
re_delta_redshift    = 1.5
re_ionization_frac   = -1
re_helium_redshift = 3.5
re_helium_delta_redshift = 0.5
RECFAST_fudge = 1.14
RECFAST_fudge_He = 0.86
RECFAST_Heswitch = 6
RECFAST_Hswitch  = T

initial_condition   = 1
initial_vector = -1 0 0 0 0
vector_mode = 0
COBE_normalize = F
CMB_outputscale = 7.42835025e12 

transfer_high_precision = T
transfer_kmax           = 500 ## 1000 for columbia, 25 for S. Bird
transfer_k_per_logint   = 0
transfer_num_redshifts  = 7
transfer_interp_matterpower = T
transfer_redshift(1) = 99
transfer_filename(1) = transfer_99.dat
transfer_matterpower(1) = matterpow_99.dat
transfer_redshift(2) = 49.0
transfer_filename(2) = transfer_49.dat
transfer_matterpower(2) = matterpow_49.0.dat
transfer_redshift(3) = 10.0
transfer_filename(3) = transfer_10.dat
transfer_matterpower(3) = matterpow_10.dat
transfer_redshift(4) = 4.0
transfer_filename(4) = transfer_4.dat
transfer_matterpower(4) = matterpow_4.dat
transfer_redshift(5) = 2.0
transfer_filename(5) = transfer_2.dat
transfer_matterpower(5) = matterpow_2.dat
transfer_redshift(6) = 1.0
transfer_filename(6) = transfer_1.dat
transfer_matterpower(6) = matterpow_1.dat
transfer_redshift(7) = 0.0
transfer_filename(7) = transfer_0.dat
transfer_matterpower(7) = matterpow_0.dat

transfer_power_var = 7
scalar_output_file = scalCls.dat
vector_output_file = vecCls.dat
tensor_output_file = tensCls.dat
total_output_file  = totCls.dat
lensed_output_file = lensedCls.dat
lensed_total_output_file  =lensedtotCls.dat
lens_potential_output_file = lenspotentialCls.dat

do_lensing_bispectrum = F
do_primordial_bispectrum = F
bispectrum_nfields = 1
bispectrum_slice_base_L = 0
bispectrum_ndelta=3
bispectrum_delta(1)=0
bispectrum_delta(2)=2
bispectrum_delta(3)=4
bispectrum_do_fisher= F
bispectrum_fisher_noise=0
bispectrum_fisher_noise_pol=0
bispectrum_fisher_fwhm_arcmin=7
bispectrum_full_output_file=
bispectrum_full_output_sparse=F
bispectrum_export_alpha_beta=F

feedback_level = 1
output_file_headers = T
derived_parameters = T
lensing_method = 1
accurate_BB = F

massive_nu_approx = 1
accurate_polarization   = T
accurate_reionization   = T
do_tensor_neutrinos     = T
do_late_rad_truncation   = F
halofit_version= 7
number_of_threads       = 0
high_accuracy_default=T

accuracy_boost          = 3
l_accuracy_boost        = 3
l_sample_boost          = 3'''%(main_dir, filename, omch2, omnuh2, m1/M_nu, m2/M_nu, m3/M_nu, A_s9)
    
    f = open('params/%s.param'%(filename), 'w')
    f.write(paramtext)
    f.close()



def ngenic_gen(M_nu, omega_m, A_s9):
    print 'generating NGENIC parameter files'
    omnuh2 = Mnu2Omeganu(M_nu, omega_m)*h**2
    omch2 = omega_m*h**2 - omnuh2 - ombh2
    filename = 'ngenic_mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    fn_matter = '%scamb/camb_mnv%.5f_om%.5f_As%.4f_matterpow_99.dat'%(main_dir, M_nu, omega_m, A_s9)
    fn_transfer = '%scamb/camb_mnv%.5f_om%.5f_As%.4f_transfer_99.dat'%(main_dir, M_nu, omega_m, A_s9)
    cosmo = 'mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    os.system('mkdir -p %s%s/ICs'%(temp_dir, cosmo))
    
    paramtext='''#==Required parameters==
# This is the size of the FFT grid used to
# compute the displacement field. One
# should have Nmesh >= Npart.
Nmesh = 2048
# Random seed for modes in phase realisation
Seed = 10027	

# Periodic box size of simulation
Box = 512000

# Base-filename of output files
FileBase = ICs
# Directory for storing output files
OutputDir = %s%s/ICs

# Total matter density  (at z=0)
Omega = %.5f
# Cosmological constant (at z=0)
OmegaLambda = %.5f
# Baryon density        (at z=0)
OmegaBaryon = %.5f
# Hubble parameter (may be used for power spec parameterization)
HubbleParam = 0.7
# Starting redshift
Redshift = 99

#Number of files used in output snapshot set, for ICFormat < 4.
NumFiles = 28
# filename of tabulated MATTER powerspectrum from CAMB
FileWithInputSpectrum = %s
# filename of transfer functions from CAMB
FileWithTransfer = %s

#==Optional Parameters==
# (Cube root of) number of particles
NBaryon = 0
NCDM = 1024
NNeutrino = 0

#Particle mass of neutrinos in eV.
#OmegaNu is derived self-consistently from this value
#and hence not specified separately.
NU_PartMass_in_ev = %.5f
#1,0,-1 correspond to normal, degenerate and inverted neutrino species hierarchies, respectively.
#Note if you ask for a mass below the minimum allowed by the hierarchy, 
#you will get a single massive neutrino species.
Hierarchy = 1
#Output format of ICs: 1 and 2 correspond to Gadget format 1 and 2, 3 is HDF5 and 4 is BigFile.
ICFormat = 2

# Enable twolpt
# Note this is only formally derived for single-fluid (CDM) simulations.
TWOLPT = 0
#If 1, each mode will be scattered from the mean power to mimic cosmic variance
RayleighScatter = 0
#If true, do not include radiation (inc. massless neutrinos) 
#when computing the Hubble function (for the velocity prefactor)
NoRadiation = 0

#==Specialised optional parameters you are unlikely to use==
# "1" selects Eisenstein & Hu spectrum,
# "2" selects a tabulated power spectrum in
# the file 'FileWithInputSpectrum'
# otherwise, Efstathiou parametrization is used
WhichSpectrum = 2

# defines length unit of tabulated
# input spectrum in cm/h. By default 1 Mpc.
# Only used for CAMB power spectra.
InputSpectrum_UnitLength_in_cm = 3.085678e24

# if set to zero, the tabulated spectrum is
# assumed to be normalized already in its amplitude to
# the starting redshift, otherwise this is recomputed
# based on the specified sigma8
ReNormalizeInputSpectrum = 0

#Amplitude of matter fluctuations at z=0, used only if ReNormalizeInputSpectrum=1.
Sigma8 = 0.8
#ns - only used for non-tabulated power spectra.
PrimordialIndex = 1.

# defines length unit of output (in cm/h)
UnitLength_in_cm = 3.085678e21
# defines mass unit of output (in g/cm)
UnitMass_in_g = 1.989e43
# defines velocity unit of output (in cm/sec)
UnitVelocity_in_cm_per_s = 1e5

# If 1, the neutrino masses will be included in the dark matter particles,
# as a change in the transfer function. This is a very inaccurate way to simulate neutrinos.
NU_in_DM = 0
#If one, add neutrino thermal velocities to type 2 particles.
NU_Vtherm_On = 1

#Shape parameter, only for Efstathiou power spectrum
ShapeGamma = 0.201
    '''%(temp_dir, cosmo, omega_m, 1-omega_m, omb, fn_matter, fn_transfer, M_nu)
    f = open('params/%s.param'%(filename), 'w')
    f.write(paramtext)
    f.close()


def gadget_gen (M_nu, omega_m, A_s9):
    print 'generating GADGET parameter files'
    cosmo = 'mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    fn_ICs='%s%s/ICs/ICs'%(temp_dir, cosmo)
    fn_transfer = '%scamb/camb_mnv%.5f_om%.5f_As%.4f_transfer_99.dat'%(main_dir,M_nu, omega_m, A_s9)
    filename = 'gadget_mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    os.system('mkdir -p %s/snapshots'%(temp_dir+cosmo))
    m1, m2, m3 = neutrino_mass_calc(M_nu)
    
    paramtext='''InitCondFile		        %s
OutputDir		        %s/snapshots
EnergyFile			energy.txt
InfoFile			info.txt
TimingsFile			timings.txt
CpuFile			cpu.txt
RestartFile			restart
SnapshotFileBase			snapshot
OutputListFilename		%sparams/outputs_%s.txt

%%    cpu_timings

TimeLimitCPU		172800.0
ResubmitOn		0
ResubmitCommand		my-scriptfile


%%    code_options

ICFormat		2
SnapFormat		1
ComovingIntegrationOn		1
TypeOfTimestepCriterion		0
OutputListOn		1
PeriodicBoundariesOn		1


TimeBegin			0.01
%%    characteristics_of_run

TimeMax		1.0


Omega0			%.5f
OmegaLambda			%.5f
OmegaBaryon			%.5f
HubbleParam			0.7
BoxSize			512000.000000

%%    output_frequency

TimeBetSnapshot		0.5
TimeOfFirstSnapshot		0
CpuTimeBetRestartFile		1800.0
TimeBetStatistics		0.05
NumFilesPerSnapshot		28
NumFilesWrittenInParallel		28


%%     accuracy_time_integration

ErrTolIntAccuracy		0.025
MaxRMSDisplacementFac		0.2
CourantFac		0.15
MaxSizeTimestep		0.02
MinSizeTimestep		0.0


%%     tree_algorithm

ErrTolTheta		0.45
TypeOfOpeningCriterion		1
ErrTolForceAcc		0.005
TreeDomainUpdateFrequency		0.025


%%    sph

DesNumNgb		33
MaxNumNgbDeviation		2
ArtBulkViscConst		0.8
InitGasTemp		1000.0
MinGasTemp		50.0


%%     memory_allocation

PartAllocFactor		1.5
TreeAllocFactor		1.5
BufferSize		20.0


%%    system_of_units

UnitLength_in_cm		3.085678e+21
UnitMass_in_g		1.989e+43
UnitVelocity_in_cm_per_s		100000.0
GravityConstantInternal		0


%%    softening

MinGasHsmlFractional		0.25
SofteningGas		0
SofteningHalo		9.0
SofteningDisk		0
SofteningBulge		0
SofteningStars		0
SofteningBndry		0
SofteningGasMaxPhys		0
SofteningHaloMaxPhys		9.0
SofteningDiskMaxPhys		0
SofteningBulgeMaxPhys		0
SofteningStarsMaxPhys		0
SofteningBndryMaxPhys		0

%%     neutrinos

KspaceTransferFunction      %s ; File containing CAMB formatted output transfer functions.

TimeTransfer                0.01  ;     Scale factor at which the CAMB transfer functions were generated.
InputSpectrum_UnitLength_in_cm               3.085678e24  ; Units of the CAMB transfer function in cm. By default Mpc.
MNue                        %.5f   ;    Mass of the lightest neutrino in eV.
MNum                        %.5f   ;    Second neutrino mass in eV.
MNut                        %.5f   ;    Third neutrino mass. Note the observed mass splitting is not enforced.
Vcrit                       500    ;    Critical velocity in the Fermi-Dirac distribution below which the neutrinos
NuPartTime                  0.3333   ;    Scale factor at which to 'turn on', ie, make active gravitators,

HybridNeutrinosOn           0  ;       Whether hybrid neutrinos are enabled.
'''%(fn_ICs, temp_dir+cosmo, main_dir, cosmo, omega_m, 1.0-omega_m, omb, fn_transfer,  m1,m2,m3)
    f = open('params/%s.param'%(filename), 'w')
    f.write(paramtext)
    f.close()

###################################
######## parameter design #########
###################################
design = 0
if design:
    lhd_gridcm_3d=loadtxt('lhd_gridcm_3d.txt')
    lhd_gridcm = lhd_gridcm_3d[:,:-1]
    lhd_gaus_cm = stats.distributions.norm(loc=0, scale=1).ppf(lhd_gridcm)
    nu_rand_cm = stats.distributions.halfnorm(loc=0, scale=0.2).ppf(lhd_gridcm_3d[:,-1])+Mmin_NH
    fidu_om, fidu_As = 0.3, 2.1 # Omega_m, A_s*1e9
    #lhd_gaus_cm = norm(loc=0, scale=1).ppf(lhd_gridcm)
    lhd_gaus_cm_tramsformed = zeros(shape=lhd_gaus_cm.shape)
    lhd_gaus_cm_tramsformed.T[0] = lhd_gaus_cm.T[0]*fidu_om*0.15 + fidu_om
    lhd_gaus_cm_tramsformed.T[1] = lhd_gaus_cm.T[1]*fidu_As*0.15 + fidu_As

    params = array([nu_rand_cm, lhd_gaus_cm_tramsformed.T[0], lhd_gaus_cm_tramsformed.T[1]]).T
    params = append(params, [[0.1, 0.3, 2.1],[0.0, 0.3, 2.1]],axis=0)
    savetxt('params.txt',params)

params = loadtxt('params.txt')

from scipy.interpolate import interp1d
### out to z = 3.0, every 128 Mpc/h=182.857Mpc output, interpolation
z_arr = linspace(0,50.0,1001)

def outputs(iparams):
    M_nu, omega_m, A_s9 = iparams
    omnu = Mnu2Omeganu(M_nu, omega_m)
    #omch2 = omega_m*h**2 - omnu*h**2 - ombh2
    nu_masses = neutrino_mass_calc(M_nu) * u.eV
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, m_nu = nu_masses)
    DC_interp = interp1d(cosmo.comoving_distance(z_arr)/h, z_arr)
    DC_arr = arange(0, cosmo.comoving_distance(z_arr[-1]).value /h, plane_thickness/h)
    newz_arr = DC_interp(DC_arr)
    a_arr = 1.0/(1.0+newz_arr)
    cosmo = 'mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    fn = '%sparams/outputs_%s.txt'%(main_dir, cosmo)
    savetxt(fn, sort(a_arr))
    #cosmo.comoving_distance(newz_arr).value/h/DC_arr-1

def sbatch_camb():
    M_nu, omega_m, A_s9 = iparams
    fn='jobs/camb.sh'
    f = open(fn, 'w')
    scripttext='''#!/bin/bash 
#SBATCH -N 4 # node count 
#SBATCH -t 1:00:00 
#SBATCH --array=1-104
#SBATCH --ntasks=1
#SBATCH --output=/tigress/jialiu/neutrino-batch/logs/camb_%A_%a.out
#SBATCH --error=/tigress/jialiu/neutrino-batch/logs/camb_%A_%a.err
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end 
#SBATCH --mail-user=jia@astro.princeton.edu 

# Load openmpi environment
module load intel
/tigress/jialiu/PipelineJL/CAMB-Jan2017/camb $(ls /tigress/jialiu/neutrino-batch/params/camb* | sed -n ${SLURM_ARRAY_TASK_ID}p)
'''
    f.write(scripttext)
    f.close()

def sbatch_ngenic():
    for x in arange(1,101,10):
        y=x+9
        if y==100:
            y+=1
        fn = 'jobs/ngenic_%s_%s.sh'%(x,y)
        f = open(fn, 'w')
        scripttext='''#!/bin/bash 
#SBATCH -N 1 # node count 
#SBATCH --ntasks-per-node=28
#SBATCH -t 24:00:00 
#SBATCH --output=%slogs/ngenic_%i-%i_%%A.out
#SBATCH --error=%slogs/ngenic_%i-%i_%%A.err
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end 
#SBATCH --mail-user=jia@astro.princeton.edu 
#SBATCH --mem 110000

# Load openmpi environment
module load intel
module load fftw
module load hdf5
export CC=icc
export CXX=icpc

for i in {%i..%s}
do
echo $i
%s $(ls %sparams/ngen* | sed -n ${i}p) &
wait
done
'''%(main_dir, x,y,main_dir,x,y, x,y,NgenIC_loc,main_dir)
        f.write(scripttext)
        f.close()


def sbatch_gadget(iparams, N=Ncore, job='j'):
    M_nu, omega_m, A_s9 = iparams
    n=N*nnodes
    if machine=='perseus':
        job='A'
    filename = 'gadget_mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    scripttext='''#!/bin/bash 
#SBATCH -N %i # node count 
#SBATCH -n %i
#SBATCH -J Gadget_mnv%.3f
#SBATCH --ntasks-per-node=%i 
#SBATCH -t 48:00:00 
#SBATCH --output=%slogs/%s_%%%s.out
#SBATCH --error=%slogs/%s_%%%s.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jia@astro.princeton.edu 
%s
module load intel
module load hdf5

%s  %s %sparams/%s.param'''%(N, n, M_nu, nnodes, main_dir, filename, job, main_dir, filename, job, extracomments,  mpicc,  Gadget_loc, main_dir, filename)
    f = open('jobs/%s_%s.sh'%(filename,machine), 'w')
    f.write(scripttext)
    f.close()
    
#sbatch_camb()
#os.system('cp /tigress/jialiu/neutrino-batch/camb_mnv0.00000_om0.30000_As2.1000.param /tigress/jialiu/neutrino-batch/params')

#sbatch_ngenic()
for iparams in params:
    print iparams
    M_nu, omega_m, A_s9 = iparams
    #camb_gen(M_nu, omega_m, A_s9)
    #ngenic_gen(M_nu, omega_m, A_s9)
    #gadget_gen(M_nu, omega_m, A_s9)
    #outputs(iparams)
    #sbatch_gadget(iparams)

###### generate parameter files for camb, ICs, gadget, and slurm
from scipy import *

h = 0.7
ombh2 = 0.0223

###Planck 2015 parameters
omega_m = 0.7438
A_s9 = 2.142
M_nu = 0.1

def camb_gen(M_nu, omega_m, A_s9):
    '''M_nu: total mass of neutrinos in unit of eV
    A_s9 = A_s * 1e9
    modify omch2, omnuh2, scalar_amp(1)
    '''
    omnuh2 = M_nu / 93.14
    omch2 = omega_m*h**2 - omnuh2 - ombh2
    filename = 'camb_mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
    paramtext='''#Parameters for CAMB

#output_root is prefixed to output file names
output_root = ../camb/%s

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
omnuh2         = %.5f ##### = M_nu / 93.14eV 
omk            = 0
hubble         = 70
w              = -1
cs2_lam        = 1
temp_cmb           = 2.7255
helium_fraction    = 0.24

massless_neutrinos = 0.046
nu_mass_eigenstates = 1
massive_neutrinos  = 3
share_delta_neff = T
nu_mass_fractions = 1
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
transfer_filename(2) = transfer_49.0.dat
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

accuracy_boost          = 1
l_accuracy_boost        = 1
l_sample_boost          = 1'''%(filename, omch2, omnuh2, A_s9)
    
    f = open('params/%s.param'%(filename), 'w')
    f.write(paramtext)
    f.close()

#camb_gen(M_nu, omega_m, A_s9)


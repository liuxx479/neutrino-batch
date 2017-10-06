from scipy import *
import numpy as np
from pylab import *
from scipy.interpolate import interp1d
import os
from astropy.io import fits
from matplotlib import cm

plot_design = 0
plot_pmatter = 0
plot_pmatterall = 0
plot_pmatter_evolve = 0
plot_sample_maps_include_reconstructed = 0
plot_sample_maps = 1
plot_hmf = 0
plot_merger_tree = 0
plot_mass_accretion = 0

h=0.7
k_arr = logspace(-2,1.5,100)

if plot_design:
    ims = 5
    ifs=16
    ilw=2
    imk = 'o'
    Muarr, Omarr, Asarr, S8arr=genfromtxt('cosmo_params_MuOmAsS8.txt').T
    
    f=figure(figsize=(8,8))
    ax1=f.add_subplot(2,2,1)
    ax2=f.add_subplot(2,2,2)
    ax3=f.add_subplot(2,2,3)
    ax4=f.add_subplot(2,2,4)
        
    ax1.scatter(Muarr, Asarr, marker=imk, s=ims, color='k')
    ax2.scatter(Muarr, Omarr, marker=imk, s=ims, color='k')
    ax3.scatter(Omarr, Asarr, marker=imk, s=ims, color='k')
    ax4.scatter(S8arr, Asarr, marker=imk, s=ims, color='k')

    ax1.scatter([0.1, 0],    [2.1, 2.1], marker=imk, s=ims+5, color='r')
    ax2.scatter([0.1, 0],    [0.3, 0.3], marker=imk, s=ims+5, color='r')
    ax3.scatter(0.3,    2.1, marker=imk, s=ims+5, color='r')
    ax4.scatter(0.8295, 2.1, marker=imk, s=ims+5, color='r')
    
    ax1.set_xlabel(r'$M_\nu$',fontsize=ifs)
    ax1.set_ylabel(r'$10^9A_s$',fontsize=ifs)
    ax2.set_xlabel(r'$M_\nu$',fontsize=ifs)
    ax2.set_ylabel(r'$\Omega_m$',fontsize=ifs)
    ax3.set_xlabel(r'$\Omega_m$',fontsize=ifs)
    ax3.set_ylabel(r'$10^9A_s$',fontsize=ifs)
    ax4.set_ylabel(r'$10^9A_s$',fontsize=ifs)
    ax4.set_xlabel(r'$\sigma_8$',fontsize=ifs)
    ax1.set_xlim(-0.05, 0.7)
    ax2.set_xlim(-0.05, 0.7)
    ax2.set_ylim(0.15, 0.45)
    ax3.set_xlim(0.15, 0.45)
    ax1.set_xlim(-0.05, 0.7)
    for iax in (ax1,ax2,ax3,ax4):
        iax.grid(True)
        iax.locator_params(axis = 'both', nbins = 5)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.35, left=0.07, right=0.97,bottom=0.18)
    savefig('plots/plot_design.pdf')
    close()
    
    
if plot_pmatter:
    kcamb1, Pcamb1 = np.loadtxt('camb/camb_mnv0.10000_om0.30000_As2.1000_matterpow_0.dat').T
    kcamb0, Pcamb0 = np.loadtxt('camb/camb_mnv0.00000_om0.30000_As2.1000_matterpow_0.dat').T

    ####### bird halofit
    kbird1, Pbird1 = np.loadtxt('camb-fidu/camb_mnv0.10000_om0.30000_As2.1000-bird_matterpow_0.dat').T
    kbird0, Pbird0 = np.loadtxt('camb-fidu/camb_mnv0.00000_om0.30000_As2.1000-bird_matterpow_0.dat').T

    ####### takahashi halofit
    ktaka1, Ptaka1 = np.loadtxt('camb-fidu/camb_mnv0.10000_om0.30000_As2.1000-halofit_matterpow_0.dat').T
    ktaka0, Ptaka0 = np.loadtxt('camb-fidu/camb_mnv0.00000_om0.30000_As2.1000-halofit_matterpow_0.dat').T
    
    ####### sims
    knb0_1024, Pnb0_1024 = np.loadtxt('matterpower/mnv0.00000_om0.30000_As2.1000/powerspec_tot_067.txt').T
    knb1_1024, Pnb1_1024 = np.loadtxt('matterpower/mnv0.10000_om0.30000_As2.1000/powerspec_tot_067.txt').T
    knb0_1024 *= 1e3
    knb1_1024 *= 1e3
    Pnb0_1024 *= 1e-9
    Pnb1_1024 *= 1e-9
    ####### interpolation
    Pcamb1_interp = interp1d(kcamb1, Pcamb1)(k_arr)
    Pcamb0_interp = interp1d(kcamb0, Pcamb0)(k_arr)
    Pbird1_interp = interp1d(kbird1, Pbird1)(k_arr)
    Pbird0_interp = interp1d(kbird0, Pbird0)(k_arr)
    Ptaka1_interp = interp1d(ktaka1, Ptaka1)(k_arr)
    Ptaka0_interp = interp1d(ktaka0, Ptaka0)(k_arr)

    ############  plots
    f=figure(figsize=(8,6))
    ax1=f.add_subplot(211)
    ax2=f.add_subplot(212)

    ax1.plot(kcamb0, Pcamb0, 'k--',lw=2.5, label=r'${\rm Linear\; Theory}\; (M_\nu = 0\; {\rm eV})$'  )
    ax1.plot(kcamb1, Pcamb1, 'k-', lw=1.5, label=r'${\rm Linear\; Theory}\; (M_\nu = 0.1\; {\rm eV})$')

    ax1.plot(knb0_1024, Pnb0_1024, 'g--', lw=2.5,label=r'${\rm Simulation}\; (M_\nu = 0\; {\rm eV})$'  )
    ax1.plot(knb1_1024, Pnb1_1024, 'g-',  lw=1.5,label=r'${\rm Simulation}\; (M_\nu = 0.1\; {\rm eV})$')

    #ax.plot((0.2,0.2), (0.1,1e5),'k--')
    ax1.set_ylabel('$P_{mm}(k)$',fontsize=18)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(0.1, 1e5)
    ax1.legend(frameon=0,loc=3,fontsize=12)
    #plt.yticks(fontsize=16)
    #ax1.set_xticklabels(fontsize=16)
    ax1.plot([0.2,0.2], [1e-2,1e5],'k--',lw=1)
    
    ax2.plot(k_arr,Pcamb1_interp/Pcamb0_interp-1,'k-',lw=2, label=r'${\rm Linear\; Theory}$')
    ax2.plot(k_arr,Pbird1_interp/Pbird0_interp-1,'--',color='orange',lw=2.0, label=r'${\rm Halofit\; (Bird2012)}$')
    ax2.plot(k_arr,Ptaka1_interp/Ptaka0_interp-1,'--',color='purple',lw=1.0, label=r'${\rm Halofit\; (Smith2003+Takahashi2012)}$')
    ax2.plot(knb0_1024,Pnb1_1024/Pnb0_1024-1,'g-',lw=1, label=r'${\rm Simulation}$')
    
    ax2.legend(frameon=0,fontsize=12,ncol=1,loc=0)
    ax2.set_xscale('log')
    ax2.set_xlim(1e-2, 10)
    ax2.set_ylim(-0.09,0.0)
    ax2.set_xlabel(r'$k\; (h/{\rm Mpc})$',fontsize=18)
    ax2.set_ylabel(r'$P_{mm}^{0.1eV} / P_{mm}^{0 eV} - 1$',fontsize=18)
    ax2.plot([0.2,0.2], [-0.5,0],'k--',lw=1)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.09,left=0.15)
    #ax1.locator_params(axis = 'y', nbins = 5)
    ax2.locator_params(axis = 'y', nbins = 5)
    plt.tight_layout()
    #ax1.grid(True)
    #ax2.grid(True)
    savefig('plots/plot_pmatter_fidu.pdf')
    close()

if plot_pmatterall:
    kcamb0, Pcamb0 = np.loadtxt('camb/camb_mnv0.00000_om0.30000_As2.1000_matterpow_0.dat').T
    Pcamb0_interp = interp1d(kcamb0, Pcamb0)(k_arr)    
    knb0_1024, Pnb0_1024 = np.loadtxt('matterpower/mnv0.00000_om0.30000_As2.1000/powerspec_tot_067.txt').T
    knb0_1024*=1e3
    
    
    def get_PS (cosmo,nsnap):
        ifn='matterpower/%s/powerspec_tot_%03d.txt'%(cosmo, nsnap)
        kcamb, Pcamb = np.loadtxt('camb/camb_%s_matterpow_0.dat'%(cosmo)).T
        Pcamb_interp = interp1d(kcamb,Pcamb)(k_arr)
        knb, Pnb = np.loadtxt(ifn).T
        #knb *= 1e3
        #Pnb *= 1e-9
        return Pcamb_interp/Pcamb0_interp-1, Pnb/Pnb0_1024-1

    params = loadtxt('params.txt')
    m_nu_arr = params.T[0]
    params[:-2]=params[:-2][argsort(m_nu_arr[:-2])]

    nsnaps_arr = genfromtxt('nsnaps.txt')#[notfailed==0]

    for i in arange(0, len(params)-1,25):
        f=figure(figsize=(8,5.5))
        ax2=f.add_subplot(111)
        seed(2046)
        ax2.plot([1e-2,20],[0,0],'k-',lw=1,alpha=0.5)
        j=0
        for iparam in params[i:i+25]:
            M_nu, omega_m, A_s9 = iparam
            #ilabel = r'$M_\nu=%.3f, \Omega_m=%.3f, 10^9A_s=%.3f$'%(M_nu, omega_m, A_s9)
            ilabel=r'$%.3f,\; %.2f,\; %.2f$'%(M_nu, omega_m, A_s9)
            icosmo = 'mnv%.5f_om%.5f_As%.4f'%(M_nu, omega_m, A_s9)
            dPcamb, dPnb = get_PS (icosmo, nsnaps_arr[i+j]-1)
            j+=1
            icolor=rand(3)
            ax2.plot(k_arr, dPcamb,'--',lw=1,color=icolor)
            ax2.plot(k_arr[:5], dPcamb[:5],'-',lw=1,color=icolor)
            ax2.plot(knb0_1024, dPnb,'-',lw=1,color=icolor,label=ilabel)
        ax2.set_xscale('log')
        ax2.set_ylim(-1,1.5)
        ax2.set_xlim(1e-2,10)
        ax2.set_xlabel(r'$k\; (h/{\rm Mpc})$',fontsize=18)
        ax2.set_ylabel(r'$P_{mm} / P_{mm}^{{\rm fidu}, 0eV} - 1$',fontsize=18)
        ax2.legend(frameon=0,loc=2,fontsize=10,ncol=4,labelspacing=0.11,columnspacing=0.3, 
                   handletextpad=0.1)
        ax2.set_title(r'${\rm Parameters}=[M_\nu, \Omega_m, 10^9A_s], \; {\rm Dashed: \; Linear\; Theory \;,\; Solid:\; Simulations}$',fontsize=13)
        plt.tight_layout()
        savefig('plots/plot_pmatter_all%i.pdf'%(i))
        
        close()

if plot_pmatter_evolve:
    def pmatter_evolve(nsnap):
        ell0,ps_0=loadtxt('matterpower/mnv0.00000_om0.30000_As2.1000/powerspec_tot_%03d.txt'%(nsnap)).T
        ell1,ps_1=loadtxt('matterpower/mnv0.10000_om0.30000_As2.1000/powerspec_tot_%03d.txt'%(nsnap)).T
        #diff_fidu = interp1d(ell1,ps_1,fill_value="extrapolate")(k_arr) /interp1d(ell0,ps_0,fill_value="extrapolate")(k_arr)-1
        diff_fidu = ps_1/ps_0-1.0
        return ell0,diff_fidu
    a_arr = loadtxt('params/outputs_mnv0.00000_om0.30000_As2.1000.txt')
    z_arr = 1.0/a_arr - 1.0
    f=figure(figsize=(8,5.5))
    ax=f.add_subplot(111)
    for nsnap in range(10,67):
        ell0, delta_P = pmatter_evolve(nsnap)
        ax.plot(ell0*1e3, delta_P, c=cm.jet(a_arr[nsnap]))
    ax.set_xscale('log')
    ax.set_ylim(-0.09,0.0)
    ax.set_xlim(1e-2,10)
    ax.set_xlabel(r'$k\; (h/{\rm Mpc})$',fontsize=18)
    ax.set_ylabel(r'$P_{mm}^{0.1eV} / P_{mm}^{0 eV} - 1$',fontsize=18)
    #ax.set_ylabel(r'$P_{mm} / P_{mm}^{{\rm fidu}, 0eV} - 1$',fontsize=18)
    cbaxes = f.add_axes([0.2, 0.3, 0.3, 0.03]) 

    cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cm.jet,#RdPu ,#
                                orientation='horizontal')
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=6)
    cb1.locator = tick_locator
    cb1.update_ticks()
    cb1.set_label(r'${\rm Scale\; Factor\;} a$', fontsize=16)
    plt.tight_layout()
    savefig('plots/plot_pmatter_evolve_jet.pdf')
    close()
        
        
if plot_sample_maps_include_reconstructed:
    import WLanalysis
    conv_cmb0=fits.open('sample_maps/mnv0.00000_om0.30000_As2.1000/WLconv_z1100.00_0001r.fits')[0].data
    conv_cmb1=fits.open('sample_maps/mnv0.10000_om0.30000_As2.1000/WLconv_z1100.00_0001r.fits')[0].data
    conv_recon0=fits.open('sample_maps/mnv0.00000_om0.30000_As2.1000/jia_recon_massless_000000000.fits')[0].data
    conv_recon1=fits.open('sample_maps/mnv0.10000_om0.30000_As2.1000/jia_recon_massive_000000000.fits')[0].data
    
    conv_gal0=fits.open('sample_maps/mnv0.00000_om0.30000_As2.1000/12deg/WLconv_z2.00_0001r.fits')[0].data
    conv_gal1=fits.open('sample_maps/mnv0.10000_om0.30000_As2.1000/12deg/WLconv_z2.00_0001r.fits')[0].data
    
    widths=[3.5,]*4+[6.3,]*2
    fns = ['conv_cmb0', 'conv_cmb1', 'conv_recon0', 'conv_recon1', 'conv_gal012deg', 'conv_gal112deg']
    for ismooth in (1, 5):
        for i in range(4,6):
            img=[conv_cmb0, conv_cmb1, conv_recon0, conv_recon1, conv_gal0, conv_gal1][i]
            #if i>3:
             #   img = img[455:-455,455:-455]
            iw = 3.5#widths[i]
            img = WLanalysis.smooth(img, img.shape[0]/iw/60*ismooth)
            if i%2==0:
                istd = std(img)
                print i, istd
            f=figure(figsize=(8,6))
            imshow(img,origin='lower', extent=[0,iw,0,iw], vmin=-2*istd, vmax=3*istd)
            xlabel(r'${\rm deg}$', fontsize=16)
            ylabel(r'${\rm deg}$', fontsize=16)
            title(fns[i])
            colorbar()
            plt.tight_layout()
            savefig('plots/map_%s_%iarcmin.png'%(fns[i],ismooth))
            close()

if plot_sample_maps:
    import WLanalysis
    conv_cmb0=fits.open('sample_maps/mnv0.00000_om0.30000_As2.1000/WLconv_z1100.00_0001r.fits')[0].data
    conv_cmb1=fits.open('sample_maps/mnv0.10000_om0.30000_As2.1000/WLconv_z1100.00_0001r.fits')[0].data
   
    conv_gal0=fits.open('sample_maps/mnv0.00000_om0.30000_As2.1000/12deg/WLconv_z2.00_0001r.fits')[0].data
    conv_gal1=fits.open('sample_maps/mnv0.10000_om0.30000_As2.1000/12deg/WLconv_z2.00_0001r.fits')[0].data
    
    #conv_gal0, conv_gal1, conv_cmb0, conv_cmb1 = [WLanalysis.smooth(img, img.shape[0]/60/3.5) for img in [conv_gal0, conv_gal1, conv_cmb0, conv_cmb1]]
    
    f=figure(figsize=(8,6))
    iii=0
    for img in [conv_gal0, conv_gal1-conv_gal0, conv_cmb0, conv_cmb1-conv_cmb0]:
        ax=f.add_subplot(2,2,iii+1)
        if iii%2:
            imgs=img.copy()
            
        else:
            imgs=WLanalysis.smooth(img, img.shape[0]/60/3.5)#
        istd = std(imgs)
        imshow(imgs,origin='lower', extent=[0,3.5,0,3.5], vmin=-3*istd, vmax=3*istd,cmap='jet')
        #title(fns[i])
        icb=colorbar()
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=6)
        icb.locator = tick_locator
        icb.update_ticks()
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        iii+=1
    plt.tight_layout()
    savefig('plots/plot_sample_maps.pdf')
    close()

            
if plot_hmf:#[2,3,4,5,6,17,27,28]
    import hmf
    import astropy.units as u
    ##0ID 1DescID 2Mvir 3Vmax 4Vrms 5Rvir 6Rs 7Np 8X 9Y 10Z 11VX 12VY 13VZ 14JX 15JY 16JZ 17Spin 18rs_klypin 19Mvir_all 20M200b 21M200c 22M500c 23M2500c 24Xoff 25Voff 26spin_bullock 27b_to_a 28c_to_a 29A[x] 30A[y] 31A[z] 32b_to_a(500c) 33c_to_a(500c) 34A[x](500c) 35A[y](500c) 36A[z](500c) 37T/|U| 38M_pe_Behroozi 39M_pe_Diemer 40Halfmass_Radius
    Mvir0, Vmax0, Vrms0, Rvir0, Rs0, Spin0, b2a0, c2a0, ID0, X0, Y0, Z0 = load('sample_maps/mnv0.00000_om0.30000_As2.1000/out_66.list.npy').T
    Mvir1, Vmax1, Vrms1, Rvir1, Rs1, Spin1, b2a1, c2a1, ID1, X1, Y1, Z1 = load('sample_maps/mnv0.10000_om0.30000_As2.1000/out_66.list.npy').T
    
    hist0, edges0 = histogram(log10(Mvir0), bins=linspace(11.5, 15.5,31))
    hist1, edges1 = histogram(log10(Mvir1), bins=linspace(11.5, 15.5,31))
    bincenter = edges0[:-1]
    delta_logbin = edges0[1]-edges0[0]
    
    Hmassles = hmf.MassFunction(cosmo_params={"Ob0":0.04551, "H0":70.0, "Om0":0.29997, 
           "m_nu":array([0,0,0])*u.eV},z=0, Mmin=11, Mmax=15.5, dlog10m=0.1, sigma_8=0.8523, n=0.97,hmf_model=hmf.fitting_functions.Tinker10)
    Hmassive = hmf.MassFunction(cosmo_params={"Ob0":0.04551, "H0":70.0, "Om0":0.29780, 
           "m_nu":array([0.1,0,0])*u.eV},z=0, Mmin=11, Mmax=15.5, dlog10m=0.1, sigma_8=0.8295, n=0.97,hmf_model=hmf.fitting_functions.Tinker10)
    
    f=figure(figsize=(8,6))
    ax1=f.add_subplot(211)
    ax2=f.add_subplot(212)
    
    mbins = arange(11, 15.5, 0.1)
    ax1.plot(10**bincenter, hist0/delta_logbin/512**3, 'g--', lw=2.5,label=r'${\rm Simulation}\; (M_\nu = 0\; {\rm eV})$'  )
    ax1.plot(10**bincenter, hist1/delta_logbin/512**3, 'g-',  lw=1.5,label=r'${\rm Simulation}\; (M_\nu = 0.1\; {\rm eV})$')
    ax1.plot(10**mbins, Hmassles.dndlog10m, '--', color='orange',lw=2.5, label=r'${\rm Tinker\; et\; al.\; 2010 \; (M_\nu = 0\; {\rm eV})}$')
    ax1.plot(10**mbins, Hmassive.dndlog10m, '-', color='orange', label=r'${\rm Tinker\; et\; al.\; 2010 \; (M_\nu = 0.1\; {\rm eV})}$')
   
    ax1.set_ylabel(r'$n \; (h^3/{\rm Mpc}^3)$',fontsize=22)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(frameon=0,loc=3,fontsize=12)    
    ax1.set_xlim(5e11, 1e15)
    ax2.plot(10**bincenter, hist1.astype(float)/hist0-1,'g-', lw=2.5, label=r'${\rm Simulation}$')  
    ax2.plot(10**mbins, Hmassive.dndlog10m/Hmassles.dndlog10m-1,'-', color='orange', lw=1.5, label=r'${\rm Tinker\; et\; al.\; 2010}$')  
    
    ax2.legend(frameon=0,fontsize=12,ncol=1,loc=0)
    ax2.set_xscale('log')
    ax2.set_xlim(5e11, 1e15)
    #ax2.set_ylim(-0.09,0.0)
    ax2.set_xlabel(r'$M_{vir}\; (M_\odot)$',fontsize=22)
    ax2.set_ylabel(r'$n^{0.1eV} / n^{0 eV} - 1$',fontsize=22)
    ax2.plot([5e11, 5e15], [0,0],'k--',lw=1)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.09,left=0.15)
    ax2.locator_params(axis = 'y', nbins = 5)
    plt.tight_layout()
    savefig('plots/plot_hmf.pdf')
    close()

if plot_merger_tree:
    #data0 = load('sample_maps/mnv0.00000_om0.30000_As2.1000/out_66.list.npy')
    #data1 = load('sample_maps/mnv0.10000_om0.30000_As2.1000/out_66.list.npy')
    #Mvir0, Vmax0, Vrms0, Rvir0, Rs0, Spin0, b2a0, c2a0, ID0, X0, Y0, Z0 = data0.T
    #Mvir1, Vmax1, Vrms1, Rvir1, Rs1, Spin1, b2a1, c2a1, ID1, X1, Y1, Z1 = data1.T
    
    #mass145=amin(abs(log10(Mvir0)-14.5))
    #idx0 = where (abs(log10(Mvir0)-14.5)==mass145)
    #dist = amin( sqrt((X0[idx0]-X1)**2 + (Y0[idx0]-Y1)**2 + (Y0[idx0]-Y1)**2) )
    #idx1 = where(sqrt((X0[idx0]-X1)**2 + (Y0[idx0]-Y1)**2 + (Y0[idx0]-Y1)**2) == dist)
    
    ##forest0 = loadtxt('sample_maps/mnv0.00000_om0.30000_As2.1000/forests.list').T
    ##forest1 = loadtxt('sample_maps/mnv0.10000_om0.30000_As2.1000/forests.list').T
    #forest0 = load('sample_maps/mnv0.00000_om0.30000_As2.1000/forests.list.npy')
    #forest1 = load('sample_maps/mnv0.10000_om0.30000_As2.1000/forests.list.npy')
    
    tree0 = loadtxt('sample_maps/tree_massless.txt')[:,1:]
    tree1 = loadtxt('sample_maps/tree_massive.txt')[:,1:]
    IDt0, IDt1=139120302,137345326
    ###### select all halos fall in the same tree root ID
    tree0 = tree0[where(tree0[:,29]==IDt0)]
    tree1 = tree1[where(tree1[:,29]==IDt1)]
    
    f=figure(figsize=(8,8))
    ax=f.add_subplot(111)
    jjj=0
    for itree in [tree0,tree1]:
        icolor=['orange','green',][jjj]
        icolor2=['darkorange','darkgreen'][jjj]
        a0, x0, mvir0 = itree[:, [0,19,10]].T
        x0-=itree[0,19]
        z0 = 1/a0
        
        #ax.scatter(x0,z0,edgecolor='none',facecolor='r',s=1.8**(log10(mvir0)-10))
        for iii in range(1,len(itree)):
            idx_proj = iii
            idx_desc = where (itree[:,1] ==itree[iii,3])[0][0]
            if iii==1:
                ax.plot( x0[[idx_proj, idx_desc]], z0[[idx_proj, idx_desc]], '-',color=icolor,lw=1,label=[r'$M_\nu = 0\; {\rm eV}$', r'$M_\nu = 0.1\; {\rm eV}$'][jjj] )
            else:
                ax.plot( x0[[idx_proj, idx_desc]], z0[[idx_proj, idx_desc]], '-',color=icolor,lw=0.5)
        idx_large = where(log10(mvir0)>13)#itree.T[14]
        ax.scatter(x0[idx_large],z0[idx_large],facecolor=icolor2,edgecolor='none',s=5**(log10(mvir0[idx_large])-12))#
        jjj+=1
    ax.set_yscale('log')
    ax.set_ylim(0.9,8)
    ax.set_xlim(-8,8)
    ax.set_yticks(arange(1,11))
    a=ax.get_yticks().tolist()
    anew = [str(int(ia-1)) for ia in a]
    ax.set_yticklabels(anew)
    [i.set_linewidth(1.5) for i in ax.spines.itervalues()]
    ax.tick_params(which='both', labelsize=18, width=1.5)
    ax.set_xlabel(r'$X - X_{\rm root}\; ({\rm Mpc}/h)$',fontsize=22)
    ax.set_ylabel(r'$z$',fontsize=22)
    ax.grid(True)#,axis='x')
    ax.legend(fontsize=18,loc=3,frameon=1)
    plt.tight_layout()
    savefig('plots/plot_tree.pdf')
    
    close()
    
if plot_mass_accretion:
    tree0 = loadtxt('sample_maps/tree_massless.txt')[:,1:]
    tree1 = loadtxt('sample_maps/tree_massive.txt')[:,1:]
    IDt0, IDt1=139120302,137345326
    ###### select all halos fall in the same tree root ID
    tree0 = tree0[where(tree0[:,29]==IDt0)]
    tree1 = tree1[where(tree1[:,29]==IDt1)]
    f=figure(figsize=(8,4))
    ax=f.add_subplot(111)
    jjj=0
    for itree in [tree0,tree1]:
        icolor=['darkorange','green',][jjj]
        ilabel=[r'$M_\nu = 0\; {\rm eV}$', r'$M_\nu = 0.1\; {\rm eV}$'][jjj]
        idx_desc = 0
        idx_proj_all = where (itree[:,3] ==itree[idx_desc,1])[0]
        imassarr,iaarr  = [itree[idx_desc,10]],[itree[idx_desc,0]]
        halfmass = itree[idx_desc,10]/2.0
        while len(idx_proj_all):
            idx_proj_biggest = idx_proj_all[argmax(itree[idx_proj_all,10])]
            imassarr.append(itree[idx_proj_biggest,10])
            iaarr.append(itree[idx_proj_biggest,0])
            idx_desc = idx_proj_biggest
            idx_proj_all = where (itree[:,3] ==itree[idx_desc,1])[0]
        imassarr = array(imassarr)
        iaarr = array(iaarr)
        ax.plot(1/iaarr,imassarr,color=icolor,lw=3-jjj,label=ilabel,alpha=0.8)#drawstyle='steps',
        #idx_half = argmin( abs(imassarr-halfmass))
        #ax.plot([1/iaarr[idx_half],1/iaarr[idx_half]], [1e11,1e15],'--',lw=3-jjj,color=icolor)
        jjj+=1
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks(arange(1,11))
    a=ax.get_xticks().tolist()
    anew = [str(int(ia-1)) for ia in a]
    ax.set_xticklabels(anew)
    [i.set_linewidth(1.5) for i in ax.spines.itervalues()]
    ax.tick_params(which='both', labelsize=18, width=1.5)
    ax.set_ylabel(r'$M_{vir}\; (M_\odot)$',fontsize=22)
    ax.set_xlabel(r'$z$',fontsize=22)
    ax.legend(fontsize=18,loc=0,frameon=0)
    ax.set_xlim(1,8)
    plt.tight_layout()
    
    savefig('plots/plot_most_massive_progenitor.pdf')
    close()
        
        

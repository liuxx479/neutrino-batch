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
plot_pmatter_evolve = 1
plot_sample_maps = 0

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
    for nsnap in range(1,67):
        ell0, delta_P = pmatter_evolve(nsnap)
        ax.plot(ell0*1e3, delta_P, c=cm.RdPu(a_arr[nsnap]))
    ax.set_xscale('log')
    ax.set_ylim(-0.1,0.0)
    ax.set_xlim(1e-2,10)
    ax.set_xlabel(r'$k\; (h/{\rm Mpc})$',fontsize=18)
    ax.set_ylabel(r'$P_{mm}^{0.1eV} / P_{mm}^{0 eV} - 1$',fontsize=18)
    #ax.set_ylabel(r'$P_{mm} / P_{mm}^{{\rm fidu}, 0eV} - 1$',fontsize=18)
    cbaxes = f.add_axes([0.25, 0.85, 0.5, 0.03]) 

    cb1 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cm.RdPu ,#jet,#
                                orientation='horizontal')
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=6)
    cb1.locator = tick_locator
    cb1.update_ticks()
    cb1.set_label(r'${\rm Scale\; Factor\;} a$', fontsize=16)
    savefig('plots/plot_pmatter_evolve.pdf')
    close()
        
        
if plot_sample_maps:
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

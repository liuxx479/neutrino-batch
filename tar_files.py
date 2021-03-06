import os
from scipy import *
from emcee.utils import MPIPool 
import sys

#os.system('''mkdir -pv /scratch/02977/jialiu/neutrino_sims
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/rockstar
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/trees
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/planes
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/convergence_maps
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/subsample''')

cosmo_jia_arr = genfromtxt('cosmo_jia_arr.txt',dtype='string')
cosmo_apetri_arr = genfromtxt('cosmo_apetri_arr.txt',dtype='string')

def create_targz(i):
    cosmo_jia = cosmo_jia_arr[i]
    cosmo_apetri =cosmo_apetri_arr[i]
    bash_planes='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512/ic1
tar -cvzf /scratch/02977/jialiu/neutrino_sims/planes/planes_{0}.tar.gz Planes'''.format(cosmo_jia, cosmo_apetri)
    bash_maps_gal='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512
tar -cvf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_gal_{0}.tar Maps??'''.format(cosmo_jia, cosmo_apetri)
    bash_maps_CMB='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512
tar -cvf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_CMB_{0}.tar Maps11000'''.format(cosmo_jia, cosmo_apetri)
    bash_maps_both='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512
tar -cvf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_6redshifts_{0}.tar Maps*'''.format(cosmo_jia, cosmo_apetri)
    bash_tree='''cd /scratch/02977/jialiu/temp/{0}/rockstar/trees
tar -cvzf /scratch/02977/jialiu/neutrino_sims/trees/trees_{0}.tar.gz *'''.format(cosmo_jia)
    #if not os.path.isfile('/scratch/02977/jialiu/neutrino_sims/trees/trees_{0}.tar.gz'.format(cosmo_jia)):
    bash_rockstar = '''cd /scratch/02977/jialiu/temp/{0}/rockstar
tar -cvzf /scratch/02977/jialiu/neutrino_sims/rockstar/rockstar_{0}.tar.gz out_*.list'''.format(cosmo_jia) 
    bash_subsample = '''cd /scratch/02977/jialiu/temp/{0}/snapshots_subsample
tar -cvzf /scratch/02977/jialiu/neutrino_sims/subsample/subsample_{0}.tar.gz *hdf5'''.format(cosmo_jia) 
    bash_maps_both2='''cd /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_6redshifts_{0}
tar -cvf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_6redshifts_{0}.tar Maps*'''.format(cosmo_jia, cosmo_jia)
    bash_maps_box5='''cd /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_6redshifts_{0}
tar -cvf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_6redshifts_{0}_box5.tar b*'''.format(cosmo_jia, cosmo_jia)
    bash_recon = '''cd /scratch/02977/jialiu/peakaboo/{0}/1024b512 
tar -cvf /scratch/02977/jialiu/neutrino_sims/recon/recon_eb_5000_s4_{1}.tar output_eb_5000_s4'''.format(cosmo_apetri, cosmo_jia)
    bash_recon_box5 = '''cd /scratch/02977/jialiu/peakaboo/{0}/1024b512/box5
tar -cvf /scratch/02977/jialiu/neutrino_sims/recon/recon_eb_5000_s4_{1}_box5.tar output_eb_5000_s4'''.format(cosmo_apetri, cosmo_jia)
    #os.system(bash_maps_CMB)
    #os.system(bash_maps_gal)
    #os.system(bash_planes)
    #os.system(bash_tree)
    #os.system(bash_rockstar)
    #os.system(bash_subsample)
    #os.system(bash_maps_both)
    os.system(bash_recon)
    #if cosmo_jia =='mnv0.00000_om0.30000_As2.1000':
        ##os.system(bash_maps_both2)
        #os.system(bash_recon_box5)
        #os.system(bash_maps_box5)
    
def unzip(i):    
    cosmo_jia = cosmo_jia_arr[i]
    cosmo_apetri =cosmo_apetri_arr[i]
    
    print cosmo_jia
    
    bash_planes='''tar -xvzf /scratch/02977/jialiu/neutrino_sims/planes/planes_{0}.tar.gz -C /scratch/02977/jialiu/lenstools_storage/{1}/1024b512/ic1/'''.format(cosmo_jia, cosmo_apetri)
    os.system(bash_planes)

def unzip_maps_edison (i):
    cosmo_jia = cosmo_jia_arr[i]
    print cosmo_jia
    foldername = '/global/cscratch1/sd/jialiu/convergence_6redshifts/convergence_6redshifts_'+cosmo_jia
    bash_tar_maps = '''mkdir -pv {0}    
    tar -xvf {0}.tar -C {0}    
    echo done untar {0}'''.format(foldername)

    bash_chmod = '''
    for i in {0}/M*
    do echo $i
    chmod 644 $i/*
    done
    echo done-done-done
    '''.format(foldername)
    #os.system(bash_tar_maps)
    os.system(bash_chmod)

def unzip_CMBmaps_edison (i):
    cosmo_jia = cosmo_jia_arr[i]
    print cosmo_jia
    foldername = '/global/cscratch1/sd/jialiu/convergence_6redshifts/convergence_6redshifts_'+cosmo_jia
    tarname = '/global/cscratch1/sd/jialiu/convergence_6redshifts/convergence_CMB_'+cosmo_jia+'.tar'
    bash_tar_maps = '''mkdir -pv {0}    
    tar -xvf {1} -C {0}    
    echo done untar {0}'''.format(foldername,tarname)

    bash_chmod = '''
    chmod 755 {0}/Maps11000
    chmod 644 {0}/Maps11000/*
    done
    echo done-done-done
    '''.format(foldername)
    os.system(bash_tar_maps)
    os.system(bash_chmod)
    
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

pool.map(create_targz, range(101))
#pool.map(create_targz, [0,])
#pool.map(unzip, range(101))
#pool.map(unzip_maps_edison, range(101))
#pool.map(unzip_CMBmaps_edison, range(101))
pool.close()

print 'done-done-done'

import os
from scipy import *
from emcee.utils import MPIPool 
import sys

os.system('''mkdir -pv /scratch/02977/jialiu/neutrino_sims
mkdir -pv /scratch/02977/jialiu/neutrino_sims/rockstar
mkdir -pv /scratch/02977/jialiu/neutrino_sims/trees
mkdir -pv /scratch/02977/jialiu/neutrino_sims/planes
mkdir -pv /scratch/02977/jialiu/neutrino_sims/convergence_maps
mkdir -pv /scratch/02977/jialiu/neutrino_sims/subsample''')

cosmo_jia_arr = genfromtxt('cosmo_jia_arr.txt',dtype='string')
cosmo_apetri_arr = genfromtxt('cosmo_apetri_arr.txt',dtype='string')

def create_targz(i):
    cosmo_jia = cosmo_jia_arr[i]
    cosmo_apetri =cosmo_apetri_arr[i]
    bash_planes='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512/ic1
tar -cvzf /scratch/02977/jialiu/neutrino_sims/planes/planes_{0}.tar.gz Planes'''.format(cosmo_jia, cosmo_apetri)
    bash_maps_gal='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512
tar -cvzf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_gal_{0}.tar.gz Maps??'''.format(cosmo_jia, cosmo_apetri)
    bash_maps_CMB='''cd /scratch/02977/jialiu/lenstools_storage/{1}/1024b512
tar -cvzf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_CMB_{0}.tar.gz Maps11000'''.format(cosmo_jia, cosmo_apetri)
    bash_tree='''cd /scratch/02977/jialiu/temp/{0}/rockstar/trees
tar -cvzf /scratch/02977/jialiu/neutrino_sims/trees/trees_{0}.tar.gz *'''.format(cosmo_jia)
    #if not os.path.isfile('/scratch/02977/jialiu/neutrino_sims/trees/trees_{0}.tar.gz'.format(cosmo_jia)):
    bash_rockstar = '''cd /scratch/02977/jialiu/temp/{0}/rockstar
tar -cvzf /scratch/02977/jialiu/neutrino_sims/rockstar/rockstar_{0}.tar.gz out_*.list'''.format(cosmo_jia) 
    bash_subsample = '''cd /scratch/02977/jialiu/temp/{0}/snapshots_subsample
tar -cvzf /scratch/02977/jialiu/neutrino_sims/subsample/subsample_{0}.tar.gz *hdf5'''.format(cosmo_jia) 
    #os.system(bash_maps_gal)
    #os.system(bash_maps_CMB)
    #os.system(bash_planes)
    #os.system(bash_tree)
    #os.system(bash_rockstar)
    os.system(bash_subsample)
    
def unzip(i):
    cosmo_jia = cosmo_jia_arr[i]
    cosmo_apetri =cosmo_apetri_arr[i]
    bash_planes='''tar -xvzf /scratch/02977/jialiu/neutrino_sims/planes/planes_{0}.tar.gz /scratch/02977/jialiu/lenstools_storage/{1}/1024b512/ic1/Planes'''.format(cosmo_jia, cosmo_apetri)
    os.system(bash_planes)

    
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

#pool.map(create_targz, range(101))
pool.map(unzip, range(101))
pool.close()

print 'done-done-done'

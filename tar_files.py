import os
from scipy import *
from emcee.utils import MPIPool 

#os.system('''mkdir -pv /scratch/02977/jialiu/neutrino_sims
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/rockstar
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/trees
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/planes
#mkdir -pv /scratch/02977/jialiu/neutrino_sims/convergence_maps''')

cosmo_jia_arr = genfromtxt('cosmo_jia_arr.txt',dtype='string')
cosmo_apetri_arr = genfromtxt('cosmo_apetri_arr.txt',dtype='string')

def create_targz(i):
    cosmo_jia = cosmo_jia_arr[i]
    cosmo_apetri =cosmo_apetri_arr[i]
    bash_command='''
tar -cvzf /scratch/02977/jialiu/neutrino_sims/trees/trees_{0}.tar.gz /scratch/02977/jialiu/temp/{0}/rockstar/trees
tar -cvzf /scratch/02977/jialiu/neutrino_sims/convergence_maps/convergence_{0}.tar.gz /scratch/02977/jialiu/lenstools_storage/{1}/1024b512/Maps??
'''.format(cosmo_jia, cosmo_apetri)
    if not os.path.isfile('/scratch/02977/jialiu/neutrino_sims/trees/trees_{0}.tar.gz'):
        bash_rockstar = 'tar -cvzf /scratch/02977/jialiu/neutrino_sims/rockstar/rockstar_{0}.tar.gz /scratch/02977/jialiu/temp/{0}/rockstar/out_*.list'.format(cosmo_jia) 
        os.system(bash_rockstar)
    os.system(bash_command)
    
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

pool.map(create_targz, range(101))
pool.close()

print 'done-done-done'

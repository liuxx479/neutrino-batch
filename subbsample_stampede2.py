from scipy import *
import subsample_gadget_snapshot
from emcee.utils import MPIPool 
import sys
import os

cosmo_arr = genfromtxt('cosmo_jia_arr.txt',dtype='string')
nsnaps_arr = loadtxt('nsnaps.txt')

def subsample(jjj):
    cosmo = cosmo_arr[jjj]
    cosmo_dir = '/scratch/02977/jialiu/temp/'+cosmo
    nsnaps = nsnaps_arr[jjj]
    for isnap in arange(nsnaps)[::-1]:
        print cosmo, isnap
        INPUT_FILENAME = cosmo_dir + '/snapshots/snapshot_%03d'%(isnap)
        OUTPUT_DIR = cosmo_dir + '/snapshots_subsample/'
        os.system('python subsample_gadget_snapshot.py %s %s' % (INPUT_FILENAME, OUTPUT_DIR))

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

pool.map(subsample, range(1,101))
pool.close()

print 'done-done-done'

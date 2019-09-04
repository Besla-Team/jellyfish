import numpy as np
import jellyfish


def test_loading_halo_pos():
    path = './examples/'
    snap_name = 'test_snap'
    nhost = 1000000
    nsat = 450000
    sim = jellyfish.Hello_sim(path, snap_name, nhost, nsat, 'host_dm', 'com_host', 'pos') 
    pos = sim.read_MW_snap_com_coordinates()
    assert(len(pos)==nhost), 'Length of particle array does not much with the \
                              expected {}'.format(nhost)
    assert(np.shape(pos)==(nhost, 3)), 'Wrong dimension of the position vector'
    



if __name__ == "__main__":
    test_loading_halo_pos()

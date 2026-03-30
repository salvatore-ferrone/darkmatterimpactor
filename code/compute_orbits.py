"""
This file computes the orbits of dark matter subhalos from dark matter subhalos taken from Pierre Bodlrini's DM only cosmological simulations.
These DM subhalos go to lower masses than typically computed for galaxy formation and evolution simulations.
We need these 'smaller' mass subhalos because they could impact gaps in stellar streams. 

1. Pick integration parameters
2. rescale the DM subhaloes positions and velocities for the analytical DM halo model used here
3. Find which sub-haloes have bound orbits
4. plot and save the image as E_Lz_subhaloes.png
5. Compute the orbits
6. Store them. 

INPUT: "DMsubhaloMW634Norm26PB.hdf5"
OUTPUT: "bound_orbitsDMsubhaloMW634Norm26PB.hdf5"

"""

import matplotlib.pyplot as plt
import numpy as np
import tstrippy
import h5py
import os 
import astropy.units as u
import datetime
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 16
})
author = "Salvatore Ferrone"

def main():

    infilename = "DMsubhaloMW634Norm26PB.hdf5"
    outfilename = "bound_orbits"+infilename.split(".hdf5")[0]+".hdf5"
    # SET THE INTEGRATION PARAMETERS 
    unitT = u.s * (u.kpc/u.km)
    integration_time = 5e9*u.yr
    dt = 5e5*u.yr
    Nsteps = int(integration_time/dt)
    # convert to good units
    dt = dt.to(unitT).value
    integrationparams = [0, dt, Nsteps]
    
    os.makedirs("outputs", exist_ok=True)
    
    # import the data
    with h5py.File('inputdata/'+infilename, 'r') as myfile:
        x = myfile['xh'][:]
        y = myfile['yh'][:]
        z = myfile['zh'][:]
        vx = myfile['vxh'][:]
        vy = myfile['vyh'][:]
        vz = myfile['vzh'][:]
        mh = myfile['mh'][:]

    # import the milky way parameters 
    MWparams            =   tstrippy.Parsers.pouliasis2017pii()
    ahalo               =   MWparams[2]

    # scale the positions 
    xh,yh,zh = x*ahalo,y*ahalo,z*ahalo
    fx,fy,fz,PHI = tstrippy.potentials.pouliasis2017pii(MWparams,xh,yh,zh)

    # get the circular velocity at each point 
    gmag    =   np.sqrt(fx**2+fy**2+fz**2)
    R       =   np.sqrt(xh**2+yh**2 + zh**2)
    vcirc   =   np.sqrt(gmag*R)
    # rescale the velocities 
    vxh,vyh,vzh = vx*vcirc,vy*vcirc,vz*vcirc
    T = 0.5*(vxh**2 + vyh**2 + vzh**2)
    E = T + PHI
    plot_E_Lz((xh,yh,zh),(vxh,vyh,vzh), tstrippy.potentials.pouliasis2017pii, MWparams)

    # now compute the orbits 
    bound = E<0
    Nobs = int(np.sum(bound))
    initconds = [xh[bound], yh[bound], zh[bound], vxh[bound], vyh[bound], vzh[bound]]

    print(f"Computing orbits for {Nobs} bound subhalos...")
    # perform the integration 
    starttime = datetime.datetime.now()
    tstrippy.integrator.deallocate()
    tstrippy.integrator.setinitialkinematics(*initconds)
    tstrippy.integrator.setintegrationparameters(*integrationparams)
    tstrippy.integrator.setstaticgalaxy("pouliasis2017pii",MWparams)
    tstrippy.integrator.setbackwardorbit()
    xt, yt, zt, vxt, vyt, vzt = tstrippy.integrator.leapfrogintime(Nsteps,Nobs)
    tstrippy.integrator.deallocate()
    endtime = datetime.datetime.now()
    print(f"Integration completed in {(endtime-starttime).total_seconds()/60:.2f} minutes.")
    comptime = (endtime-starttime).total_seconds()
    # create the time steps 
    timestamps = -np.arange(Nsteps+1)*dt
    timestamps = timestamps[::-1]
    # flip the data so that the final timestep is today 
    xt = xt[:,::-1]
    yt = yt[:,::-1]
    zt = zt[:,::-1]
    vxt = vxt[:,::-1]
    vyt = vyt[:,::-1]
    vzt = vzt[:,::-1]
    # write out the orbits
    data = np.zeros((Nsteps+1, Nobs, 6))
    data[:,:,0] = xt.T
    data[:,:,1] = yt.T
    data[:,:,2] = zt.T
    data[:,:,3] = vxt.T
    data[:,:,4] = vyt.T
    data[:,:,5] = vzt.T
    creationtime = datetime.datetime.now().isoformat()
    # print the file size
    filesize=(np.prod(data.shape[:]) * 8 * u.byte).to(u.GB)
    print(f"Output file size will be {filesize:.2f}."   )
    print(f"Writing output to outputs/{outfilename}...")
    with h5py.File('outputs/'+outfilename, 'w') as f:
        f.create_dataset('orbits', data=data)
        f.create_dataset('masses', data=mh[bound])
        f.create_dataset('timestamps', data=timestamps)
        f.create_dataset('integrationparams', data=integrationparams)
        f.create_dataset('MWparams', data=MWparams)
        f.create_dataset('initconds', data=initconds)
        f.attrs['computation_time_seconds'] = comptime
        f.attrs['creationtime'] = creationtime
        f.attrs['author'] = author
        f.attrs['description'] = "Orbits of bound subhalos in the MW potential, computed using tstrippy. The orbits are integrated backward in time for 5 Gyr with a timestep of 0.5 Myr. The data includes the positions and velocities at each timestep, as well as the masses of the subhalos and the parameters used for the integration."


def get_circ_velocity_E_Lz_curve(halomodel, haloparams,rmax=300,npoinst=1000):
    r = np.linspace(0,rmax,npoinst)
    x,y,z           =   r,0*r,0*r
    fx,fy,fz,PHI    =   halomodel(haloparams,x,y,z)    
    gmag            =   np.sqrt(fx**2+fy**2+fz**2)
    vcirc           =   np.sqrt(gmag*r)
    E = 0.5*vcirc**2 + PHI
    Lz = r*vcirc
    return E,Lz

def plot_E_Lz(positions,velocities, halomodel, haloparams,): 
    AXIS = {"xlabel": r'$L_z$ [$10^4$ kpc km/s]', 
            "ylabel": r'$E$ [$10^4$ km$^2$/s$^2$]',
            "title": r"L CDM subhalo orbits in the MW potential"}
    factor = 1e4
    vxh,vyh,vzh = velocities
    vs = np.sqrt(vxh**2+vyh**2+vzh**2)
    T = 0.5*vs**2
    PHI = halomodel(haloparams,positions[0],positions[1],positions[2])[3]
    E = T + PHI
    Lz = positions[0]*vyh - positions[1]*vxh
    bound = np.sum(E<0)

    # get the theoretical curve for circular orbits
    E_circ,Lz_circ = get_circ_velocity_E_Lz_curve(halomodel, haloparams)
    # get the axis limits
    xmax = np.max(Lz_circ/factor)
    xmin = -xmax
    ymin = np.min(E_circ/factor)
    ymax = np.max(E/factor)
    AXIS["xlim"] = (xmin,xmax)
    AXIS["ylim"] = (ymin,ymax)

    fig,axis = plt.subplots(1,1,figsize=(8.1,4))
    # add the circular orbit curve
    axis.plot(Lz_circ/factor,E_circ/factor,color='k',ls='-',)
    axis.plot(-Lz_circ/factor,E_circ/factor,color='k',ls='-',)
    # add the data points, coloring by whether they are bound or unbound
    axis.scatter(Lz[E<0]/factor,E[E<0]/factor,s=10)
    axis.scatter(Lz/factor,E/factor,s=1)
    axis.set(**AXIS)
    axis.text(0.05,0.05,f'Bound : {bound} / {len(E)}',transform=axis.transAxes,va='bottom',ha='left')
    fig.savefig('outputs/E_Lz_subhalos.png',dpi=300,bbox_inches='tight')
    plt.close(fig)        





if __name__ == "__main__":    
    main()
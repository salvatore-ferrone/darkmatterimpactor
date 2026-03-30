import numpy as np
import h5py
import plotly.graph_objects as go
import multiprocessing as mp 
from astropy import units as u 
import tstrippy
import datetime

# globals for worker processes (populated by init_worker)
_positions = None
_masses = None
_camera = None
_ranges = None
_intrinsic_sizes = None
_timestamps = None
_unitT = u.s * (u.kpc/u.km)
_aspect_ratio = 640 / 360 
_axis_limit = 40
_cylinder_verticies = None
_cylinder_faces = None

def init_worker(hdf5_path="outputs/bound_orbitsDMsubhaloMW634Norm26PB.hdf5"):
    """Initialization executed once in each worker process."""
    global _positions, _masses, _camera, _ranges, _intrinsic_sizes,_timestamps,_cylinder_faces,_cylinder_verticies
    MWparams = tstrippy.Parsers.pouliasis2017pii()
    rH = MWparams[6]
    zH = MWparams[-4]
    f = h5py.File(hdf5_path, "r")
    _positions = f["orbits"]
    _masses = f["masses"][:]
    _timestamps = f['timestamps'][:]
    _camera = dict(eye=dict(x=1.2, y=0, z=0.6))
    _ranges = ([-_aspect_ratio*_axis_limit,_aspect_ratio*_axis_limit], [-_aspect_ratio*_axis_limit,_aspect_ratio*_axis_limit], [-_axis_limit,_axis_limit])
    _intrinsic_sizes = set_marker_sizes_log_mass(_masses, floor=.2, ceiling=1.2)
    print(rH,zH)
    _cylinder_faces, _cylinder_verticies = make_disc(rH,zH)


def make_disc(r, height):
    # Add disk for scale
    theta = np.linspace(0, 2*np.pi, 20)
    r = 5
    x_top = r * np.cos(theta)
    y_top = r * np.sin(theta)
    z_top = np.ones_like(theta) * height
    x_bottom = x_top
    y_bottom = y_top
    z_bottom = np.ones_like(theta) * (-height)
    vertices_x = np.concatenate([x_top, x_bottom])
    vertices_y = np.concatenate([y_top, y_bottom])
    vertices_z = np.concatenate([z_top, z_bottom])
    # faces for sides
    faces_i = []
    faces_j = []
    faces_k = []
    n = len(theta)
    for i in range(n):
        i_next = (i + 1) % n
        faces_i.append(i)
        faces_j.append(i_next)
        faces_k.append(i + n)
        faces_i.append(i_next)
        faces_j.append(i_next + n)
        faces_k.append(i + n)   
    verticies = [vertices_x,vertices_y,vertices_z]
    faces = [faces_i,faces_j,faces_k]
    return faces, verticies

def make_frame(args):
    """Render frame using (sequential_frame_num, timestep_index)."""
    frame_index, snapshot_index = args  # j: sequential 0,1,2,...; i: actual timestep 0,100,200,...
    xyz = (_positions[snapshot_index,:,0], _positions[snapshot_index,:,1], _positions[snapshot_index,:,2])
    camera_xyz = (_camera['eye']['x'], _camera['eye']['y'], _camera['eye']['z'])
    depth_sizes = depth_perception(xyz, _ranges, camera_xyz, scalemin=0.5, scalemax=10)
    current_time = _timestamps[snapshot_index]
    # convert to Myr
    ctime = (current_time*_unitT).to(u.Myr)
    # format the string nicely 
    mytext = "{:5.1f} Myr".format(ctime.value)

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=_cylinder_verticies[0],
        y=_cylinder_verticies[1],
        z=_cylinder_verticies[2],
        i=_cylinder_faces[0],
        j=_cylinder_faces[1],
        k=_cylinder_faces[2],
        color='white',
        opacity=1
    ))
    fig.add_trace(go.Scatter3d(
        x=xyz[0], y=xyz[1], z=xyz[2],
        mode='markers',
        marker=dict(size=_intrinsic_sizes * depth_sizes, opacity=1)
    ))
    fig.add_annotation(
        text = mytext,
        x=1, y=1,
        xref = "paper", yref="paper",
        showarrow=False,
        font = dict(size=12,color="white"),
        xanchor="right", yanchor="top"
    )

    fig.add_annotation(
        text = "Milky Way thin disk shown to scale",
        x=0, y=0,
        xref = "paper", yref="paper",
        showarrow=False,
        font = dict(size=12,color="white"),
        xanchor="left", yanchor="bottom"
    )    

    fig.add_annotation(
        text = "LCDM subhalo population",
        x=0, y=1,
        xref = "paper", yref="paper",
        showarrow=False,
        font = dict(size=12,color="white"),
        xanchor="left", yanchor="top"
    )

    fig.add_annotation(
        text = "Boldrini, Di Matteo, Ferrone, 2026",
        x=1, y=0,
        xref = "paper", yref="paper",
        showarrow=False,
        font = dict(size=12,color="white"),
        xanchor="right", yanchor="bottom"
    )

    fig.update_layout(
        width=2*640,
        height=2*360,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(
            camera=_camera,
            bgcolor='black',
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showline=False, showticklabels=False, title=dict(text=''),range=_ranges[0]),
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showline=False, showticklabels=False, title=dict(text=''),range=_ranges[1]),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showline=False, showticklabels=False, title=dict(text=''),range=_ranges[2]),
            aspectmode="data"
        ),
        showlegend=False,
    )
    fname = f'frames/frame_{frame_index:05d}.png'  # j is sequential
    fig.write_image(fname)
    del fig
    return frame_index


def depth_perception(xyz,ranges,camera_xyz,scalemin=1,scalemax=2):
    x, y, z = xyz
    xrange, yrange, zrange = ranges
    cam_x, cam_y, cam_z = camera_xyz

    # normalize data to [-1, 1] space to match camera coordinates
    xnorm = 2 * (x - xrange[0]) / (xrange[1] - xrange[0]) - 1
    ynorm = 2 * (y - yrange[0]) / (yrange[1] - yrange[0]) - 1
    znorm = 2 * (z - zrange[0]) / (zrange[1] - zrange[0]) - 1
    
    # compute distance from camera in normalized space
    dist = np.sqrt((xnorm - cam_x)**2 + (ynorm - cam_y)**2 + (znorm - cam_z)**2)

    # closer points = larger; invert and scale
    dist_min, dist_max = dist.min(), dist.max()
    size_func = scalemin + scalemax * (1 - (dist - dist_min) / (dist_max - dist_min))  # range [2, 10]

    return size_func


def set_marker_sizes_log_mass(masses,floor=1,ceiling=10):
    m = np.log10(masses)
    m_mean = m.mean()
    m_std = m.std()
    bottom = m_mean - 2*m_std
    top = m_mean + 2*m_std
    size_func = floor + (ceiling-floor)*(m - bottom)/(top - bottom) 
    return np.clip(size_func, floor, ceiling)


def main(nskip = 5):
    # spawn a pool where each worker initializes its own copy of the data
    ncpu = mp.cpu_count()
    with h5py.File("outputs/bound_orbitsDMsubhaloMW634Norm26PB.hdf5", "r") as f:
        ntimesteps = f['orbits'].shape[0]


    nframes = (ntimesteps + nskip - 1) // nskip
    print(f"Rendering {nframes} frames using {ncpu} CPU cores...")
    # create list of (sequential_j, timestep_i) pairs
    # ntimesteps = 600
    frame_args = [(frame_index, snap_index) for frame_index, snap_index in enumerate(range(0, ntimesteps, nskip))]
    # print hte first 10 
    # for i in range(10):
        # print(frame_args[i])
    starttime = datetime.datetime.now()
    with mp.Pool(ncpu, initializer=init_worker) as pool:
        pool.map(make_frame, frame_args, chunksize=10)
    endtime = datetime.datetime.now()

    comptime = (endtime-starttime).total_seconds()


    print("comp_time", comptime, "s")
    print("time per frame", comptime*ncpu/nframes)


if __name__ == "__main__":
    main()

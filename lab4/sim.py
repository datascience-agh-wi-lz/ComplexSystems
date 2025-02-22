import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import gamma
from tqdm import tqdm
"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

def W( x, y, z, h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w


def cubicSplineKernel(x, y, z, h):
    """
    Cubic Spline Kernel (3D)
    x, y, z : numpy arrays representing positions in 3D space
    h       : smoothing length (scalar)
    
    Returns:
        The cubic spline kernel evaluated at the input positions.
    """
    # Compute distance `r` in 3D space
    r = np.sqrt(x**2 + y**2 + z**2)  # Vector length
    
    # Normalized distance `q`
    q = r / h
    
    # Preallocate kernel array with zeros
    w = np.zeros_like(r)
    
    # Cubic spline kernel weights
    mask1 = (q >= 0) & (q < 1)
    mask2 = (q >= 1) & (q < 2)
    
    w[mask1] = (1 / (np.pi * h**3)) * (1 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3)
    w[mask2] = (1 / (np.pi * h**3)) * (0.25 * (2 - q[mask2])**3)
    
    # Outside the kernel's support radius, `w` remains zero
    return w

def cubicSplineKernelGrad(dx, dy, dz, h):
    """
    Gradient of the Cubic Spline Kernel (3D)
    dx, dy, dz : numpy arrays representing the differences in x, y, z coordinates
    h          : smoothing length (scalar)
    
    Returns:
        Gradients of the kernel in x, y, and z directions.
    """
    # Compute distance `r` in 3D space
    r = np.sqrt(dx**2 + dy**2 + dz**2)  # Vector length
    
    # Normalized distance `q`
    q = r / h
    
    # Preallocate gradient arrays with zeros
    dWx = np.zeros_like(dx)
    dWy = np.zeros_like(dy)
    dWz = np.zeros_like(dz)
    
    # Avoid division by zero for zero distances
    r_safe = np.where(r == 0, np.finfo(float).eps, r)
    
    # Compute the gradients for q < 1
    mask1 = (q >= 0) & (q < 1)
    coeff1 = (-3 * q[mask1] + 2.25 * q[mask1]**2) * (-1 / (np.pi * h**4))
    dWx[mask1] = coeff1 * (dx[mask1] / r_safe[mask1])
    dWy[mask1] = coeff1 * (dy[mask1] / r_safe[mask1])
    dWz[mask1] = coeff1 * (dz[mask1] / r_safe[mask1])
    
    # Compute the gradients for 1 <= q < 2
    mask2 = (q >= 1) & (q < 2)
    coeff2 = (-0.75 * (2 - q[mask2])**2) * (-1 / (np.pi * h**4))
    dWx[mask2] = coeff2 * (dx[mask2] / r_safe[mask2])
    dWy[mask2] = coeff2 * (dy[mask2] / r_safe[mask2])
    dWz[mask2] = coeff2 * (dz[mask2] / r_safe[mask2])
    
    # Gradients outside the kernel's support radius remain zero
    return dWx, dWy, dWz

 
	
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T
	
	return dx, dy, dz
	

def getDensity( r, pos, m, h , spline_kernel=False):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	M = r.shape[0]
	
	dx, dy, dz = getPairwiseSeparations( r, pos );
	
	if spline_kernel:
		rho = np.sum( m * cubicSplineKernel(dx, dy, dz, h), 1 ).reshape((M,1))
	else:
		rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
	
	return rho
	
	
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = k * rho**(1+1/n)
	
	return P
	

def getAcc( pos, vel, m, h, k, n, lmbda, nu, spline_kernel=False ):
	"""
	Calculate the acceleration on each SPH particle
	pos   is an N x 3 matrix of positions
	vel   is an N x 3 matrix of velocities
	m     is the particle mass
	h     is the smoothing length
	k     equation of state constant
	n     polytropic index
	lmbda external force constant
	nu    viscosity
	a     is N x 3 matrix of accelerations
	"""
	
	N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	rho = getDensity( pos, pos, m, h, spline_kernel )
	
	# Get the pressures
	P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	dx, dy, dz = getPairwiseSeparations( pos, pos )
 
	if spline_kernel:
		dWx, dWy, dWz = cubicSplineKernelGrad( dx, dy, dz, h )
	else:
		dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
	ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
	az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))
	
	# Add external potential force
	a -= lmbda * pos
	
	# Add viscosity
	a -= nu * vel
	
	return a
	
import os
from tqdm import tqdm
def simulate(N=400, h=0.1, nu=1, spline_kernel=False, split_star=False):
    """ SPH simulation with animation """
    # Simulation parameters
    t = 0      # current time of the simulation
    tEnd = 10  # time at which simulation ends
    dt = 0.07  # timestep
    M = 2      # star mass
    R = 0.75   # star radius
    k = 0.1    # equation of state constant
    n = 1      # polytropic index
    
    # Generate Initial Conditions
    np.random.seed(42)
    lmbda = 2 * k * (1 + n) * np.pi**(-3 / (2 * n)) * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n))**(1 / n) / R**2
    m = M / N
    pos = np.random.randn(N, 3)
    vel = np.zeros(pos.shape)
    if split_star:
        pos[:N // 2, 0] -= 1
        pos[N // 2:, 0] += 1
        vel[N // 2:, 1] = 1
        vel[:N // 2, 1] = -1
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu, spline_kernel)
    Nt = int(np.ceil(tEnd / dt))
    
    # For Animation
    frames = []  # Store frames as (positions, density)
    rlin = np.linspace(0, 1, 100)
    rr = np.zeros((100, 3))
    rr[:, 0] = rlin
    rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)

    # Simulation Main Loop
    for i in tqdm(range(Nt)):
        vel += acc * dt / 2
        pos += vel * dt
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu, spline_kernel)
        vel += acc * dt / 2
        t += dt
        
        rho = getDensity(pos, pos, m, h, spline_kernel)
        rho_radial = getDensity(rr, pos, m, h, spline_kernel)
        frames.append((pos.copy(), rho.flatten(), rho_radial.flatten()))
    
    # Plot and save animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.2, 1.2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 3)
    
    scatter = ax1.scatter([], [], c=[], cmap='autumn', s=10, alpha=0.5)
    line_analytic, = ax2.plot(rlin, rho_analytic, color='gray', linewidth=2)
    line_sim, = ax2.plot([], [], color='blue')
    
    def update(frame):
        pos, rho, rho_radial = frame
        scatter.set_offsets(pos[:, :2])
        scatter.set_array(np.minimum((rho - 3) / 3, 1))
        line_sim.set_data(rlin, rho_radial)
        return scatter, line_sim
    
    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(os.path.join(f"sph{N}_{h}_{nu}_{spline_kernel}_{split_star}.gif"), writer='pillow', fps=15)    
    return 0

if __name__ == "__main__":
    simulate(400, 0.1, 1, False)
    
    simulate(100, 0.1, 1, False)
    simulate(800, 0.1, 1, False)
    
    simulate(400, 3, 1, False)
    simulate(400, 0.001, 1, False)
    
    simulate(400, 0.1, 8, False)
    simulate(400, 0.1, 0.1, False)
    
    simulate(400, 0.1, 1, True)
    simulate(400, 0.1, 1, True, True)
    simulate(400, 0.1, 1, False, True)
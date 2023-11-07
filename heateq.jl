const USE_GPU = false
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel function diffusion2D_step!(T2, T, Ci, lam, dt, dx, dy)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2));
    return
end

function diffusion2D()
# Physics
lam        = 1.0;                                        # Thermal conductivity
cp_min     = 1.0;                                        # Minimal heat capacity
lx, ly     = 10.0, 10.0;                                 # Length of domain in dimensions x, y and z.

# Numerics
nx, ny = 256, 256;                              # Number of gridpoints dimensions x, y and z.
nt         = 30;                                        # Number of time steps
init_global_grid(nx, ny, 0);
dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension

# Array initializations
T   = @zeros(nx, ny);
T2  = @zeros(nx, ny);
Ci  = @zeros(nx, ny);

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-((x_g(ix,dx,Ci)-lx/1.5))^2-((y_g(iy,dy,Ci)-ly/2))^2) +
                                   5*exp(-((x_g(ix,dx,Ci)-lx/3.0))^2-((y_g(iy,dy,Ci)-ly/2))^2) for ix=1:size(T,1), iy=1:size(T,2)]) )
T  .= Data.Array([100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2) +
                   50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)])
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.

# Time loop
dt = min(dx^2,dy^2)*cp_min/lam/8.1;                 # Time step for the 2D Heat diffusion
for it = 1:nt
    @parallel diffusion2D_step!(T2, T, Ci, lam, dt, dx, dy);
    update_halo!(T2);
    T, T2 = T2, T;
end

p = heatmap(T[:,:], aspect_ratio=1, color=:viridis, cbar=false, title="Temperature distribution")

finalize_global_grid();
return p 
end

diffusion2D()
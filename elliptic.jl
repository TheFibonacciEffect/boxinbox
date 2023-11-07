using Plots
#imports a nice progress bar
using ProgressMeter

# supress warnings about No strict ticks found
import Logging
Logging.disable_logging(Logging.Warn)

"Struct to initialize all variables needed for the computation"
Base.@kwdef mutable struct Setup
    # physics
    lx   = 20.0
    β    = 1.
    η    = 10
    ρ    = 1.
    # numerics
    nx   = 1000
    nvis = 15
    # derived numerics
    dx   = lx/nx
    dt   = dx/√(ρ*β)/2
    dτ   = dt
    nt   = 2nx
    xc   = LinRange(dx/2,lx-dx/2,nx)
    # inital condition
    ic   = x-> 1+exp(-(x-lx/4)^2)-x/lx
    # C0 = ic.(xc)
    P0   = ic.(xc)
    P    = copy(P0)
    Vₓ   = zeros(nx-1) 

    qx = zeros(nx-1)

end


function finite_step!(s::Setup)
    #damped wave equation
    s.Vₓ .+= - 1/s.ρ .* diff(s.P0)/s.dx .* s.dt
    s.P0[2:end-1] .+= s.dt * (-diff(s.Vₓ)/s.dx - s.P0[2:end-1]/s.η) * 1/s.β
    # diffusion equation
    s.qx .-= s.dτ./(s.ρ + s.dτ/s.dc).*(diff(s.C)./s.dx.+s.dx./s.dc)
    s.C[2:end-1] .-= s.dτ.* (diff(s.qx)./s.dx) 
end

"modefies the Setup type in the input so that it contains the solution in the Setup.C field"
function solve_pde!(s::Setup)
    p = Progress(s.nt, 0.2)
    anim = @animate for i in 1:s.nt
        finite_step!(s)
        # visualisation
        plot(s.xc,s.P,label="concentration at t=$(round(s.dt*i,digits=1))",xlabel="distance",ylabel="concentration")
        plot!(s.xc,s.P0,label="inital concentration")
        next!(p)
    end every s.nvis
    return anim
end

function main()
    s = Setup()
    anim = solve_pde!(s)
    anim,s
end

anim,s = main();
gif(anim,fps=15)

m,n = findmax(s.C)
plot(s.xc,s.C,label="final concentration  (max:$(round(m,digits=2)) at $(s.xc[n])",xlabel="distance",ylabel="concentration")
plot!(s.xc,s.C0,label="inital concentration",xlabel="distance",ylabel="concentration")

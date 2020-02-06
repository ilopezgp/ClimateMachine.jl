using MPI
using CLIMA
using Logging
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using LinearAlgebra
using Printf
using Dates
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.Mesh.Grids: EveryDirection, HorizontalDirection, VerticalDirection, min_node_distance

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  if length(ARGS) > 0
    const integration_testing = parse(Bool, ARGS[1])
  else
    const integration_testing =
      parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  end
end

const output = parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_OUTPUT","false")))

include("advection_hyperdiffusion_model.jl")

struct ConstantHyperDiffusion <: HyperDiffusionProblem
end

function init_hyperdiffusion_tensor!(problem::ConstantHyperDiffusion, aux::Vars,
                                     geom::LocalGeometry)
  aux.D = problem.D
end

function initial_condition!(problem::ConstantHyperDiffusion, state, aux, x, t)
  r = norm(x) - t
  @inbounds begin 
    dr = r - 15
    state.ρ = abs(dr) < 1 ? cos(pi / 2 * dr) ^ 4 : 0
  end
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model, testname)
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  exactnames = statenames .* "_exact"

  writevtk(filename, Q, dg, statenames, Qe, exactnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., exactnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end


function run(mpicomm, dim, topl, N, timeend, FT, vtkdir, outputtime)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = cubedshellwarp
                                         )

  dx = min_node_distance(grid)
  #dt = dx ^ 4 / 25
  dt = dx
  @info "time step" dt
  dt = outputtime / ceil(Int64, outputtime / dt)

  model = HyperDiffusion{dim}(ConstantHyperDiffusion())
  dg = DGModel(model,
               grid,
               CentralNumericalFluxNonDiffusive(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end
  callbacks = (cbinfo,)
  if ~isnothing(vtkdir)
    # create vtk dir
    mkpath(vtkdir)

    vtkstep = 0
    # output initial step
    do_output(mpicomm, vtkdir, vtkstep, dg, Q, Q, model, "hyperdiffusion")

    # setup the output callback
    cbvtk = EveryXSimulationSteps(floor(outputtime/dt)) do
      vtkstep += 1
      Qe = init_ode_state(dg, gettime(lsrk))
      do_output(mpicomm, vtkdir, vtkstep, dg, Q, Qe, model,
                "hyperdiffusion")
    end
    callbacks = (callbacks..., cbvtk)
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=callbacks)

  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end

using Test
let
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  polynomialorder = 4
  base_num_elem = 4

  numlevels = 4
  for FT in (Float64,)
    result = zeros(FT, numlevels)
    for dim in (3)
      for l = 1:numlevels
        Ne = 2^(l-1) * base_num_elem

        vert_range = grid1d(10, 30, nelem = Ne)
        topl = StackedCubedSphereTopology(mpicomm, 6, vert_range)

        timeend = 1
        outputtime = 100

        @info (ArrayType, FT, dim)
        vtkdir = output ? "vtk_hyperdiffusion" *
                          "_poly$(polynomialorder)" *
                          "_dim$(dim)_$(ArrayType)_$(FT)" *
                          "_level$(l)" : nothing
        result[l] = run(mpicomm, dim, topl, polynomialorder,
                        timeend, FT, vtkdir,
                        outputtime)
      end
      @info begin
        msg = ""
        for l = 1:numlevels-1
          rate = log2(result[l]) - log2(result[l+1])
          msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
        end
        msg
      end
    end
  end
end

nothing

# Load Packages
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.SubgridScaleParameters
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using Random
using CLIMA.Atmos: vars_state, vars_aux

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- #
const (xmin,xmax)      = (0,20000)
const (ymin,ymax)      = (0,400)
const (zmin,zmax)      = (0,10000)
const Ne        = (160,2,80)
const polynomialorder = 1
const dim       = 3
const dt        = 1.0
const timeend   = 1000.0
# ------------- Initial condition function ----------- #
"""
@article{doi:10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2,
author = {Bryan, George H. and Fritsch, J. Michael},
title = {A Benchmark Simulation for Moist Nonhydrostatic Numerical Models},
journal = {Monthly Weather Review},
volume = {130},
number = {12},
pages = {2917-2928},
year = {2002},
doi = {10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2},
URL = { https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2 },
eprint = { https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2 }
"""
function Initialise_Rising_Bubble!(bl, state::Vars, aux::Vars, (x1,x2,x3), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP

  xc::FT        = 10000
  zc::FT        = 2000
  r             = sqrt((x1 - xc)^2 + (x3 - zc)^2)
  rc::FT        = 2000
  θ_ref::FT     = 300
  Δθ::FT        = 0

  if r <= rc
    Δθ = 2*cospi(0.5*r/rc)^2
  end
  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  state.moisture.ρq_tot = FT(0)
end
# --------------- Driver definition ------------------ #
function run(mpicomm, ArrayType, LinearType,
             topl, dim, Ne, polynomialorder,
             timeend, FT, dt)
  # -------------- Define grid ----------------------------------- #
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                           )
  # -------------- Define model ---------------------------------- #
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         ref_state=HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0)),
                        turbulence=AnisoMinDiss{FT}(1),
                            source=Gravity(),
                 boundarycondition=NoFluxBC(),
                        init_state=Initialise_Rising_Bubble!)
  #=
  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(IsothermalProfile(FT(T_0)),FT(0)), #NoReferenceState()
                     Vreman{FT}(C_smag), #SmagorinskyLilly{FT}(0.23) -> Allgemein verwendbar
                     EquilMoist(),
                     NoPrecipitation(),
                     NoRadiation(),
                     NoSubsidence{FT}(),
                     Gravity(),
                     NoFluxBC(),
                     Initialise_Rising_Bubble!)
                     =#
  # -------------- Define dgbalancelaw --------------------------- #
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())



  fast_model = LinearType(model)
  slow_model = RemainderModel(model, (fast_model,))
  slow_dg = DGModel(slow_model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient() ; auxstate=dg.auxstate)


  fast_dg = DGModel(fast_model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient(); auxstate=dg.auxstate)

  Q = init_ode_state(dg, FT(0))

  ns = 15
  mis = MIS2(slow_dg, fast_dg, (dg,Q) -> StormerVerlet(dg, [1,5], 2:4, Q), ns, Q; dt = dt, t0 = 0)


  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s
  FloatType = %s""" eng0 ArrayType FT

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(mis),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(20)  do (init=false)
    mkpath("./vtk-rtb/")
      outprefix = @sprintf("./vtk-rtb/DC_%dD_mpirankSPLITGravityF%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
      @debug "doing VTK output" outprefix
      writevtk(outprefix, Q, slow_dg, flattenednames(vars_state(model,FT)), dg.auxstate, flattenednames(vars_aux(model,FT)))
      step[1] += 1
      nothing
  end

  solve!(Q, mis; timeend=timeend, callbacks=(cbinfo,cbvtk))
  # End of the simulation information
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
engf/eng0
end
# --------------- Test block / Loggers ------------------ #
using Test
let
  CLIMA.init()
  ArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  for FT in (Float64,)
    brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                  range(FT(ymin); length=Ne[2]+1, stop=ymax),
                  range(FT(zmin); length=Ne[3]+1, stop=zmax))
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
    #for LinearType in (AtmosAcousticLinearModel, AtmosAcousticGravityLinearModel)
      #@show LinearType
      LinearType=AtmosAcousticGravityLinearModel
      engf_eng0 = run(mpicomm, ArrayType, LinearType,
                      topl, dim, Ne, polynomialorder,
                      timeend, FT, dt)
      @test engf_eng0 ≈ FT(0.9999997771981113)
    #end
  end
end

#nothing

export MultirateRungeKutta

LSRK2N = LowStorageRungeKutta2N
ARK = AdditiveRungeKutta

"""
    MultirateRungeKutta(slow_solver, fast_solver; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multirate Runge-Kutta scheme using two different RK
solvers. This is based on

Currently only the low storage RK methods can be used as slow solvers

### References

    @article{SchlegelKnothArnoldWolke2012,
      title={Implementation of multirate time integration methods for air
             pollution modelling},
      author={Schlegel, M and Knoth, O and Arnold, M and Wolke, R},
      journal={Geoscientific Model Development},
      volume={5},
      number={6},
      pages={1395--1405},
      year={2012},
      publisher={Copernicus GmbH}
    }
"""
mutable struct MultirateRungeKutta{SS, FS, RT} <: AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT

    function MultirateRungeKutta(
        slow_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        dt = getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))
        new{SS, FS, RT}(slow_solver, fast_solver, RT(dt), RT(t0))
    end
end

mutable struct MultirateRungeKutta{SS, FS, RT} <: AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT

    function MultirateRungeKutta(
        slow_solver::ARK,
        fast_solver,
        Q = nothing;
        dt = getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.Qhat))
        new{SS, FS, RT}(slow_solver, fast_solver, RT(dt), RT(t0))
    end
end

function MultirateRungeKutta(
    solvers::Tuple,
    Q = nothing;
    dt = getdt(solvers[1]),
    t0 = solvers[1].t,
) where {AT <: AbstractArray}
    if length(solvers) < 2
        error("Must specify atleast two solvers")
    elseif length(solvers) == 2
        fast_solver = solvers[2]
    else
        fast_solver = MultirateRungeKutta(solvers[2:end], Q; dt = dt, t0 = t0)
    end

    slow_solver = solvers[1]

    MultirateRungeKutta(slow_solver, fast_solver, Q; dt = dt, t0 = t0)
end

function dostep!(
    Q,
    mrrk::MultirateRungeKutta{SS},
    param,
    time,
    in_slow_δ = nothing,
    in_slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
) where {SS <: LSRK2N}
    dt = mrrk.dt

    slow = mrrk.slow_solver
    fast = mrrk.fast_solver

    slow_rv_dQ = realview(slow.dQ)

    groupsize = 256

    fast_dt_in = getdt(fast)

    for slow_s in 1:length(slow.RKA)
        # Currnent slow state time
        slow_stage_time = time + slow.RKC[slow_s] * dt

        # Evaluate the slow mode
        slow.rhs!(slow.dQ, Q, param, slow_stage_time, increment = true)

        if in_slow_δ !== nothing
            slow_scaling = nothing
            if slow_s == length(slow.RKA)
                slow_scaling = in_slow_scaling
            end
            # update solution and scale RHS
            event = Event(device(Q))
            event = update!(device(Q), groupsize)(
                slow_rv_dQ,
                in_slow_rv_dQ,
                in_slow_δ,
                slow_scaling;
                ndrange = length(realview(Q)),
                dependencies = (event,),
            )
            wait(device(Q), event)
        end

        # Fractional time for slow stage
        if slow_s == length(slow.RKA)
            γ = 1 - slow.RKC[slow_s]
        else
            γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
        end

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        slow_δ = slow.RKB[slow_s] / (γ)

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        nsubsteps = fast_dt_in > 0 ? ceil(Int, γ * dt / fast_dt_in) : 1
        fast_dt = γ * dt / nsubsteps

        updatedt!(fast, fast_dt)

        for substep in 1:nsubsteps
            slow_rka = nothing
            if substep == nsubsteps
                slow_rka = slow.RKA[slow_s % length(slow.RKA) + 1]
            end
            fast_time = slow_stage_time + (substep - 1) * fast_dt
            dostep!(Q, fast, param, fast_time, slow_δ, slow_rv_dQ, slow_rka)
        end
    end
    updatedt!(fast, fast_dt_in)
end

@kernel function update!(fast_dQ, slow_dQ, δ, slow_rka = nothing)
    i = @index(Global, Linear)
    @inbounds begin
        fast_dQ[i] += δ * slow_dQ[i]
        if slow_rka !== nothing
            slow_dQ[i] *= slow_rka
        end
    end
end

function dostep!(
    Q,
    mrrk::MultirateRungeKutta{SS},
    param,
    time::Real,
    in_slow_δ = nothing,
    in_slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
) where {SS <: ARK}

    dt = mrrk.dt
    slow = mrrk.slow_solver
    fast = mrrk.fast_solver

    # Get the linear solver for the outer IMEX loop
    slow_besolver! = slow.besolver!

    # Butcher tableau for the ARK method (explicit and implicit parts)
    slow_RKA_explicit = slow.RKA_explicit
    slow_RKA_implicit = slow.RKA_implicit
    slow_RKB = slow.RKB
    slow_RKC = slow.RKC

    # Explicit (slow_rhs!) and implicit (slow_rhs_implicit!) tendencies
    slow_rhs! = slow.rhs!
    slow_rhs_implicit! = slow.rhs_implicit!
    slow_Qhat = slow.Qhat
    slow_Qtt = slow.variant_storage.Qtt

    slow_Qstages = slow.Qstages
    slow_Rstages = slow.Rstages
    # slow_Lstages = slow.variant_storage.Lstages
    slow_rv_Q = realview(Q)
    slow_rv_Qstages = realview.(slow_Qstages)
    # slow_rv_Lstages = realview.(slow_Lstages)
    slow_rv_Rstages = realview.(slow_Rstages)
    slow_rv_Qhat = realview(slow_Qhat)
    slow_rv_Qtt = realview(slow_Qtt)
    slow_split_explicit_implicit = slow.split_explicit_implicit

    groupsize = 256
    Nouter_stages = length(slow_RKB)
    fast_dt_in = getdt(fast)

    for slow_s in 1:Nouter_stages
        # Currnent slow state time
        slow_stage_time = time + slow_RKC[slow_s] * dt

        # Evaluate the slow mode using an IMEX method.
        # NOTE: This part of the code assumes that the IMEX method
        # employs an additive RK method with an explicit first stage
        # (no linear solve in the first stage).
        if slow_s === 1
            # No implicit solve during first slow stage
            slow_rhs!(
                slow_Rstages[slow_s],
                slow_Qstages[slow_s],
                param,
                slow_stage_time,
                increment = false,
            )
        else
            # implicit linear solve only appears after the
            # first slow stage is completed
            if in_slow_δ !== nothing
                slow_scaling = nothing
                if slow_s == length(slow_RKB)
                    slow_scaling = in_slow_scaling
                end
                # Update solution and scale RHS
                event = Event(device(Q))
                event = stage_update!(device(Q), groupsize)(
                    slow.variant,
                    slow_rv_Q,
                    slow_rv_Qstages,
                    slow_rv_Rstages,
                    slow_rv_Qhat,
                    slow_rv_Qtt,
                    slow_RKA_explicit,
                    slow_RKA_implicit,
                    dt,
                    Val(slow_s),
                    Val(slow_split_explicit_implicit),
                    slow_δ,
                    slow_scaling;
                    ndrange = length(slow_rv_Q),
                    dependencies = (event,),
                )
                wait(device(Q), event)
            end

            # Solves
            # Q_tt = Qhat + dt * RKA_implicit[istage, istage] * rhs_implicit!(Q_tt)
            α = dt * slow_RKA_implicit[slow_s, slow_s]
            slow_besolver!(slow_Qtt, slow_Qhat, α, param, slow_stage_time)

            # update Qstages
            slow_Qstages[slow_s] .+= slow_Qtt

            slow_rhs!(
                slow_Rstages[slow_s],
                slow_Qstages[slow_s],
                param,
                slow_stage_time,
                increment = false,
            )
        end

        # Fractional time for slow stage
        if slow_s == Nouter_stages
            γ = 1 - slow_RKC[slow_s]
        else
            γ = slow_RKC[slow_s + 1] - slow_RKC[slow_s]
        end

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        slow_δ = slow_RKB[slow_s] / γ
        nsubsteps = fast_dt_in > 0 ? ceil(Int, γ * dt / fast_dt_in) : 1
        fast_dt = γ * dt / nsubsteps

        updatedt!(fast, fast_dt)

        for substep in 1:nsubsteps
            slow_rka = nothing
            if substep == nsubsteps
                slow_rka = slow_RKA_explicit[slow_s % length(slow_RKA_explicit) + 1]
            end
            fast_time = slow_stage_time + (substep - 1) * fast_dt
            dostep!(Q, fast, param, fast_time, slow_δ, slow_rv_Rstages, slow_rka)
        end
    end

    # if slow_split_explicit_implicit
    #     for istage in 1:Nouter_stages
    #         stagetime = time + RKC[istage] * dt
    #         slow_rhs_implicit!(
    #             slow_Rstages[istage],
    #             slow_Qstages[istage],
    #             param,
    #             slow_stage_time,
    #             increment = true,
    #         )
    #     end
    # end
    #
    # # compose the final solution
    # event = Event(device(Q))
    # event = solution_update!(device(Q), groupsize)(
    #     slow.variant,
    #     slow_rv_Q,
    #     slow_rv_Rstages,
    #     slow_RKB,
    #     dt,
    #     Val(Nouter_stages),
    #     in_slow_δ,
    #     in_slow_rv_dQ,
    #     in_slow_scaling;
    #     ndrange = length(slow_rv_Q),
    #     dependencies = (event,),
    # )
    # wait(device(Q), event)
    updatedt!(fast, fast_dt_in)
end

@kernel function stage_mrrk_update!(
    ::LowStorageVariant,
    Q,
    Qstages,
    Rstages,
    Qhat,
    Qtt,
    RKA_explicit,
    RKA_implicit,
    dt,
    ::Val{is},
    ::Val{split_explicit_implicit},
    slow_δ,
    slow_dQ,
) where {is, split_explicit_implicit}
    i = @index(Global, Linear)
    @inbounds begin
        Qhat_i = Q[i]
        Qstages_is_i = -zero(eltype(Q))

        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end

        @unroll for js in 1:(is - 1)
            if split_explicit_implicit
                rkcoeff = RKA_implicit[is, js] / RKA_implicit[is, is]
            else
                rkcoeff =
                    (RKA_implicit[is, js] - RKA_explicit[is, js]) /
                    RKA_implicit[is, is]
            end
            commonterm = rkcoeff * Qstages[js][i]
            Qhat_i += commonterm + dt * RKA_explicit[is, js] * Rstages[js][i]
            Qstages_is_i -= commonterm
        end
        Qstages[is][i] = Qstages_is_i
        Qhat[i] = Qhat_i
        Qtt[i] = Qhat_i
    end
end

### Reference state
using DocStringExtensions
export NoReferenceState,
    HydrostaticState,
    IsothermalProfile,
    DecayingTemperatureProfile,
    DryAdiabaticProfile

using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav

"""
    ReferenceState

Reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

"""
    TemperatureProfile

Specifies the temperature profile for a reference state.

Instances of this type are required to be callable objects with the following signature

    T,p = (::TemperatureProfile)(orientation::Orientation, aux::Vars)

where `T` is the temperature (in K), and `p` is the pressure (in hPa).
"""
abstract type TemperatureProfile{FT <: AbstractFloat} end

vars_state_conservative(m::ReferenceState, FT) = @vars()
vars_state_gradient(m::ReferenceState, FT) = @vars()
vars_state_gradient_flux(m::ReferenceState, FT) = @vars()
vars_state_auxiliary(m::ReferenceState, FT) = @vars()
atmos_init_aux!(
    ::ReferenceState,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) = nothing

"""
    NoReferenceState <: ReferenceState

No reference state used
"""
struct NoReferenceState <: ReferenceState end



"""
    HydrostaticState{P,T} <: ReferenceState

A hydrostatic state specified by a temperature profile and relative humidity.
"""
struct HydrostaticState{P, FT} <: ReferenceState
    temperature_profile::P
    relative_humidity::FT
end
function HydrostaticState(
    temperature_profile::TemperatureProfile{FT},
) where {FT}
    return HydrostaticState{typeof(temperature_profile), FT}(
        temperature_profile,
        FT(0),
    )
end

vars_state_auxiliary(m::HydrostaticState, FT) =
    @vars(ρ::FT, p::FT, T::FT, ρe::FT, ρq_tot::FT)


function atmos_init_aux!(
    m::HydrostaticState{P, F},
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) where {P, F}
    T, p = m.temperature_profile(atmos.orientation, atmos.param_set, aux)
    FT = eltype(aux)
    _R_d::FT = R_d(atmos.param_set)

    aux.ref_state.T = T
    aux.ref_state.p = p
    aux.ref_state.ρ = ρ = p / (_R_d * T)
    q_vap_sat = q_vap_saturation(atmos.param_set, T, ρ)
    aux.ref_state.ρq_tot = ρq_tot = ρ * m.relative_humidity * q_vap_sat

    q_pt = PhasePartition(ρq_tot)
    aux.ref_state.ρe = ρ * internal_energy(atmos.param_set, T, q_pt)

    e_kin = F(0)
    e_pot = gravitational_potential(atmos.orientation, aux)
    aux.ref_state.ρe = ρ * total_energy(atmos.param_set, e_kin, e_pot, T, q_pt)
end


"""
    IsothermalProfile(param_set, T)

A uniform temperature profile, which is implemented
as a special case of [`DecayingTemperatureProfile`](@ref).
"""
IsothermalProfile(param_set::AbstractParameterSet, T::FT) where {FT} =
    DecayingTemperatureProfile{FT}(param_set, T, FT(0))

"""
    DryAdiabaticProfile{FT} <: TemperatureProfile{FT}


A temperature profile that has uniform dry potential temperature `θ`

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DryAdiabaticProfile{FT} <: TemperatureProfile{FT}
    "Surface temperature (K)"
    T_surface::FT
end

function (profile::DryAdiabaticProfile)(
    orientation::Orientation,
    param_set,
    aux::Vars,
)
    FT = eltype(aux)
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    z = altitude(orientation, param_set, aux)

    # Temperature
    T = profile.T_surface - _grav * z / _cp_d

    # Pressure
    p = _MSLP * (1 - _grav * z / (_cp_d * profile.T_surface))^(_cp_d / _R_d)

    return (T, p)
end

"""
    DecayingTemperatureProfile{F} <: TemperatureProfile{FT}

A virtual temperature profile that decays smoothly with height `z`, dropping by a specified temperature difference `ΔTv` over a height scale `H_t`.

```math
Tv(z) = \\max(Tv{\\text{surface}} − ΔTv \\tanh(z/H_{\\text{t}})
```

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DecayingTemperatureProfile{FT} <: TemperatureProfile{FT}
    "Virtual temperature at surface (K)"
    T_virt_surf::FT
    "Virtual temperature drop from surface to top of the atmosphere (K)"
    ΔTv::FT
    "Height scale over which virtual temperature drops (m)"
    H_t::FT
end
function DecayingTemperatureProfile{FT}(
    param_set::AbstractParameterSet,
    T_virt_surf::FT=FT(290),
    ΔTv::FT=FT(60),
) where {FT <: AbstractFloat}
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    H_t = _R_d * T_virt_surf / _grav
    return DecayingTemperatureProfile{FT}(T_virt_surf, ΔTv, H_t)
end


function (profile::DecayingTemperatureProfile)(
    orientation::Orientation,
    param_set::AbstractParameterSet,
    aux::Vars,
)
    z = altitude(orientation, param_set, aux)
    Tv = profile.T_virt_surf - profile.ΔTv * tanh(z / profile.H_t)
    FT = typeof(z)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    ΔTv_p = profile.ΔTv / profile.T_virt_surf
    p = -z - profile.H_t * ΔTv_p * log(cosh(z / profile.H_t) - atanh(ΔTv_p))
    p /= profile.H_t * (1 - ΔTv_p^2)
    p = _MSLP * exp(p)
    return (Tv, p)
end

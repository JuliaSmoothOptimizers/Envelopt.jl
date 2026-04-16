export EnveloptNLPModel, EnveloptLBFGSModel, EnveloptLSR1Model
export set_penalty!, set_multiplier!

"""
A structure to represent the problem

    minimize f(x) + h_μ (F(x) + μ y)  subject to  c(x) ∈ C,

where h_μ is the Moreau envelope of h with parameter μ > 0.

    EnveloptNLPModel(
      model::AbstractNLPModel{T, S},
      F::AbstractNLPModel{T, S},
      h,
      μ::T = 1.0,
    ) where {T, S}

## Arguments

* `model::AbstractNLPModel{T, S}`: a model that represents the problem

    minimize_x f(x) subject to c(x) ∈ C.

  If `model` is an `NCLModel`, model `F` below must be defined accordingly.

* `F::AbstractNLPModel{T, S}`: a model that represents the mapping F via the feasiblity problem

    minimize_x 0  subject to F(x) = 0.

  The objective and right-hand side of the constraints are ignored; only the body of the constraints is used.
  The feasiblity problem above is merely a convenient way to define a mapping F with access to its Jacobian.

  If `model` is an `NCLModel` (in variables x and r), `F` must represent a problem of the form

    minimize_{x,r} 0  subject to F(x) = 0,

  i.e., it has the same number of variables as `model`, but the constraints are independent of `r`.

  If argument `F` is omitted, it will be defined implictly as the identity mapping, i.e., F(x) = x,
  or, in the case of `NCLModel`s, F(x, r) = x.

* `h`: a proper, closed, convex function from ProximalOperators.jl.

## Optional arguments

* `μ::T`: the initial penalty parameter (default: 1.0).
"""
mutable struct EnveloptNLPModel{
  T,
  S,
  META <: AbstractNLPModelMeta{T, S},
  NLP <: AbstractNLPModel{T, S},
  FMODEL <: AbstractNLPModel{T, S},
  H,
} <: AbstractNLPModel{T, S}
  meta::META
  model::NLP  # min f(x)  s.t. c(x) ∈ C
  F::FMODEL   # should be given as  min 0  s.t. F(x) = 0 because we need the Jacobian of F
  h::H        # an operator from ProximalOperators.jl
  envelope::MoreauEnvelope{T, H}  # FIXME: allocates; implement our own?
  μ::T
  y::S
  Fμy::S   # temporary storage
  ph::S    # temporary storage
  jtFv::S  # temporary storage for the gradient of the Moreau envelope term, i.e., ∇F(x)' * Fμy
end

function EnveloptNLPModel(
  model::M,
  F::FMODEL,
  h::H;
  μ::T = one(T),
  x0::S = get_x0(model),
) where {T, S, M <: AbstractNLPModel{T, S}, FMODEL <: AbstractNLPModel{T, S}, H}
  get_nvar(model) == get_nvar(F) || error("number of variables in model and F must match")
  μ > 0 || error("penalty parameter must be > 0")
  get_ncon(F) > 0 || error("F model must have constraints")
  envelope = MoreauEnvelope(h, μ)
  y = fill!(similar(model.meta.x0, get_ncon(F)), zero(T))
  Fμy = similar(y)
  ph = similar(y)
  jtFv = similar(y, get_nvar(model))

  meta = NLPModelMeta(
    get_nvar(model),
    x0 = copy(x0),
    lvar = model.meta.lvar,
    uvar = model.meta.uvar,
    nlvb = model.meta.nlvb,
    nlvo = model.meta.nlvo,
    nlvc = model.meta.nlvc,
    ncon = model.meta.ncon,
    y0 = model.meta.y0,
    lcon = model.meta.lcon,
    ucon = model.meta.ucon,
    nnzj = model.meta.nnzj,
    lin_nnzj = model.meta.lin_nnzj,
    nln_nnzj = model.meta.nln_nnzj,
    nnzh = 0,
    lin = model.meta.lin,
    minimize = model.meta.minimize,
    islp = false,
    name = "envelopt-$(model.meta.name)",
    grad_available = model.meta.grad_available,
    jac_available = model.meta.jac_available,
    hess_available = false,
    jprod_available = model.meta.jprod_available,
    jtprod_available = model.meta.jtprod_available,
    hprod_available = model.meta.hprod_available,
  )

  return EnveloptNLPModel{T, S, typeof(model.meta), M, FMODEL, H}(
    meta,
    model,
    F,
    h,
    envelope,
    μ,
    y,
    Fμy,
    ph,
    jtFv,
  )
end

# By default, F(x) = x.
function EnveloptNLPModel(model::M, h::H; kwargs...) where {T, S, M <: AbstractNLPModel{T, S}, H}
  nvar = get_nvar(model)
  z = zeros(nvar)
  Fmodel = ADNLPModel(x -> 0.0, z, z, z, x -> x, z, z)
  EnveloptNLPModel(model, Fmodel, h; kwargs...)
end

# special case for NCLModels.
function EnveloptNLPModel(
  model::NCLModel{T, S, M},
  h::H;
  kwargs...,
) where {T, S, M <: AbstractNLPModel{T, S}, H}
  nvar = get_nvar(model)
  nx = model.nx
  zvar = zeros(nvar)
  zcon = zeros(nx)
  Fmodel = ADNLPModel(x -> 0.0, zvar, zvar, zvar, x -> x[1:nx], zcon, zcon)  # min_{x,r} 0  s.t. x = 0.
  EnveloptNLPModel(model, Fmodel, h; kwargs...)
end

@default_counters EnveloptNLPModel model

function eval_F(
  model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H},
  x::AbstractVector,
) where {T, S, META, NLP, FMODEL, H}
  Fval = similar(model.y)
  eval_F!(model, x, Fval)
end

function eval_F!(
  model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H},
  x::AbstractVector,
  out::AbstractVector,
) where {T, S, META, NLP, FMODEL, H}
  cons!(model.F, x, out)
  return out
end

function set_penalty!(
  model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H},
  μ::T,
) where {T, S, META, NLP, FMODEL, H}
  μ > 0 || error("penalty parameter must be > 0")
  model.μ = μ
  model.envelope = MoreauEnvelope(model.h, μ)
  return model
end

function set_multiplier!(
  model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H},
  y::S,
) where {T, S, META, NLP, FMODEL, H}
  model.y .= y
  return model
end

function set_multiplier!(
  model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H},
  y::T,
) where {T, S, META, NLP, FMODEL, H}
  model.y .= y
  return model
end

NLPModels.obj(model::EnveloptNLPModel, x::AbstractVector) = begin
  val = obj(model.model, x)
  eval_F!(model, x, model.Fμy)  # Fμy <- F(x)
  @. model.Fμy += model.μ * model.y
  val += model.envelope(model.Fμy)
  return val
end

# Efficient variant when f(x) and F(x) + μ * y are known.
NLPModels.obj(model::EnveloptNLPModel, fx::Real, Fμy::AbstractVector) = begin
  val = fx + model.envelope(Fμy)
  return val
end

NLPModels.grad!(model::EnveloptNLPModel, x::AbstractVector, g::AbstractVector) = begin
  grad!(model.model, x, g)
  eval_F!(model, x, model.Fμy)  # Fμy <- F(x)
  @. model.Fμy += model.μ * model.y
  prox!(model.ph, model.h, model.Fμy, model.μ)
  model.Fμy .-= model.ph
  model.Fμy ./= model.μ
  jtprod!(model.F, x, model.Fμy, model.jtFv)  # jtFv <- ∇F(x)' * Fμy = gradient of h_μ at F(x) + μy
  g .+= model.jtFv
  return g
end

# Solvers that call objgrad() save an evaluation of F on accepted steps
# and waste a gradient evaluation on rejected steps.
# A better approach is to cache the value of F(x) in obj() a reuse it in grad!().
# Perhaps use Memoize.jl on the definition of F?
NLPModels.objgrad!(model::EnveloptNLPModel, x::AbstractVector, g::AbstractVector) = begin
  fval = obj(model.model, x)
  eval_F!(model, x, model.Fμy)  # Fμy <- F(x)
  @. model.Fμy += model.μ * model.y
  fval += model.envelope(model.Fμy)
  grad!(model.model, x, g)
  prox!(model.ph, model.h, model.Fμy, model.μ)
  model.Fμy .-= model.ph
  model.Fμy ./= model.μ
  jtprod!(model.F, x, model.Fμy, model.jtFv)  # jtFv <- ∇F(x)' * Fμy = gradient of h_μ at F(x) + μy
  g .+= model.jtFv
  return fval, g
end

# A quasi-Newton model of an EnveloptNLPModel combines
# the second derivatives of f (as implemented in the input model) with a quasi-Newton
# approximation of the Hessian of g_μ.

abstract type EnveloptQuasiNewtonModel{T, S} <: QuasiNewtonModel{T, S} end

mutable struct EnveloptLBFGSModel{
  T,
  S,
  META <: AbstractNLPModelMeta{T, S},
  NLP <: AbstractNLPModel{T, S},
  FMODEL <: AbstractNLPModel{T, S},
  H,
  Op <: LBFGSOperator{T},
} <: EnveloptQuasiNewtonModel{T, S}
  meta::META
  env_model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H}
  op::Op  # LBFGS approximation of the Hessian of h_μ
  v::S    # temporary storage for Hessian approximation udpates
  hv::S   # temporary storage for Hessian-vector products
end

NLPModels.get_counters(model::EnveloptLBFGSModel) = get_counters(model.env_model)
NLPModelsModifiers.get_model(model::EnveloptLBFGSModel) = model.env_model
NLPModelsModifiers.get_op(model::EnveloptLBFGSModel) = model.op
@default_counters EnveloptLBFGSModel env_model (neval_hprod,)
NLPModels.neval_hprod(model::EnveloptLBFGSModel) = get_op(model).nprod

"Construct a structured LBFGSModel from an EnveloptNLPModel."
function EnveloptLBFGSModel(
  env_model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H};
  kwargs...,
) where {T, S, META, NLP, FMODEL, H}
  inner_model = env_model.model
  nx = isa(inner_model, NCLModel) ? inner_model.nx : get_nvar(inner_model)
  op = LBFGSOperator(T, nx; kwargs...)
  v = S(undef, nx)
  hv = S(undef, nx)
  return EnveloptLBFGSModel{T, S, META, NLP, FMODEL, H, typeof(op)}(
    env_model.meta,
    env_model,
    op,
    v,
    hv,
  )
end

mutable struct EnveloptLSR1Model{
  T,
  S,
  META <: AbstractNLPModelMeta{T, S},
  NLP <: AbstractNLPModel{T, S},
  FMODEL <: AbstractNLPModel{T, S},
  H,
  Op <: LSR1Operator{T},
} <: EnveloptQuasiNewtonModel{T, S}
  meta::META
  env_model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H}
  op::Op  # LSR1 approximation of the Hessian of h_μ
  v::S    # temporary storage for Hessian approximation udpates
  hv::S   # temporary storage for Hessian-vector products
end

NLPModels.get_counters(model::EnveloptLSR1Model) = get_counters(model.env_model)
NLPModelsModifiers.get_model(model::EnveloptLSR1Model) = model.env_model
NLPModelsModifiers.get_op(model::EnveloptLSR1Model) = model.op
@default_counters EnveloptLSR1Model env_model (neval_hprod,)
NLPModels.neval_hprod(model::EnveloptLSR1Model) = get_op(model).nprod

"Construct a structured LSR1Model from an EnveloptNLPModel."
function EnveloptLSR1Model(
  env_model::EnveloptNLPModel{T, S, META, NLP, FMODEL, H};
  kwargs...,
) where {T, S, META, NLP, FMODEL, H}
  inner_model = env_model.model
  nx = isa(inner_model, NCLModel) ? inner_model.nx : get_nvar(inner_model)
  op = LSR1Operator(T, nx; kwargs...)
  v = S(undef, nx)
  hv = S(undef, nx)
  return EnveloptLSR1Model{T, S, META, NLP, FMODEL, H, typeof(op)}(
    env_model.meta,
    env_model,
    op,
    v,
    hv,
  )
end

# For a "standard" initial model, we use H + B as approximation of the Hessian of the Lagranagian,
# where H is the Hessian of f and B is the quasi-Newton approximation of the Hessian of h_μ.
# If the initial model is an NCLModel, H has the form
#
# [ ∇²f   0  ]
# [  0    ρI ]
#
# and our approximation has the form
#
# [ ∇²f + B   0  ]  } nx
# [  0        ρI ].
NLPModels.hprod!(
  qn_model::EnveloptQuasiNewtonModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector,
) = begin
  inner_model = qn_model.env_model.model
  hprod!(inner_model, x, y, v, hv)  # product with the Hessian of the Lagrangian f(x) - yᵀc(x)
  nx = isa(inner_model, NCLModel) ? inner_model.nx : get_nvar(inner_model)
  @views hprod!(qn_model.op, x[1:nx], v[1:nx], qn_model.hv)  # product with the quasi-Newton approximation of the Hessian of h_μ
  @views hv[1:nx] .+= qn_model.hv
  return hv
end

NLPModels.hprod!(
  qn_model::EnveloptQuasiNewtonModel,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector,
) = begin
  inner_model = qn_model.env_model.model
  hprod!(inner_model, x, v, hv)     # product with the Hessian of f
  nx = isa(inner_model, NCLModel) ? inner_model.nx : get_nvar(inner_model)
  @views hprod!(qn_model.op, x[1:nx], v[1:nx], qn_model.hv)  # product with the LBFGS approximation of the Hessian of h_μ
  @views hv[1:nx] .+= qn_model.hv
  return hv
end

# Override default quasi-Newton callback to perform structured updates.
function JSOSolvers.default_callback_quasi_newton(
  qn_model::EnveloptQuasiNewtonModel,
  solver::AbstractSolver,
  stats::GenericExecutionStats,
)
  @debug "in callback_quasi_newton for EnveloptQuasiNewtonModels"
  if !stats.iter_reliable
    @error "iteration counter is not reliable, skipping Hessian approximation update"
    return
  end
  inner_model = qn_model.env_model.model
  nx = isa(inner_model, NCLModel) ? inner_model.nx : get_nvar(inner_model)
  if stats.iter == 0
    # save current gradient of Moreau envelopt for future update
    @views qn_model.v[1:nx] .= qn_model.env_model.jtFv[1:nx]  # jtFv is the gradient of the Moreau envelope term at the current point, i.e., ∇F(x)' * Fμy
  else
    if !stats.step_status_reliable
      @error "step status is not reliable, skipping Hessian approximation update"
      return
    end
    if stats.step_status == :accepted
      @debug "updating Hessian approximation with new gradient information"
      @views qn_model.v[1:nx] .-= qn_model.env_model.jtFv[1:nx]
      @views qn_model.v[1:nx] .*= -1  # v = ∇ₖ₊₁ - ∇ₖ
      @views push!(qn_model, solver.s[1:nx], qn_model.v[1:nx])
      @views qn_model.v[1:nx] .= qn_model.env_model.jtFv[1:nx]  # save gradient for next update
    end
  end
end

# the constraints are the same as in the original model

NLPModels.cons!(model::EnveloptNLPModel, x::AbstractVector, c::AbstractVector) =
  cons!(model.model, x, c)

# May need to be defined too in case some NLPModels don't implement the lin/nln parts separately.
#
# NLPModels.jac_structure!(
#   model::EnveloptNLPModel,
#   rows::AbstractVector,
#   cols::AbstractVector,
# ) = jac_structure!(model.model, rows, cols)
#
# NLPModels.jprod!(
#   model::EnveloptNLPModel,
#   x::AbstractVector,
#   v::AbstractVector,
#   jv::AbstractVector,
# ) = jprod!(model.model, x, v, jv)
#
# NLPModels.jtprod!(
#   model::EnveloptNLPModel,
#   x::AbstractVector,
#   v::AbstractVector,
#   jtv::AbstractVector,
# ) = jtprod!(model.model, x, v, jtv)

NLPModels.jac_lin_structure!(model::EnveloptNLPModel, rows::AbstractVector, cols::AbstractVector) =
  jac_lin_structure!(model.model, rows, cols)

NLPModels.jac_nln_structure!(model::EnveloptNLPModel, rows::AbstractVector, cols::AbstractVector) =
  jac_nln_structure!(model.model, rows, cols)

NLPModels.jac_lin_coord!(model::EnveloptNLPModel, x::AbstractVector, vals::AbstractVector) =
  jac_lin_coord!(model.model, x, vals)

NLPModels.jac_nln_coord!(model::EnveloptNLPModel, x::AbstractVector, vals::AbstractVector) =
  jac_nln_coord!(model.model, x, vals)

NLPModels.jprod_lin!(
  model::EnveloptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  jv::AbstractVector,
) = jprod_lin!(model.model, x, v, jv)

NLPModels.jprod_nln!(
  model::EnveloptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  jv::AbstractVector,
) = jprod_nln!(model.model, x, v, jv)

NLPModels.jtprod_lin!(
  model::EnveloptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
) = jtprod_lin!(model.model, x, v, jtv)

NLPModels.jtprod_nln!(
  model::EnveloptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
) = jtprod_nln!(model.model, x, v, jtv)

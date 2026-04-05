export EqualityFeasibilityModel

# Extract a feasibility problem from a problem with equality constraints.
# If the input model is
#
# minimize f(x)
# subj. to c(x) = 0
#          L ≤ g(x) ≤ U
#          ℓ ≤ x ≤ u,
#
# the resulting model is
#
# minimize 0  subj. to c(x) = 0.
#
mutable struct EqualityFeasibilityModel{
  T,
  S,
  M <: AbstractNLPModel{T, S},
  META <: AbstractNLPModelMeta{T, S},
} <: AbstractNLPModel{T, S}
  model::M
  meta::META
  c::S  # temporary storage for constraint values of the original model
  jrows::Vector{Int} # temporary storage for row indices of the Jacobian of the original model
  jcols::Vector{Int} # temporary storage for column indices of the Jacobian of the original model
  jvals::S  # temporary storage for Jacobian values of the original model
  v::S  # temporary storage for jtprod of the original model
end

function EqualityFeasibilityModel(model::AbstractNLPModel{T, S}) where {T, S}
  has_equalities(model) || error("model does not have equality constraints")
  #
  # compute indices of linear constraints among the equality constraints
  lin = Int[]
  jlin = 0
  for j ∈ model.meta.jfix ∩ model.meta.lin
    jlin += 1
    push!(lin, jlin)
  end

  # compute number of nonzeros in Jacobian
  nnzj = 0
  lin_nnzj = 0
  if model.meta.jac_available
    jrows, jcols = jac_structure(model)
    jvals = S(undef, model.meta.nnzj)

    for k = 1:model.meta.nnzj
      cons = jrows[k]
      if cons ∈ model.meta.jfix
        nnzj += 1
        if cons ∈ model.meta.lin
          lin_nnzj += 1
        end
      end
    end
  else
    jrows = Int[]
    jcols = Int[]
    jvals = S[]
  end

  nvar = get_nvar(model)
  ncon = length(model.meta.jfix)
  meta = NLPModelMeta(
    nvar,
    x0 = model.meta.x0,
    lvar = fill(-Inf, nvar),
    uvar = fill(Inf, nvar),
    nlvb = model.meta.nlvb,  # could be wrong, e.g., if equality constraints are linear
    nlvo = 0,
    nlvc = model.meta.nlvc,  # could be wrong, e.g., if equality constraints are linear
    ncon = ncon,
    y0 = model.meta.y0[model.meta.jfix],
    lcon = model.meta.lcon[model.meta.jfix],
    ucon = model.meta.ucon[model.meta.jfix],
    nnzo = 0,
    nnzj = nnzj,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nnzj - lin_nnzj,
    nnzh = model.meta.nnzh,  # upper bound
    lin = lin,
    minimize = true,
    islp = length(lin) == ncon,
    name = "eq-feasibility-$(model.meta.name)",
    # variable_bounds_analysis = false,
    # constraint_bounds_analysis = false,
    # sparse_jacobian = model.meta.sparse_jacobian,
    # sparse_hessian = model.meta.sparse_hessian,
    grad_available = true,  # gradient is always zero
    jac_available = model.meta.jac_available,
    hess_available = model.meta.hess_available,
    jprod_available = model.meta.jprod_available,
    jtprod_available = model.meta.jtprod_available,
    hprod_available = model.meta.hprod_available,
  )

  c = S(undef, model.meta.ncon)
  v = S(undef, model.meta.ncon)

  return EqualityFeasibilityModel{T, S, typeof(model), typeof(meta)}(
    model,
    meta,
    c,
    jrows,
    jcols,
    jvals,
    v,
  )
end

@default_counters EqualityFeasibilityModel model

NLPModels.obj(model::EqualityFeasibilityModel, x::AbstractVector) = zero(eltype(x))
NLPModels.grad!(model::EqualityFeasibilityModel, x::AbstractVector, g::AbstractVector) =
  fill!(g, zero(eltype(x)))

NLPModels.cons!(model::EqualityFeasibilityModel, x::AbstractVector, c::AbstractVector) = begin
  inner = model.model
  cons!(inner, x, model.c)
  @views c .= model.c[inner.meta.jfix]
  c
end

NLPModels.jac_structure!(
  model::EqualityFeasibilityModel,
  rows::AbstractVector,
  cols::AbstractVector,
) = begin
  inner = model.model
  jac_structure!(inner, model.jrows, model.jcols)
  l = 1
  for k = 1:inner.meta.nnzj
    cons = jrows[k]
    if cons ∈ inner.meta.jfix
      rows[l] = cons
      cols[l] = model.jcols[k]
      l += 1
    end
  end
  rows, cols
end

NLPModels.jac_coord!(model::EqualityFeasibilityModel, x::AbstractVector, jvals::AbstractVector) =
  begin
    jac_coord!(model.model, x, model.jvals)
    l = 1
    for k = 1:model.meta.nnzj
      cons = jrows[k]
      if cons ∈ model.meta.jfix
        jvals[l] = model.jvals[k]
        l += 1
      end
    end
    jvals
  end

# NLPModels.jprod_lin!(
#   model::EnveloptNLPModel,
#   x::AbstractVector,
#   v::AbstractVector,
#   jv::AbstractVector,
# ) = jprod_lin!(model.model, x, v, jv)

# NLPModels.jprod_nln!(
#   model::EnveloptNLPModel,
#   x::AbstractVector,
#   v::AbstractVector,
#   jv::AbstractVector,
# ) = begin
# jprod_nln!(model.model, x, v, jv)
#     end

NLPModels.jtprod_lin!(
  model::EqualityFeasibilityModel,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
) = begin
  model.v .= 0
  model.v[model.model.meta.jfix] .= v
  jtprod_lin!(model.model, x, model.v, jtv)
  jtv
end

NLPModels.jtprod_nln!(
  model::EqualityFeasibilityModel,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
) = begin
  model.v .= 0
  model.v[model.model.meta.jfix] .= v
  jtprod_nln!(model.model, x, model.v, jtv)
  jtv
end

NLPModels.jtprod!(
  model::EqualityFeasibilityModel,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
) = begin
  model.v .= 0
  model.v[model.model.meta.jfix] .= v
  jtprod!(model.model, x, model.v, jtv)
  jtv
end

# TODO: implement hess & Co.

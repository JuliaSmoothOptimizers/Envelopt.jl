export EqualityLessModel

# Remove equality constraints from a model.
# If the input model is
#
# minimize f(x)
# subj. to c(x) = 0
#          L ≤ g(x) ≤ U
#          ℓ ≤ x ≤ u,
#
# the resulting model is
#
# minimize f(x)
# subj. to L ≤ g(x) ≤ U
#          ℓ ≤ x ≤ u,
#
mutable struct EqualityLessModel{
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
  y::S  # temporary storage for multipliers of the original model
end

function EqualityLessModel(model::AbstractNLPModel{T, S}) where {T, S}
  has_equalities(model) || error("model does not have equality constraints")

  # compute indices of linear constraints among the inequality constraints
  lin = Int[]
  jlin = 0
  for j ∈ model.meta.lin
    j ∈ model.meta.jfix && continue
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
      cons ∈ model.meta.jfix && continue
      nnzj += 1
      if cons ∈ model.meta.lin
        lin_nnzj += 1
      end
    end
  else
    jrows = Int[]
    jcols = Int[]
    jvals = S[]
  end

  # compute initial multipliers and constraint limits
  ncon = model.meta.ncon - length(model.meta.jfix)
  y0 = similar(model.meta.y0, ncon)
  lcon = similar(model.meta.lcon, ncon)
  ucon = similar(model.meta.ucon, ncon)
  l = 1
  for j = 1:model.meta.ncon
    j ∈ model.meta.jfix && continue
    y0[l] = model.meta.y0[j]
    lcon[l] = model.meta.lcon[j]
    ucon[l] = model.meta.ucon[j]
    l += 1
  end

  meta = NLPModelMeta(
    get_nvar(model),
    x0 = model.meta.x0,
    lvar = model.meta.lvar,
    uvar = model.meta.uvar,
    nlvb = model.meta.nlvb,  # could be wrong if some linear constraints are removed
    nlvo = model.meta.nlvo,
    nlvc = model.meta.nlvc,  # could be wrong if some linear constraints are removed
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzo = model.meta.nnzo,
    nnzj = nnzj,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nnzj - lin_nnzj,
    nnzh = model.meta.nnzh,  # upper bound
    lin = lin,
    minimize = model.meta.minimize,
    islp = length(lin) == ncon,
    name = "eq-less-$(model.meta.name)",
    # variable_bounds_analysis = true,
    # constraint_bounds_analysis = true,
    # sparse_jacobian = get_sparse_jacobian(model),
    # sparse_hessian = get_sparse_hessian(model),
    grad_available = model.meta.grad_available,
    jac_available = model.meta.jac_available,
    hess_available = model.meta.hess_available,
    jprod_available = model.meta.jprod_available,
    jtprod_available = model.meta.jtprod_available,
    hprod_available = model.meta.hprod_available,
  )

  c = S(undef, model.meta.ncon)
  y = S(undef, model.meta.ncon)

  return EqualityLessModel{T, S, typeof(model), typeof(meta)}(
    model,
    meta,
    c,
    jrows,
    jcols,
    jvals,
    y,
  )
end

@default_counters EqualityLessModel model

NLPModels.obj(model::EqualityLessModel, x::AbstractVector) = obj(model.model, x)
NLPModels.grad!(model::EqualityLessModel, x::AbstractVector, g::AbstractVector) =
  grad!(model.model, x, g)
NLPModels.objgrad!(model::EqualityLessModel, x::AbstractVector, g::AbstractVector) =
  objgrad!(model.model, x, g)

NLPModels.cons!(model::EqualityLessModel, x::AbstractVector, c::AbstractVector) = begin
  inner = model.model
  cons!(inner, x, model.c)
  l = 1
  for j = 1:inner.meta.ncon
    j ∈ inner.meta.jfix && continue
    c[l] = model.c[j]
    l += 1
  end
  c
end

NLPModels.jac_structure!(model::EqualityLessModel, rows::Vector{Int}, cols::Vector{Int}) = begin
  inner = model.model
  jac_structure!(inner, model.jrows, model.jcols)
  l = 1
  for k = 1:inner.meta.nnzj
    cons = model.jrows[k]
    cons ∈ inner.meta.jfix && continue
    rows[l] = cons
    cols[l] = model.jcols[k]
    l += 1
  end
  rows, cols
end

NLPModels.jac_coord!(model::EqualityLessModel, x::AbstractVector, jvals::AbstractVector) = begin
  inner = model.model
  jac_coord!(inner, x, model.jvals)
  l = 1
  for k = 1:inner.meta.nnzj
    cons = jrows[k]
    cons ∈ inner.meta.jfix && continue
    jvals[l] = model.jvals[k]
    l += 1
  end
  jvals
end

NLPModels.hess_structure!(model::EqualityLessModel, rows::Vector{Int}, cols::Vector{Int}) =
  hess_structure(model.model, rows, cols)

NLPModels.hess_coord!(model::EqualityLessModel, x::AbstractVector, hvals::AbstractVector) =
  hess_coord(model.model, x, hvals)

# zero out multipliers corresponding to removed equality constraints
zero_out_eq_multipliers!(model::EqualityLessModel, y::AbstractVector) = begin
  inner = model.model
  l = 1
  for j = 1:inner.meta.ncon
    if j ∈ inner.meta.jfix
      model.y[j] = zero(eltype(y))
    else
      model.y[j] = y[l]
      l += 1
    end
  end
  model
end

NLPModels.hess_coord!(
  model::EqualityLessModel,
  x::AbstractVector,
  y::AbstractVector,
  hvals::AbstractVector;
  kwargs...,
) = begin
  zero_out_eq_multipliers!(model, y)
  hess_coord(model.model, x, model.y, hvals; kwargs...)
end

NLPModels.hprod!(
  model::EqualityLessModel,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  kwargs...,
) = hprod!(model.model, x, v, hv; kwargs...)

NLPModels.hprod!(
  model::EqualityLessModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  kwargs...,
) = begin
  zero_out_eq_multipliers!(model, y)
  hprod!(model.model, x, model.y, v, hv; kwargs...)
end

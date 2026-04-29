export envelopt

# abstract type for Envelopt subproblem solvers
abstract type AbstractEnveloptSubSolver end
name(sub::AbstractEnveloptSubSolver) = sub.name
failed(stats::GenericExecutionStats) = stats.status != :first_order
first_order(stats::GenericExecutionStats) = stats.status == :first_order
include("madnlp_sub.jl")
include("trunk_sub.jl")
include("tron_sub.jl")

# TODO: preallocate solver object so envelopt can be called in a loop
"""
    envelopt(::AbstractNLPModel, args...; kwargs...)
    envelopt(::EnveloptNLPModel, args...; kwargs...)

A solver for the nonsmooth regularized problem

    minimize f(x) + h(F(x))  subject to  c(x) ∈ C,

where h is a proper, closed, convex function, and f, F and c are smooth.
In the first calling form, the `AbstractNLPModel` represents the problem

    minimize f(x) subject to  c(x) ∈ C,

h should be part of `args...`, and F is implicity set to the identity mapping (F(x) = x).

In the second calling form, the `EnveloptNLPModel` will have been constructed from a user-defined F.

The Envelopt solver solves a sequence of Moreau envelope approximations of the problem via an augmented Lagrangian method.
Each subproblem has the form

    minimize f(x) + h_μ (F(x) + μ y)  subject to  c(x) ∈ C,

where h_μ is the Moreau envelope of g with parameter μ > 0, and y is a dual variable estimate.
The subproblems are solved with a solver chosen by the user using a quasi-Newton approximation of the Hessian.
"""
function envelopt(model::AbstractNLPModel, args...; kwargs...)
  env_model = EnveloptNLPModel(model, args...)
  envelopt(env_model; kwargs...)
end

function envelopt(
  env_model::EnveloptNLPModel;
  subsolver::AbstractEnveloptSubSolver = MadNLPEnveloptSubSolver(env_model),
  verbose::Bool = true,
  max_outer::Int = 20,
  dtol_min::Float64 = 1.0e-6,
  ptol_min::Float64 = 1.0e-6,
  callback = env_model -> false,  # use to update the model, e.g., if it is an NCLModel
  indent::String = "",
)
  NLPModels.reset!(env_model)
  NLPModels.reset!(env_model.model)
  NLPModels.reset!(env_model.F)

  x = get_x0(env_model)
  x0 = copy(x)  # to restore env_model.meta.x0 at the end
  y0 = copy(get_y0(env_model))
  # set_constraint_multipliers!(subsolver.stats, y0)
  subsolver.stats.multipliers .= y0
  subsolver.stats.multipliers_L .= 0
  subsolver.stats.multipliers_U .= 0

  T = eltype(x)
  set_multiplier!(env_model, zero(T))  # FIXME: smarter initialization
  set_penalty!(env_model, one(T))      # initial penalty
  Fμy = similar(env_model.Fμy)
  u = similar(env_model.ph)
  y = similar(env_model.y)

  dtol = 1.0e-1
  ptol = 1.0e-1

  dual_feasible = false
  primal_feasible = false  # true when b(x) ≈ u
  stationary = dual_feasible && primal_feasible
  subsolver_failed = false
  outer_iter = 0

  inner_model = env_model.model
  fval = obj(inner_model, x)
  # We don't have u at this point.
  # To save a prox evaluation, we use h(F(x)) instead of h(u).
  F_val = eval_F(env_model, x)
  hval = env_model.h(F_val)
  env_val = obj(env_model, x)
  yNorm = norm(env_model.y)

  if verbose
    @info @sprintf "%s%4s  %8s  %8s  %8s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %5s  %1s\n" indent "iter" "f" "h" "f + h_μ" "‖y‖" "μ" "ϵd" "ϵp" "dfeas" "pfeas" "‖F - u‖" "inner" "type"
    log_line =
      @sprintf "%s%4d  %8.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e  " indent outer_iter fval hval env_val yNorm env_model.μ dtol ptol
  end

  stats = subsolver.stats

  # FIXME: smarter stopping condition
  while !(stationary || subsolver_failed || outer_iter ≥ max_outer)
    # solve subproblem with x as initial guess
    subsolver(env_model, x, outer_iter; tol = dtol)
    subsolver_failed = failed(stats)

    if subsolver_failed
      verbose && @error "subproblem solver fails with" stats.status
      # FIXME: try to recover
      continue
    end

    x .= stats.solution
    eval_F!(env_model, x, F_val)
    @. Fμy = F_val + env_model.μ * env_model.y
    prox!(u, env_model.h, Fμy, env_model.μ)
    lift_feasibility = norm(F_val - u)
    kkt = max(stats.dual_feas, stats.primal_feas)

    feasibility = lift_feasibility
    if isa(inner_model, NCLModel)
      nx = inner_model.nx
      r = x[(nx + 1):end]
      rNorm = norm(r)
      feasibility = norm([lift_feasibility, rNorm])
    end

    if verbose
      log_line *=
        @sprintf "%7.1e  %7.1e  %7.1e  %5d  " stats.dual_feas stats.primal_feas feasibility stats.iter
    end

    if feasibility ≤ ptol
      @. y = env_model.y + (F_val - u) / env_model.μ
      set_multiplier!(env_model, y)
      yNorm = norm(env_model.y)
      if isa(inner_model, NCLModel)
        @. inner_model.y += inner_model.ρ * r
      end
      dtol = max(dtol_min / 2, min(dtol, kkt) / 5)
      ptol = max(ptol_min / 2, feasibility / 5)
      verbose && (log_line *= @sprintf "y\n")
    else
      set_penalty!(env_model, env_model.μ / 5)
      if isa(inner_model, NCLModel)
        # NCLModel uses penalty parameter ρ whereas Envelopt uses μ⁻¹.
        inner_model.ρ *= 5
      end
      dtol = max(dtol_min / 2, min(dtol, kkt) / 2)
      ptol = max(ptol_min / 2, lift_feasibility / 2)
      verbose && (log_line *= @sprintf "μ\n")
    end
    verbose && @info log_line

    fval = obj(inner_model, x)
    hval = env_model.h(u)
    env_val = obj(env_model, fval, Fμy)
    outer_iter += 1

    if verbose
      log_line =
        @sprintf "%s%4d  %8.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e  " indent outer_iter fval hval env_val yNorm env_model.μ dtol ptol
    end

    dual_feasible = kkt ≤ dtol_min
    primal_feasible = feasibility ≤ ptol_min
    stationary = dual_feasible && primal_feasible
  end
  verbose && @info log_line

  status = "unknown"
  if outer_iter ≥ max_outer
    verbose && @info "$(indent)maximum number of outer iterations reached"
    status = "max_iter"
  end
  if subsolver_failed
    status = "error"
  end
  if stationary
    verbose && @info "$(indent)found an approximate stationary point"
    status = "first_order"
  end

  copyto!(get_x0(env_model), x0)  # restore env_model.meta.x0
  copyto!(get_y0(env_model), y0)  # restore env_model.meta.y0

  # TODO: return u in stats?
  return stats, status, u
end

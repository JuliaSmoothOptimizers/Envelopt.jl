module EnveloptRegularizedOptimizationExt

using Envelopt
using NLPModels
using NLPModelsModifiers
using RegularizedOptimization
using RegularizedProblems
using SolverCore

# Subproblem solver based on a solver from RegularizedOptimization.jl for
#
#   minimize f(x) + h_μ(F(x) + μy) + regularizer(x),
#
# where the smooth part is the objective of an EnveloptNLPModel and
# the regularizer is handled through its proximal operator.
mutable struct RegOptEnveloptSubSolver{
  E <: EnveloptNLPModel,
  R <: AbstractRegularizedNLPModel,
  S <: AbstractOptimizationSolver,
  ST <: GenericExecutionStats,
  O <: NamedTuple,
} <: Envelopt.AbstractEnveloptSubSolver
  env_model::E  # the model passed to envelopt
  reg_nlp::R    # smooth part (env_model or a quasi-Newton model of it) + regularizer
  solver::S     # e.g., R2NSolver or R2Solver
  stats::ST
  options::O    # options passed to the solver at each solve
  name::String
end

const regopt_fixed_options = (max_iter = 10000, max_time = 600.0)

function check_unconstrained(env_model::EnveloptNLPModel, name::String)
  get_ncon(env_model) == 0 ||
    error("$name only supports problems without constraints c(x) ∈ C; include them in F")
end

# Allocate once: envelopt updates μ and y in env_model in place,
# so reg_nlp and the solver workspace remain valid across outer iterations.
function RegOptEnveloptSubSolver(
  env_model::EnveloptNLPModel,
  smooth_model::AbstractNLPModel,
  regularizer,
  selected,
  make_solver,
  name::String,
  options::NamedTuple,
)
  reg_nlp = RegularizedNLPModel(smooth_model, regularizer, selected)
  solver = make_solver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  return RegOptEnveloptSubSolver(
    env_model,
    reg_nlp,
    solver,
    stats,
    merge(regopt_fixed_options, options),
    name,
  )
end

# ... constructors
function Envelopt.R2NEnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  hessian_approximation::Symbol = :exact_f,
  mem::Int = 5,
  subsolver = R2Solver,
  m_monotone::Int = 1,
  selected = 1:get_nvar(env_model),
  kwargs...,
)
  @debug "initializing R2N subproblem solver"
  check_unconstrained(env_model, "R2NEnveloptSubSolver")

  # The Hessian of env_model is ∇²f + 0 because its hprod! is that of the inner model.
  smooth_model = if hessian_approximation == :exact_f
    env_model
  elseif hessian_approximation == :lbfgs
    LBFGSModel(env_model; mem = mem)
  elseif hessian_approximation == :lsr1
    LSR1Model(env_model; mem = mem)
  else
    throw(
      ArgumentError(
        "unknown Hessian approximation :$hessian_approximation; choose :exact_f, :lbfgs or :lsr1",
      ),
    )
  end

  make_solver = reg_nlp -> R2NSolver(reg_nlp; subsolver = subsolver, m_monotone = m_monotone)
  return RegOptEnveloptSubSolver(
    env_model,
    smooth_model,
    regularizer,
    selected,
    make_solver,
    "R2N",
    values(kwargs),
  )
end

function Envelopt.R2EnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  selected = 1:get_nvar(env_model),
  kwargs...,
)
  @debug "initializing R2 subproblem solver"
  check_unconstrained(env_model, "R2EnveloptSubSolver")
  # R2 only uses first-order information of env_model.
  make_solver = reg_nlp -> R2Solver(reg_nlp)
  return RegOptEnveloptSubSolver(
    env_model,
    env_model,
    regularizer,
    selected,
    make_solver,
    "R2",
    values(kwargs),
  )
end

# ... solve
function (sub::RegOptEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Real = 1.0e-6,
  kwargs...,
)
  env_model === sub.env_model ||
    error("$(sub.name)EnveloptSubSolver was constructed for a different EnveloptNLPModel")
  T = eltype(x0)
  SolverCore.solve!(
    sub.solver,
    sub.reg_nlp,
    sub.stats;
    x = x0,
    atol = T(tol),
    rtol = zero(T),
    sub.options...,
    kwargs...,
  )
  return sub.stats
end

Envelopt.regularizer_value(sub::RegOptEnveloptSubSolver, x::AbstractVector) =
  @views sub.reg_nlp.h(x[sub.reg_nlp.selected])

end

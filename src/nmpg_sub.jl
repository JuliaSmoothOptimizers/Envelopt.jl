export NMPGEnveloptSubSolver

using RegularizedProblems, RegularizedOptimization

# NMPG subproblem solver
mutable struct NMPGEnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::RegularizedOptimization.NMPGSolver
  stats::GenericExecutionStats
  name::String
end

# ... constructor
function NMPGEnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing NMPG subproblem solver"
  reg_nlp = RegularizedNLPModel(env_model, env_model.g)
  solver = RegularizedOptimization.NMPGSolver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  return NMPGEnveloptSubSolver(solver, stats, "NMPG")
end

# TODO add fixed options

# ... solve
function (sub::NMPGEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  reg_nlp = RegularizedNLPModel(env_model, env_model.g)
  RegularizedOptimization.solve!(
    sub.solver,
    reg_nlp,
    sub.stats,
    x = x0,
    atol = tol,
    rtol = 0.0,
    max_iter = 1_000_000,
    max_time = 600.0,
    w_monotone = 0.25,
    spectral_stepsize = true,
  )
  return sub.stats
end

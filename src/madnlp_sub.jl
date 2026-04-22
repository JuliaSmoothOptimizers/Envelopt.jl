export MadNLPEnveloptSubSolver

using MadNLP

# MadNLP subproblem solver
mutable struct MadNLPEnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::MadNLPSolver
  stats::MadNLP.MadNLPExecutionStats
  name::String
end

# ... constructor
function MadNLPEnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing MADNLP subproblem solver"
  solver =
    MadNLPSolver(env_model, hessian_approximation = MadNLP.CompactLBFGS, print_level = MadNLP.ERROR)
  stats = MadNLP.MadNLPExecutionStats(solver)
  return MadNLPEnveloptSubSolver(solver, stats, "MadNLP")
end

const madnlp_fixed_options = Dict(:max_iter => 100, :dual_initialized => true)

# ... solve
# FIXME: replace outer_iter and madnlp_y with the Envelopt solver object
function (M::MadNLPEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  outer_iter::Int,
  madnlp_y::AbstractVector;
  kwargs...,
)
  # FIXME: initialize solver outside the loop only and reuse.
  # See https://github.com/MadNLP/MadNLP.jl/discussions/552
  # MadNLP._reset!(solver.kkt.quasi_newton)
  M.solver =
    MadNLPSolver(env_model, hessian_approximation = MadNLP.CompactLBFGS, print_level = MadNLP.ERROR)

  # prepare for warm start
  # TODO: try solver.mu from the previous solve
  mu_init = 1.0e-1
  bound_push = 1.0e-2
  if outer_iter == 2
    mu_init = 1e-3
    bound_push = 1e-3
  elseif outer_iter == 4
    mu_init = 1e-5
    bound_push = 1e-5
  elseif outer_iter == 6
    mu_init = 1e-6
    bound_push = 1e-6
  elseif outer_iter == 8
    mu_init = 1e-7
    bound_push = 1e-7
  elseif outer_iter >= 10
    mu_init = 1e-8
    bound_push = 1e-8
  end

  # # copy multipliers for warm start
  # if outer_iter > 0
  #   copyto!(M.solver.y, madnlp_y)
  # end

  # MadnLP uses info from the problem itself to warm start.
  # The problem is stored inside the solver.
  copyto!(get_x0(M.solver.nlp), x0)  # to warm start the next outer iteration
  copyto!(get_y0(M.solver.nlp), M.stats.multipliers)

  return MadNLP.solve!(
    M.solver,
    M.stats;
    mu_init = mu_init,
    bound_push = bound_push,
    madnlp_fixed_options...,
    kwargs...,
  )
end

failed(stats::MadNLP.MadNLPExecutionStats) = stats.status != MadNLP.SOLVE_SUCCEEDED

module EnveloptJSOSolversExt

using Envelopt, NLPModels, SolverCore
using JSOSolvers
using NCL: NCLModel

include("JSOSolvers/tron.jl")
include("JSOSolvers/trunk.jl")

# Override default quasi-Newton callback to perform structured updates.
function JSOSolvers.default_callback_quasi_newton(
  qn_model::Envelopt.EnveloptQuasiNewtonModel,
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

end

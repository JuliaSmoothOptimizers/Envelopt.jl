# using NLPModelsTest
# using Test

@testitem "single subproblem solve with Trunk" tags=[:solver, :unconstrained, :trunk] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  trunk_solver = TrunkEnveloptSubSolver(env_model)
  stats = trunk_solver(env_model, get_x0(env_model), tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with TRON" tags=[:solver, :unconstrained, :tron] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  tron_solver = TronEnveloptSubSolver(env_model)
  stats = tron_solver(env_model, get_x0(env_model), tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  madnlp_solver = MadNLPEnveloptSubSolver(env_model)
  stats = madnlp_solver(env_model, get_x0(env_model), 0, tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with IPOPT" tags=[:solver, :unconstrained, :ipopt] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  ipopt_solver = IPOPTEnveloptSubSolver(env_model)
  stats = ipopt_solver(env_model, get_x0(env_model), 0, tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "simple solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, MadNLP, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  mad_solver =
    MadNLPSolver(env_model; hessian_approximation = MadNLP.CompactLBFGS, print_level = MadNLP.ERROR)
  stats = solve!(mad_solver)
  @test stats.status == MadNLP.SOLVE_SUCCEEDED
end

@testitem "simple unconstrained envelopt solve with IPOPT" tags=[:solver, :unconstrained, :ipopt] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model, subsolver = IPOPTEnveloptSubSolver(env_model), verbose = false)
  @test status == :first_order
end

@testitem "simple unconstrained envelopt solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u = envelopt(env_model, verbose = false)
  @test status == :first_order
end

@testitem "simple unconstrained envelopt solve with Trunk" tags=[:solver, :unconstrained, :trunk] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model; subsolver = TrunkEnveloptSubSolver(env_model), verbose = false)
  @test status == :first_order
end

@testitem "simple unconstrained envelopt solve with Tron" tags=[:solver, :unconstrained, :tron] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model; subsolver = TronEnveloptSubSolver(env_model), verbose = false)
  @test status == :first_order
end

@testitem "simple bound-constrained envelopt solve with Tron" tags=[
  :solver,
  :boundconstrained,
  :tron,
] begin
  using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, ProximalOperators
  model = hs1()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model; subsolver = TronEnveloptSubSolver(env_model), verbose = false)
  @test status == :first_order
end

@testitem "test constrained problem with MadNLP" tags=[:solvers, :constrained, :madnlp] begin
  using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, ProximalOperators
  model = hs13()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u = envelopt(env_model, verbose = false)
  @test status == :first_order
end

@testitem "test constrained problem with IPOPT" tags=[:solvers, :constrained, :ipopt] begin
  using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, ProximalOperators
  model = hs13()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model, subsolver = IPOPTEnveloptSubSolver(env_model), verbose = false)
  @test status == :first_order
end

@testitem "test complementarity problem with NCL" tags=[:solvers, :constrained, :mpcc] begin
  using ADNLPModels, NCL, ProximalOperators
  # min (x₁ - 1)² + (x₂ - 1)²  s.t. x₁ * x₂ = 0, x₁ ≥ 0, x₂ \geq 0.
  model = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 1)^2,
    [1.2; 1.2],
    [0.0, 0.0],
    [Inf, Inf],
    x -> [x[1] * x[2]],
    [0.0],
    [0.0],
  )
  ncl_model = NCLModel(model)
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(ncl_model, h)  # F(x, r) = x and μ = 1 by default
  stats, status, u = envelopt(env_model, verbose = false)
  @test status == :first_order
end

@testitem "get_status returns :unknown by default" tags=[:stats] begin
  @test Envelopt.get_status() == :unknown
end

@testitem "get_status returns :max_iter when outer_iter ≥ max_iter" tags=[:stats] begin
  @test Envelopt.get_status(5, 5, false, false, :unknown) == :max_iter
end

@testitem "get_status returns :first_order when stationary" tags=[:stats] begin
  @test Envelopt.get_status(20, 3, true, false, :unknown) == :first_order
end

@testitem "get_status returns substatus when subsolver failed" tags=[:stats] begin
  @test Envelopt.get_status(20, 3, false, true, :stalled) == :stalled
  @test Envelopt.get_status(20, 3, false, true, :max_iter) == :max_iter
end

@testitem "stats.iter counts outer iterations" tags=[:stats, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)
  max_outer = 3
  stats, status, u, tot_inner = envelopt(env_model; max_outer = max_outer, verbose = false)
  @test stats.iter ≤ max_outer
end

@testitem "stats.objective is finite after solve" tags=[:stats, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)
  stats, status, u, tot_inner = envelopt(env_model; verbose = false)
  @test isfinite(stats.objective)
end

@testitem "stats.solver_specific[:subiter] tracks total inner iterations" tags=[:stats, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)
  stats, status, u, tot_inner = envelopt(env_model; verbose = false)
  @test haskey(stats.solver_specific, :subiter)
  @test stats.solver_specific[:subiter] == tot_inner
  @test stats.solver_specific[:subiter] ≥ stats.iter
end

@testitem "stats.solution matches returned u prox" tags=[:stats, :madnlp] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)
  stats, status, u, tot_inner = envelopt(env_model; verbose = false)
  fx = obj(model, stats.solution)
  hu = h(u)
  @test stats.objective ≈ fx + hu
end

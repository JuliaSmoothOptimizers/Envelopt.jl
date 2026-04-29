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

@testitem "simple unconstrained envelopt solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u = envelopt(env_model, verbose = false)
  @test status == "first_order"
end

@testitem "simple unconstrained envelopt solve with Trunk" tags=[:solver, :unconstrained, :trunk] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model; subsolver = TrunkEnveloptSubSolver(env_model), verbose = false)
  @test status == "first_order"
end

@testitem "simple unconstrained envelopt solve with Tron" tags=[:solver, :unconstrained, :tron] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u =
    envelopt(env_model; subsolver = TronEnveloptSubSolver(env_model), verbose = false)
  @test status == "first_order"
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
  @test status == "first_order"
end

@testitem "test constrained problem" tags=[:solvers, :constrained, :madnlp] begin
  using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, ProximalOperators
  model = hs13()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats, status, u = envelopt(env_model, verbose = false)
  @test status == "first_order"
end

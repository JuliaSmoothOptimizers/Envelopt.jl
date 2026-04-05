@testmodule CommonHelpers begin
  using ADNLPModels
  # Define F(x) via the feasibility model
  # minimize 0  subject to F(x) = 0.
  # Only the body of the constraints is used, nothing else.
  Fhelper(F, nvar, ncon) = begin
    zv = zeros(nvar)
    zc = zeros(ncon)
    Fmodel = ADNLPModel(x -> 0.0, zv, zv, zv, F, zc, zc)
    return Fmodel
  end
end

@testitem "wrong dimensions" setup=[CommonHelpers] tags=[:model] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  Fmodel = CommonHelpers.Fhelper(x -> x[1], 1, 1)  # wrong number of variables
  @test_throws ErrorException EnveloptNLPModel(model, Fmodel, h)
  Fmodel = ADNLPModel(x -> 0.0, [1.0, 1.0])  # unconstrained
  @test_throws ErrorException EnveloptNLPModel(model, Fmodel, h)
end

@testitem "simple Rosenbrock" tags=[:model] begin
  using ADNLPModels, NLPModels, NLPModelsTest, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  @test get_nvar(env_model) == get_nvar(model)
  @test get_ncon(env_model) == get_ncon(model)
  @test env_model.μ == 1
  @test set_penalty!(env_model, 0.5).μ == 0.5
  @test_throws ErrorException set_penalty!(env_model, 0.0)
  @test length(env_model.y) == get_nvar(model) # because b(x) = x
  set_multiplier!(env_model, ones(get_nvar(model)))
  @test all(env_model.y .== 1.0)
  set_multiplier!(env_model, 2.0)
  @test all(env_model.y .== 2.0)
  g_errs = gradient_check(env_model)
  @test length(g_errs) == 0
end

@testitem "simple Rosenbrock LBFGS" tags=[:model, :lbfgs] begin
  using ADNLPModels, NLPModelsModifiers, NLPModelsTest, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  env_lbfgs_model = EnveloptLBFGSModel(env_model, mem = 5)
  @test isa(env_lbfgs_model, QuasiNewtonModel)
  g_errs = gradient_check(env_lbfgs_model)
  @test length(g_errs) == 0
end

@testitem "simple Rosenbrock LSR1" tags=[:model, :lsr1] begin
  using ADNLPModels, NLPModelsModifiers, NLPModelsTest, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  env_lsr1_model = EnveloptLSR1Model(env_model, mem = 5)
  @test isa(env_lsr1_model, QuasiNewtonModel)
  g_errs = gradient_check(env_lsr1_model)
  @test length(g_errs) == 0
end

@testitem "simple NCL model" tags=[:model, :ncl] begin
  using NCL,
    NLPModels,
    NLPModelsTest,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    ProximalOperators
  model = hs13()
  ncl_model = NCLModel(model)
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(ncl_model, h)  # F(x, r) = x and μ = 1 by default
  @test get_nvar(env_model) == get_nvar(ncl_model)
  @test get_ncon(env_model) == get_ncon(ncl_model)
  @test length(env_model.y) == get_nvar(model) # not ncl_model !
  set_multiplier!(env_model, ones(get_nvar(model)))
  @test all(env_model.y .== 1.0)
  g_errs = gradient_check(env_model)
  @test length(g_errs) == 0
end

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

@testitem "Hessian of EnveloptNLPModel is ∇²f + 0" tags=[:model, :hessian] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  env_model = EnveloptNLPModel(model, NormL1(1.0))
  x = [0.3, -0.7]
  v = [1.0, 2.0]
  Hv = hprod(model, x, v)

  @test hprod(env_model, x, v) == Hv
  @test hprod(env_model, x, v; obj_weight = 2.0) == 2 * Hv
  @test hess_op(env_model, x) * v == Hv

  # the operator follows in-place updates of x, as solvers do with their iterate
  xop = copy(x)
  H = hess_op(env_model, xop)
  xop .= [1.1, 0.4]
  @test H * v == hprod(model, xop, v)

  # second derivatives are counted on the inner model
  reset!(env_model)
  hprod(env_model, x, v)
  @test neval_hprod(env_model) == 1

  # quasi-Newton subsolvers rely on these
  @test !env_model.meta.hess_available
  @test env_model.meta.nnzh == 0
end

@testitem "Hessian of the Lagrangian of EnveloptNLPModel" tags=[:model, :hessian] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1]^2 + x[2]^2],
    [-Inf],
    [1.0],
  )
  env_model = EnveloptNLPModel(model, NormL1(1.0))
  x = [0.3, -0.7]
  y = [2.0]
  v = [1.0, 2.0]
  @test hprod(env_model, x, y, v) == hprod(model, x, y, v)
  @test hess_op(env_model, x, y) * v == hprod(model, x, y, v)
end

@testitem "Hessian of EnveloptNLPModel with NCL" tags=[:model, :hessian, :ncl] begin
  using NCL, NLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, ProximalOperators
  ncl_model = NCLModel(hs13())
  env_model = EnveloptNLPModel(ncl_model, NormL1(1.0))
  nvar = get_nvar(env_model)
  x = collect(range(0.1, 0.9, length = nvar))
  v = collect(range(1.0, 2.0, length = nvar))
  @test hprod(env_model, x, v) == hprod(ncl_model, x, v)
  @test hess_op(env_model, x) * v == hprod(ncl_model, x, v)
end

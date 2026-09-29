@testitem "callback is called before the loop and at each outer iteration" tags=[:callback, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)

  call_count = Ref(0)
  cb = (m, sub, st) -> (call_count[] += 1; nothing)

  stats, status, u, tot_inner = envelopt(env_model; callback = cb, verbose = false)

  @test call_count[] == stats.iter + 1
end

@testitem "callback receives the env_model, subsolver and stats" tags=[:callback, :madnlp] begin
  using ADNLPModels, ProximalOperators, SolverCore
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)

  received_types = []
  cb = (m, sub, st) -> push!(received_types, (typeof(m), typeof(st)))

  envelopt(env_model; callback = cb, verbose = false)

  @test all(t -> t[1] <: EnveloptNLPModel, received_types)
  @test all(t -> t[2] <: GenericExecutionStats, received_types)
end

@testitem "callback stopping via set_status! exits the loop" tags=[:callback, :madnlp] begin
  using ADNLPModels, ProximalOperators, SolverCore
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)

  call_count = Ref(0)
  function stopping_cb(m, sub, st)
    call_count[] += 1
    if call_count[] > 1
      set_status!(st, :user)
    end
  end

  stats, status, u, tot_inner = envelopt(env_model; callback = stopping_cb, verbose = false)

  @test status == :user
  @test stats.iter == 1
end

@testitem "callback can log μ at each iteration" tags=[:callback, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)

  μ_trace = Float64[]
  cb = (m, sub, st) -> push!(μ_trace, m.μ)

  envelopt(env_model; callback = cb, verbose = false)

  @test length(μ_trace) > 0
  @test all(μ_trace .> 0)
end

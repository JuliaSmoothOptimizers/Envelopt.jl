@testitem "apply penalty method to constrained problem" tags=[:penalty, :constrained] begin
  using ProximalOperators
  using OptimizationProblems, OptimizationProblems.ADNLPProblems
  nlp = hs8()
  stats, pfeas = exact_penalty_solver(nlp, NormL2, verbose = false, sub_verbose = false)
  @test Envelopt.first_order(stats)
end

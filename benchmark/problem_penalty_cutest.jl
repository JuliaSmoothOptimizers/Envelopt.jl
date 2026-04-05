using ProximalOperators
using OptimizationProblems, OptimizationProblems.ADNLPProblems

using Envelopt

# problem_name = "HS8"
# nlp = CUTEstModel(problem_name)
nlp = hs8()
stats, pfeas = exact_penalty_solver(nlp, NormL2)

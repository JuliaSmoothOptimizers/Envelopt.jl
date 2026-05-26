using ProximalOperators
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using Statistics

using Envelopt

print_info(iter, env_iters, nlp_iters) = println(
  "$(iter) iterations, median $(median(env_iters)) ENV iterations, median $(median(nlp_iters)) NLP iterations",
)

nlp = hs18()
for pnorm in [NormL1, NormL2, NormLinf]
  println("subsolver MADNLP - penalty $(pnorm) ------------------------------")
  stats, pfeas, iter, env_iters, nlp_iters =
    exact_penalty_solver(nlp, pnorm, sub_solver = MadNLPEnveloptSubSolver)
  print_info(iter, env_iters, nlp_iters)
  println("subsolver Ipopt   - penalty $(pnorm) ------------------------------")
  stats, pfeas, iter, env_iters, nlp_iters =
    exact_penalty_solver(nlp, pnorm, sub_solver = IPOPTEnveloptSubSolver)
  print_info(iter, env_iters, nlp_iters)
end

# FIXME exact_penalty_solver returns stats with envelopt's stats, not penalty solver

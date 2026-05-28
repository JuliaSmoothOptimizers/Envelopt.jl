using ProximalOperators
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using Statistics

using Envelopt

function print_info(iter, env_iters, nlp_iters)
  println("$(iter) penalty iterations")
  println("  ENV iterations: median $(median(env_iters)), max $(maximum(env_iters))")
  println("  NLP iterations: median $(median(nlp_iters)), max $(maximum(nlp_iters))")
end

nlp = hs118()
# nlp = zangwil3()
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

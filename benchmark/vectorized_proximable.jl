import ProximalCore: prox, prox!, is_convex

"""Utility to vectorize proximable functions with matrix-valued input

    Given a function f:(n,m) -> R, the command
        g = VectorizedProximable(f, n, m)
    returns the function g:(n*m) -> R such that g(x) = f(reshape(x,(n,m)))
    for all vectors x.
"""
struct VectorizedProximable
  matfun # proximable function
  n::Int
  m::Int
  X::Array
  Z::Array
  function VectorizedProximable(matfun, n, m)
    @assert n > 0
    @assert m > 0
    X = zeros(n, m)
    Z = zeros(n, m)
    new(matfun, n, m, X, Z)
  end
end

VectorizedProximable(matfun, n) = VectorizedProximable(matfun, n, n)

is_convex(f::VectorizedProximable) = is_convex(f.matfun)

function (f::VectorizedProximable)(x)
  f.X .= reshape(x, (f.n, f.m))
  return f.matfun(f.X)
end

function prox!(z, f::VectorizedProximable, x, gamma)
  f.X .= reshape(x, (f.n, f.m))
  fz = prox!(f.Z, f.matfun, f.X, gamma)
  z .= reshape(f.Z, f.n * f.m)
  return fz
end

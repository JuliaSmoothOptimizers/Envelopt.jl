import ProximalCore: prox, prox!, is_convex

"""
    NormL1Nonnegative(λ=1)
    With a nonnegative scalar parameter λ, return the sum of ``L_1`` norm and indicator of the nonnegative orthant
    ```math
    f(x) = λ\\cdot∑_i|x_i| + (x_i >= 0 ? 0 : Inf).
    ```
"""
struct NormL1Nonnegative{T}
  lambda::T
  function NormL1Nonnegative{T}(lambda::T) where {T}
    if !(eltype(lambda) <: Real)
      error("λ must be real")
    end
    if any(lambda .< 0)
      error("λ must be nonnegative")
    else
      new(lambda)
    end
  end
end

is_separable(f::Type{<:NormL1Nonnegative}) = true
is_convex(f::Type{<:NormL1Nonnegative}) = true
is_positively_homogeneous(f::Type{<:NormL1Nonnegative}) = true

NormL1Nonnegative(lambda::R = 1) where {R} = NormL1Nonnegative{R}(lambda)

function (f::NormL1Nonnegative)(x)
  R = eltype(x)
  for k in eachindex(x)
    if x[k] < 0
      return R(Inf)
    end
  end
  return f.lambda * norm(x, 1)
end

function prox!(y, f::NormL1Nonnegative, x::AbstractArray{<:Real}, gamma)
  @assert length(y) == length(x)
  n1y = eltype(x)(0)
  gl = gamma * f.lambda
  @inbounds @simd for i in eachindex(x)
    y[i] = max(0, x[i] - gl)
    n1y += y[i]
  end
  return f.lambda * n1y
end

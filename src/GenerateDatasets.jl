

"""
This module contains functiosn for generating datasets suitable for testing density estimation
"""
module GenerateDatasets

using Distributions
using StatsBase
import DensityEstimationML: sample_from_cdf

export approximate_support, original_sample

struct RectangularInterval{T}
    lb::T
    ub::T
end

Base.minimum(iv::RectangularInterval) = iv.lb
Base.maximum(iv::RectangularInterval) = iv.ub




"""
    approximate_support(d)

Returns the effective support for a distribution,
This can be overloaded to provide a restricted support for distribitions that have infinitely large supports,
but can be approximated by bounding them.
Falls back to `support(d)` + 10% to either side
"""
function approximate_support(d)
    sp = support(d)
    a=minimum(sp)
    b=maximum(sp)
    len = b-a
    typeof(sp)(a-0.1*len, b+0.1*len)
end

"""
    original_sample(d)
Returns a sample of a distribution of the size that was used in the corresponding paper.
Falls back to a default of 5000 elements
"""
original_sample(d) = rand(d, 5000)

"""
The distribution used by
Magdon-Ismail and Atiya, in their 1998 NIPS paper "Neural Networks for Denity Estimation."
This is a 1D GMM with two peaks, without any conditioning on any properties.
for their SIC and SLC method they tested with  n=100,
for their SIC they additionally tested with n=200

This dataset is not conditional.
"""
struct MagdonIsmailAndAtiya <: UnivariateDistribution{Continuous}
end

function backing_distribition(::MagdonIsmailAndAtiya)
    comp1 = Normal(-30, 3)
    comp2 = Normal(9, 9)
    MixtureModel([comp1, comp2], [0.3, 0.7])
end

approximate_support(::MagdonIsmailAndAtiya) = RealInterval(-50.0, 50.0)
original_sample(d::MagdonIsmailAndAtiya) = rand(backing_distribition(d), 200)


"""
The distribution used by example 1 of
Likas A, 2001, "Probability density estimation using artificial neural networks"

They only estimate it between -12 and 12

This dataset is not conditional.
"""
struct Likas1 <: UnivariateDistribution{Continuous}
end

function backing_distribition(::Likas1)
    MixtureModel([Normal(-7,0.5), Uniform(-3,-1), Uniform(1,3), Normal(7,0.5)])
end
approximate_support(::Likas1) = RealInterval(-12.0,12.0)
original_sample(d::Likas1) = rand(backing_distribition(d), 5000)


for wrapper in [:MagdonIsmailAndAtiya, :Likas1]
    for (mod,funs) in [(:Distributions, [:sampler, :mode, :modes]),
                       (:Base, [:minimum, :maximum, :mean, :var]),
                       (:StatsBase, [:skewness])
                      ]
        for ff in funs
            @eval $mod.$ff(d::$wrapper)=$mod.$ff(backing_distribition(d))
        end
    end
    for (mod,funs) in [(:Distributions, [:pdf, :logpdf, :cdf, :insupport]),
                       (:Base, [:quantile]),
                       (:StatsBase, [:entropy])
                      ]
        for ff in funs
            @eval $mod.$ff(d::$wrapper, x::Real)=$mod.$ff(backing_distribition(d), x)
        end
    end
    
    for (mod,funs) in [(:Distributions, [:mgf, :cf])]
        for ff in funs
            @eval $mod.$ff(d::$wrapper, x)=$mod.$ff(backing_distribition(d), x)
        end
    end
   
    @eval StatsBase.kurtosis(d::$wrapper, x::Bool)=StatsBase.kurtosis(backing_distribition(d),x)
end



"""
Generates a dataset with the same parameters as used in example 2 of tests by
Likas A, 2001, "Probability density estimation using artificial neural networks"
and also in
Modha, D. S. & Fainman, 1994 Y. A learning law for density estimation

The PDF is given by:

\[f(x)=\begin{cases}
    \frac{2-\frac{x}{2}}{6.5523}    & 0\le x<2\\
    \frac{2-(x-3)^{2}}{6.5523}      & 0\le x<3+\sqrt{2}\\
    0                   & otherwise
\end{cases}
\]

Thus the CDF is given by
\[
F(x)=\dfrac{1}{6.5523}\begin{cases}
    2x-\frac{x^{2}}{4} & 0\le x<2\\
    \frac{-x^{3}}{3}+3x^{2}-7x+\frac{23}{3} & 2\le x<3+\sqrt{2}\\
    0 & x<0
    1 & x>3+\sqrt{2}
\end{cases}
\]

Both works use the same 5000 samples
This dataset is not conditional.
"""
struct Likas2<:UnivariateDistribution{Continuous}
end

function Distributions.pdf(::Likas2, x::Real)
    1/6.5523 * if (x ≤ 0)
    0.0
    elseif (0<x ≤ 2)
    2.0-x/2
    elseif (2<x≤3+√2)
    2.0-(x-3.0)^2
    else
    @assert(3+√2 < x)
    0.0
    end
end


function Distributions.cdf(::Likas2, x::Real)
    1/6.5523 * if (x ≤ 0)
    0.0
    elseif (0<x ≤ 2)
    2x-x^2/4
    elseif (2<x≤3+√2)
    -x^3/3 + 3x^2 -7x + 23/3
    else
    @assert(3+√2 < x)
    6.5523
    end
end


function Base.rand(::Likas2)
    likas_2_cdf(x) = cdf(Likas2(), x[1])
    likas_2_pdf(x) = pdf(Likas2(), x[1])
    sample_from_cdf(likas_2_cdf, likas_2_pdf, [2.5])[1]
end

original_sample(d::Likas2) = rand(d, 5000)

Base.minimum(::Likas2) = -1.0
Base.maximum(::Likas2) = 6.0

const ModhaAndFainman=Likas2





"""
Generates a dataset with the same parameters as used in example 3 of tests by
Likas A, 2001, "Probability density estimation using artificial neural networks"

It is a 2D uniform retangle between 0 and 0.2 on both axis

Likas used 5000 samples.

This dataset is not conditional.
"""
struct Likas3 <: ContinuousMultivariateDistribution
end

Distributions.support(::Likas3)=RectangularInterval((0., 0.),(0.2, 0.2))
approximate_support(::Likas3)=RectangularInterval((-0.1, -0.1),(0.3, 0.3))
Base.rand(::Likas3) = 0.2 .* rand(2)
Base.length(::Likas3) = 2
original_sample(::Likas3, n=5000) = 0.2 .* rand((2, n))


function Distributions._pdf(d::Likas3, X::AbstractVector)
    if all(0.<X.<0.2)
        1/(0.2^2)
    else
        0.0
    end
end

Distributions._logpdf(d::Likas3, X::AbstractVector) = log(Distributions._pdf(d,X))

end #module

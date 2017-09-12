

"""
This module contains functiosn for generating datasets suitable for testing density estimation
"""
module GenerateDatasets

using Distributions
using StaticArrays

import DensityEstimationML: sample_from_cdf

export approximate_support

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
"""
approximate_support(d) = support(d)


"""
The distribution used by
Magdon-Ismail and Atiya, in their 1998 NIPS paper "Neural Networks for Denity Estimation.
This is a 1D GMM with two peaks, without any conditioning on any properties.
for their SIC and SLC method they tested with  n=100,
for their SIC they additionally tested with n=200

This dataset is not conditional.
"""
function magdon_ismail_and_atiya()
	comp1 = Normal(-30, 3)
	comp2 = Normal(9, 9)
	MixtureModel([comp1, comp2], [0.3, 0.7])
end

approximate_support(::typeof(magdon_ismail_and_atiya)) = RealInterval(-50.0, 50.0)
Base.rand(distfun::typeof(magdon_ismail_and_atiya),  n=200) = rand(distfun(), n)


"""
The distribution used by example 1 of
Likas A, 2001, "Probability density estimation using artificial neural networks"

They only estimate it between -12 and 12

This dataset is not conditional.
"""
function likas_1()
	MixtureModel([Normal(-7,0.5), Uniform(-3,-1), Uniform(1,3), Normal(7,0.5)])
end

approximate_support(::typeof(likas_1)) = RealInterval(-12.0,12.0)
Base.rand(distfun::typeof(likas_1),  n=5000) = rand(distfun(), n)


"""
Generates a dataset with the same parameters as used in example 2 of tests by
Likas A, 2001, "Probability density estimation using artificial neural networks"
and also in
Modha, D. S. & Fainman, 1994 Y. A learning law for density estimation

The PDF is given by:

\[f(x)=\begin{cases}
    \frac{2-\frac{x}{2}}{6.5523}	& 0\le x<2\\
    \frac{2-(x-3)^{2}}{6.5523}		& 0\le x<3+\sqrt{2}\\
    0					& otherwise
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
likas_2()=nothing

function Distributions.cdf(::typeof(likas_2), x)
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

function Base.rand(::typeof(likas_2), n=5000)
	likas_2_cdf(x) = cdf(likas_2, x)
	[sample_from_cdf(likas_2_cdf, 2.5) for _ in 1:n]
end

Distributions.support(::typeof(likas_2)) = RealInterval(-1.0, 6.0)

const modha_and_fainman=likas_2





"""
Generates a dataset with the same parameters as used in example 3 of tests by
Likas A, 2001, "Probability density estimation using artificial neural networks"

It is a 2D uniform retangle between 0 and 0.2 on both axis

Likas used 5000 samples.

This dataset is not conditional.
"""
likas_3()=nothing

Distributions.support(::typeof(likas_3))=RectangularInterval([0., 0.],[0.2, 0.2])
Base.rand(::typeof(likas_3), n=200) = 0.2 .* rand((2, n))

end #module

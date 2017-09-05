

"""
This module contains functiosn for generating datasets suitable for testing density estimation
"""
module GenerateDatasets

using Distributions


"""
Generates a dataset with the same parameters as used in the tests of
Magdon-Ismail and Atiya, in their 1998 NIPS paper "Neural Networks for Denity Estimation.
This is a 1D GMM with two peaks, without any conditioning on any properties.
for their SIC and SLC method they tested with  n=100,
for their SIC they additionally tested with n=200

This dataset is not conditional.
"""
function magdon_ismail_and_atiya(n=200)
	comp1 = Normal(-30, 3)
	comp2 = Normal(9, 9)
	distr = MixtureModel([comp1, comp2], [0.3, 0.7])
	rand(distr, n)
end

support(::Type{typeof(magdon_ismail_and_atiya)}) = (-Inf, Inf)

"""
Generates a dataset with the same parameters as used in example 1 of tests by
Likas A, 2001, "Probability density estimation using artificial neural networks"

They only estimate it between -12 and 12

This dataset is not conditional.
"""
function likas_1(n=5000)
	distr = MixtureModel([Normal(-7,0.5), Uniform(-3,-1), Uniform(1,3), Normal(7,0.5)])
	rand(distr, n)
end

support(::Type{typeof(likas_1)})=(-12.0,12.0)

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
function likas_2(n=5000)
	function sample(v)
		#This is not quiet right

		if v<3
			# In linear part
			#Apply inverse CDF
			(6.5523)*(4+2sqrt(4-u))
		else
			# in quadratic part
			
			k1=((sqrt(9u^2-84u + 164) - 3u + 14)^(1/3)) / 2^(1/3)
			k2= (2(2^(1/3)))/(sqrt(9u^2 - 84u + 164) - 3u + 14)^(1/3)


			(6.5523)*(k1+k2+3)
		end
	end
	sample.(rand(n)
end

support(::Type{typeof(likas_2)})=(-1.0, 6.0)

const modha_and_fainman=likas_2





"""
Generates a dataset with the same parameters as used in example 3 of tests by
Likas A, 2001, "Probability density estimation using artificial neural networks"

It is a 2D uniform retangle between 0 and 0.2 on both axis

Likas used 5000 samples.

This dataset is not conditional.
"""
function likas_3(n=5000)
	0.2 .* rand((2, n))
end

support(::Type{typeof(likas_3)})=([0., 0.],[0.2, 0.2])

end #module

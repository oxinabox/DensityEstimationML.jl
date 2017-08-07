using Distributions


"""
This module contains functiosn for generating datasets suitable for testing density estimation
"""
module GenerateDatasets

function generate_dataset_latent_discrete_(means, vars)

end


"""
Generates a dataset with the same parameters as used in the tests of
Magdon-Ismail and Atiya, in thie 1998 NIPS paper "Neural Networks for Denity Estimation.
This is a 1D GMM with two peaks, without any conditioning on any properties.
for their SIC and SLC method they tested with  n=100,
for their SIC they additionally tested with n=200
"""
function gmm_magdon_ismail_and_atiya(n=200)
	comp1 = Normal(-30, 3)
	comp2 = Normal(9, 9)
	distr = MixtureModel([comp1, comp2], [0.3, 0.7])
	rand(distr, n)
end


end #module

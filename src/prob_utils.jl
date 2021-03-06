using Optim

"""
    sample_from_cdf(cdf, support_point=0.5; n_retries=10_000, tol=1e-7)

given a cdf function, `sample_from_cdf` attempts to find a sample from the distribution it represents.
This is done via numerical inversion.

 - `cdf` the cdf function
 - `support_point` some point from with in the support of the distribution. This is used as a starting point for the numericla inversion. Ideally, but not nesc, it should be inside the support
 - `tol` how close the numeric approximation of the inverse has to be to the true inverse
 - `n_retries` how many time to restart in an attempt to find 

### External Resources

 - https://sciencehouse.wordpress.com/2015/06/20/sampling-from-a-probability-distribution/
 - http://blog.quantitations.com/tutorial/2012/11/20/sampling-from-an-arbitrary-density/

"""
function sample_from_cdf(cdf, support_point::AbstractVector=[0.5]; n_retries=10_000, tol=1e-7, show_optimisation_details=false)
    u = rand()

    loss(xs) = (cdf(xs) - u).^2
    # find x such that u=inverse-cdf(x)
    # that x will be our sample
    start_point=copy(support_point)
    local result, best_x
    for _ in 1:n_retries
        result = optimize(loss, start_point, BFGS())
        
        if result.minimum < tol 
            if show_optimisation_details
                println(result)
            end
            
            best_x = result.minimizer
            return best_x
        end
        start_point=jitter(start_point, support_point) 
    end
    
    error("Failed to find an sample matching to $u in $(n_retries) attempts. Generally the solution to this is to provide good value for `support_point` that is somewhere in the center of the distribution.")
end

function sample_from_cdf(cdf, pdf, support_point::AbstractVector=[0.5]; n_retries=10_000, tol=1e-7, show_optimisation_details=false)
    u = rand()

    loss(xs) = (cdf(xs) - u).^2
    dloss!(store, xs) = store[1] = pdf(xs)*2*(cdf(xs) -u)

    # find x such that u=inverse-cdf(x)
    # that x will be our sample
    start_point=copy(support_point)
    local result, best_x
    for _ in 1:n_retries
        result = optimize(loss, dloss!, start_point, BFGS())
        
        best_x = result.minimizer
        if result.minimum < tol && pdf(best_x)>0.0
            if show_optimisation_details
                println(result)
            end
            
            return best_x
        end
        start_point=jitter(start_point, support_point)
    end
    
    error("Failed to find an sample matching to $u in $(n_retries) attempts. Generally the solution to this is to provide good value for `support_point` that is somewhere in the center of the distribution.")
end




# Start point jittering process
function jitter(cur_point, original_point)
    if rand()<0.1
        # reset it
        copy(original_point)
    else
        cur_point*4*(rand()-0.5)
        # Jitter the start point. This could be done better with a second start point to have a sense of scale
        # new value is somwehre between -2 times and 2x the previous
    end
end

# General functions for an estimator
# Does not depend on dimentionality of support

export NeuralDensityEstimator, component_weights, condition!


immutable NeuralDensityEstimator{N}
    sess::Session
    
    #Network nodes
    optimizer::Tensor
    conditioner::Tensor
    t::Tensor
    pdf::Tensor
end



function Distributions.pdf(est::NeuralDensityEstimator{1}, t::Real)
    ts = reshape([t], (1,1))
    pdf(est, ts) |> first
end

function Distributions.pdf{N}(est::NeuralDensityEstimator{N}, ts::AbstractArray{<:Real, N})
    gr = est.sess.graph
    run(est.sess, est.pdf, Dict(est.t=>ts')) |> vec
end

function Distributions.loglikelihood{N}(est::NeuralDensityEstimator{N}, ts::AbstractArray{<:Real, N})
    gr = est.sess.graph
    run(est.sess, gr["loglikelihood"], Dict(est.t=>ts'))
end



"""
    component_weights(est::NeuralDensityEstimator)

Extracts the final layer weights from the estimator.
These are similar to the mixture weights in a GMM
"""
function component_weights(est::NeuralDensityEstimator)
    node_names =  collect(keys(est.sess.graph))
    final_layer = node_names[ismatch.(r"^W_\d+_squared$",node_names)] |> sort |> last
    vec(run(est.sess, est.sess.graph[final_layer]))
end



ignore(args...) = nothing


function StatsBase.fit!(estimator::NeuralDensityEstimator, observations;
    callback=ignore, callback_vars=["ysmin", "ysmax", "loglikelihood", "working_loss"],
    epochs = 20_000)
    
    gr = estimator.sess.graph
    callback_nodes = [gr[var] for var in callback_vars]
        
        
    
    for ii in 1:epochs
        outs = run(estimator.sess,
            [callback_nodes..., estimator.optimizer],
            Dict(estimator.t=>observations'))
        
        if ii % 100 == 1
            callback(ii, Dict(zip(callback_vars, outs[1:end-1])))
        end
        
    end
    estimator
end





"""
    condition(est::NeuralDensityEstimator tol = 1e-15, max_epochs=2_000)
    
"Conditions" the neural density estimate so the support extrema are mapped to 1. and 2.
This improves training by adjusting the area the network has the learn over

"""
function condition!(est::NeuralDensityEstimator; tol = 1e-15, max_epochs=2_000)
    gr = est.sess.graph
    for ii in 1:max_epochs
        _, ysmin, ysmax, condition_loss = run(est.sess, [est.conditioner, gr["ysmin"],gr["ysmax"], gr["condition_loss"]])
        ii % 50 == 1 && @show (ii, ysmin, ysmax, condition_loss)
        if condition_loss[1] < 1e-15
            break
        end
    end
end



function NeuralDensityEstimator(prob_layer_sizes, support)
    NeuralDensityEstimator(prob_layer_sizes, minimum(support), maximum(support))
end



function NeuralDensityEstimator(prob_layer_sizes, support_min::Real, support_max::Real)
    NeuralDensityEstimator(prob_layer_sizes, SVector(support_min), SVector(support_max))
end




"""
    NeuralDensityEstimator{N}(prob_layer_sizes, support_min::NTuple{N, Real}, support_max::NTuple{N,Real})

Defines a Density Estimator.

support_min and support_max define the boundries of the support.
They should be at least as large as the support (though for noncompact supports an approximation will be required)

`support_min` defined the "lower left corner" 
and
`support_max` defines the "upper right corner"

Note the support boundries must be rectangular, if your support is not rectangular 
you must make a domain transform on your observations before using this.

"""
function NeuralDensityEstimator{N}(prob_layer_sizes, support_min::SVector{N,<:Real}, support_max::SVector{N,<:Real})
    sess = Session(Graph())
    @tf begin
        t = placeholder(Float32, shape=[N, -1])
        @assert all(support_min .< support_max)
        smin = constant(reshape(collect(support_min), (N,1)))
        smax = constant(reshape(collect(support_max), (N,1)))

        layer_sizes= [N; prob_layer_sizes; 1]
        
        network_fun_stack = Function[Base.identity]
       
        for ii in 2:length(layer_sizes)
            below_size = layer_sizes[ii-1]
            above_size = layer_sizes[ii]

            Wii = get_variable("W_$ii", [above_size, below_size], Float32)
            Wii2  = Ops.mul(Wii, Wii; name = "W_$(ii)_squared")
            
            #bii = get_variable("b_$ii", [above_size, 1], Float32)
            #act_fun = z -> nn.sigmoid(Wii2*z .+ bii)
            
            act_fun = if ii!=length(layer_sizes)
                bii = get_variable("b_$ii", [above_size, 1], Float32)
                z -> nn.sigmoid(Wii2*z .+ bii)
            else
                z-> exp(Wii2*z)
            end
            push!(network_fun_stack, z->act_fun(network_fun_stack[ii-1](z)))
        end
        
        network = network_fun_stack[end]

        
        ysmin = TensorFlow.identity(network(smin))
        ysmax = TensorFlow.identity(network(smax))
        yt = TensorFlow.identity(network(t))
        
        denominator = reduce_prod(ysmax-ysmin) #area
        numerator = TensorFlow.identity(gradients(yt,t))
        pdf = numerator/denominator
        @show pdf
#        @assert(get_shape(pdf,
        
        n_points = TensorFlow.shape(t)[2]
        loglikelihood = reduce_sum(log(numerator)) - n_points.*log(denominator)
        
        area_loss = (1f0.-denominator)^2
        working_loss = -1*loglikelihood + 0.1*area_loss
        
        optimizer = train.minimize(train.AdamOptimizer(), working_loss)
        
        
        # Conditioning
        # Make sure that ysmin~=1, and ysmax~=2
        condition_loss = (1f0 - ysmin)^2 + (2f0 - ysmax)^2
        condition_optimiser = train.minimize(train.AdamOptimizer(;name="adam_cond"), condition_loss)
    end
    
    run(sess, global_variables_initializer())
    
    NeuralDensityEstimator{N}(sess, optimizer, condition_optimiser, t, pdf)
end

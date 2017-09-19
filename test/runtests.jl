using DensityEstimationML
using Base.Test


function testset_from_file(filename)
    @testset "$filename" begin
        include(filename)
    end
end


testset_from_file.("datagen.jl")

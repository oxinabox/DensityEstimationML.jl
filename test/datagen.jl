using Base.Test
using DensityEstimationML
using Distributions

using ForwardDiff
@testset "cdf consistent with pdf $dataset" for dataset in [GenerateDatasets.Likas2()]    


    pdf_l2(x) = pdf(dataset, x)
    cdf_l2(x) = cdf(dataset, x)
    for x in 10*rand(1_000_000)-2.0
        abs(x-(3+√2))<10.0^-3 && continue
        abs(x-(2))<10.0^-3 && continue
        abs(x-0)<10.0^-3 && continue

        num = Calculus.gradient(cdf_l2, x)
        ana =  pdf_l2(x)
        diff = abs(num-ana)
        if diff>10.0^-3.0
            error("At x=$x, \t analytical pdf=$ana \t numerical pdf=$num")
        end
    end
end


@testset "Sample valid $dataset" for dataset in [GenerateDatasets.Likas1(), GenerateDatasets.Likas2(), GenerateDatasets.MagdonIsmailAndAtiya()]
    samples = original_sample(dataset)
    @test all(pdf.(dataset, samples) .> 0.0)
end

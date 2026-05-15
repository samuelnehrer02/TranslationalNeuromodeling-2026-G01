using CSV, DataFrames, Distributions, Random, StatsPlots

data_train = CSV.read("DataTrain\\train_data_long.csv", DataFrame)

p_reward_A = data_train[1:150, "P(R|A)"]
p_reward_B = data_train[1:150, "P(R|B)"]
p_reward_C = data_train[1:150, "P(R|C)"]
p_reward_D = data_train[1:150, "P(R|D)"]


##########################################################################################
################################# CREATE LOOKUP TABLE ####################################
##########################################################################################


# Pre-sample all outcomes for every trial and every possible action -> produces a
# deterministic lookup table for the Two-Step Task simulations.

function create_env_table(
    p_reward_A::AbstractVector,
    p_reward_B::AbstractVector,
    p_reward_C::AbstractVector,
    p_reward_D::AbstractVector;
    seed::Int = 1,
)
    n = length(p_reward_A)
    # 1. Seed a single RNG 
    rng = Random.MersenneTwister(seed)
    planet_if_rocket1 = Vector{Int}(undef, n)
    planet_if_rocket2 = Vector{Int}(undef, n)
    reward_A = Vector{Int}(undef, n)
    reward_B = Vector{Int}(undef, n)
    reward_C = Vector{Int}(undef, n)
    reward_D = Vector{Int}(undef, n)

    # 2. Sample all outcomes in a fixed, consistent order per trial:
    for t in 1:n
        # Stage 1: rocket 1 leads to planet 1 with prob 0.7:
        planet_if_rocket1[t] = rand(rng, Bernoulli(0.7)) ? 1 : 2
        # Stage 1: rocket 2 leads to planet 2 with prob 0.7:
        planet_if_rocket2[t] = rand(rng, Bernoulli(0.3)) ? 1 : 2
        # Stage 2: sample each alien's reward from its trial-specific probability:
        reward_A[t] = rand(rng, Bernoulli(p_reward_A[t])) ? 1 : 0
        reward_B[t] = rand(rng, Bernoulli(p_reward_B[t])) ? 1 : 0
        reward_C[t] = rand(rng, Bernoulli(p_reward_C[t])) ? 1 : 0
        reward_D[t] = rand(rng, Bernoulli(p_reward_D[t])) ? 1 : 0
    end

    return DataFrame(
        trial= collect(1:n),
        planet_if_rocket1 = planet_if_rocket1,
        planet_if_rocket2 = planet_if_rocket2,
        reward_A = reward_A,
        reward_B = reward_B,
        reward_C = reward_C,
        reward_D = reward_D,
    )
end


##########################################################################################
##################################### HEATMAP ############################################
##########################################################################################

#Visualise the environment lookup table: (trials × outcomes):
function plot_env_table(
    env_table::DataFrame)
    n = nrow(env_table)
    trials = collect(1:n)

    mat = Float64[
        Float64.(env_table.planet_if_rocket1 .== 1)';
        Float64.(env_table.planet_if_rocket2 .== 1)';
        Float64.(env_table.reward_A)';
        Float64.(env_table.reward_B)';
        Float64.(env_table.reward_C)';
        Float64.(env_table.reward_D)';
    ]
    row_labels = ["Planet|R1", "Planet|R2", "Reward A", "Reward B", "Reward C", "Reward D"]
    p = heatmap(
        trials,
        row_labels,
        mat,
        color = :Blues,
        clims = (0.0, 1.0),
        xlabel = "Trial",
        ylabel = "",
        title  = "Environment Table",
        yflip = true,
        colorbar = true,
        xticks = 0:25:n,
        size = (900, 300),
        left_margin = 10Plots.mm,
        bottom_margin = 8Plots.mm,
    )
    return p
end


##########################################################################################
################################### GENERATE TABLE #######################################
##########################################################################################

env_table = create_env_table(p_reward_A, p_reward_B, p_reward_C, p_reward_D, seed = 1)
jldsave("environment_lookup_table.jld2"; df_env = env_table)
p_table = plot_env_table(env_table)
savefig(p_table, "environment_lookup_table.png")
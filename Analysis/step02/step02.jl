##########################################################################################
####################################### SETUP ############################################
##########################################################################################
using Pkg
using Random
using JLD2, DataFrames
include("..\\step01\\step01.jl")

##########################################################################################
################################# LOAD LOOKUP TABLE ######################################
##########################################################################################
load_lookup_table = load("Analysis\\step02\\environment_lookup_table.jld2")
environment_lookup_table = load_lookup_table["df_env"]



##########################################################################################
##################################### SIMULATE ###########################################
##########################################################################################



include("..\\step02\\utils\\simulation_environment.jl")

results_m1 = DataFrame[]
results_m2 = DataFrame[]
results_m3 = DataFrame[]
results_m4 = DataFrame[]

xprob_results_m1 = DataFrame[]
xprob_results_m2 = DataFrame[]
xprob_results_m3 = DataFrame[]
xprob_results_m4 = DataFrame[]

for test in 1:100

    cognitive_model_1 = create_cognitive_model_1()
    cognitive_model_2 = create_cognitive_model_2()
    cognitive_model_3 = create_cognitive_model_3()
    cognitive_model_4 = create_cognitive_model_4()

    sim_agent_1 = init_agent(cognitive_model_1, save_history=true)
    sim_agent_2 = init_agent(cognitive_model_2, save_history=true)
    sim_agent_3 = init_agent(cognitive_model_3, save_history=true)
    sim_agent_4 = init_agent(cognitive_model_4, save_history=true)


    baseParameters = (
        β₁=rand(rng, LogNormal(24, 1)),
        β₂=rand(rng, LogNormal(24, 1)),
        p=rand(rng, LogNormal(-1, 1)),
        xprob_drift=0,
        ω₂=rand(rng, Normal(-4, 1)),
        xprob_autoconnection_strength=1,
        xprob_initial_precision=1,
        α₂=rand(rng, 6 * LogitNormal(0, 3)),
        xprob_initial_mean=0
    )

    parametersθFixed = merge(baseParameters, [:θ_T => 0.7])

    parametersS1HGF = merge(baseParameters, (
        xprob_T_initial_precision=1,
        xprob_autoconnection_strength=1,
        xprob_T_drift=0,
        ω₁=-4,
        xprob_T_initial_mean=0,
        α₁=rand(rng, 6 * LogitNormal(0, 3)),
        xprob_T_autoconnection_strength=1
    ))

    set_parameters!(sim_agent_1, parametersθFixed)
    set_parameters!(sim_agent_2, parametersS1HGF)
    set_parameters!(sim_agent_3, parametersθFixed)
    set_parameters!(sim_agent_4, parametersS1HGF)

    # Simulate
    sim_1, xprobs_1 = simulate_agent(sim_agent_1, environment_lookup_table)
    sim_2, xprobs_2 = simulate_agent(sim_agent_2, environment_lookup_table)
    sim_3, xprobs_3 = simulate_agent(sim_agent_3, environment_lookup_table)
    sim_4, xprobs_4 = simulate_agent(sim_agent_4, environment_lookup_table)

    # Save Results
    push!(results_m1, sim_1)
    push!(results_m2, sim_2)
    push!(results_m3, sim_3)
    push!(results_m4, sim_4)

    push!(xprob_results_m1, xprobs_1)
    push!(xprob_results_m2, xprobs_2)
    push!(xprob_results_m3, xprobs_3)
    push!(xprob_results_m4, xprobs_4)

end

##########################################################################################
##################################### FOR JONAS ##########################################
##########################################################################################

# 1. check ALL parameters there are in both the agent and the hgf's 
# 2. write down all the parameters that we keep fixed and their values (both hgf and agent)

# 3. lookup the documentation of Hierarchical Gaussian Filtering (HGF) and action models 
#    to check how to use set_parameters!() function. and make sure to verify
#    that the parameters are actually set correctly before running the simulation.

# also make sure to not re-use the same agent when running 100 simulations
# with different parameters sampled from the proposed priors. 

# Run the full 150-trial simulation for each model, SINGLE AGENT! per model.

##########################################################################################
##################################### PLOTTING ###########################################
##########################################################################################
using Statistics # Required for the mean() function

"""
Helper function to plot the aggregated trial data (rockets, planets, aliens, rewards) 
averaged across all simulation runs (e.g., 100 runs) for a single model.
"""
function plot_aggregate_experiment(results_array::Vector{DataFrame}, xprob_results_array::Vector{DataFrame}, title_str::String)
    n_runs = length(results_array)
    n_trials = nrow(results_array[1])
    n_updates = nrow(xprob_results_array[1])

    # Preallocate matrices to hold data across all runs
    rockets = zeros(Int, n_trials, n_runs)
    planets = zeros(Int, n_trials, n_runs)
    aliens = zeros(Int, n_trials, n_runs)
    rewards = zeros(Int, n_trials, n_runs)
    xprob_T = zeros(Real, n_updates, n_runs)
    xprob_A = zeros(Real, n_updates, n_runs)
    xprob_B = zeros(Real, n_updates, n_runs)
    xprob_C = zeros(Real, n_updates, n_runs)
    xprob_D = zeros(Real, n_updates, n_runs)


    alien_map = Dict("A" => 1, "B" => 2, "C" => 3, "D" => 4)

    # Extract data from all runs
    for (i, df) in enumerate(results_array)
        rockets[:, i] = df.rocket
        planets[:, i] = df.planet
        aliens[:, i] = [haskey(alien_map, a) ? alien_map[a] : 0 for a in df.alien]
        rewards[:, i] = df.reward
    end

    for (i, df) in enumerate(xprob_results_array)
        xprob_T[:, i] = df.xprob_T
        xprob_A[:, i] = df.xprob_A
        xprob_B[:, i] = df.xprob_B
        xprob_C[:, i] = df.xprob_C
        xprob_D[:, i] = df.xprob_D
    end


    # Calculate trial-by-trial proportions/averages across the 100 runs
    prop_rocket1 = mean(rockets .== 1, dims=2)
    prop_planet1 = mean(planets .== 1, dims=2)

    prop_alienA = mean(aliens .== 1, dims=2)
    prop_alienB = mean(aliens .== 2, dims=2)
    prop_alienC = mean(aliens .== 3, dims=2)
    prop_alienD = mean(aliens .== 4, dims=2)

    mean_reward = mean(rewards, dims=2)

    mean_xprob_T = mean(xprob_T, dims=2)
    mean_xprob_A = mean(xprob_A, dims=2)
    mean_xprob_B = mean(xprob_B, dims=2)
    mean_xprob_C = mean(xprob_C, dims=2)
    mean_xprob_D = mean(xprob_D, dims=2)


    trials = 1:n_trials
    updates = 1:n_updates

    # Create subplots for each variable
    p1 = plot(trials, prop_rocket1, label="P(Rocket=1)", title="Rocket Choice Proportion",
        ylim=(0, 1), legend=:topright, color=:blue, ylabel="Proportion")

    p2 = plot(trials, prop_planet1, label="P(Planet=1)", title="Planet State Proportion",
        ylim=(0, 1), legend=:topright, color=:green, ylabel="Proportion")

    p3 = plot(trials, prop_alienA, label="A", title="Alien State Proportion",
        ylim=(0, 1), legend=:outertopright, ylabel="Proportion")
    plot!(p3, trials, prop_alienB, label="B")
    plot!(p3, trials, prop_alienC, label="C")
    plot!(p3, trials, prop_alienD, label="D")

    p4 = plot(trials, mean_reward, label="Mean Reward", title="Average Reward Rate",
        ylim=(0, 1), legend=:topright, color=:black, xlabel="Trial", ylabel="Reward Rate")


    p5 = plot(updates, mean_xprob_T, label="xprob_T", title="Rocket HGF Internal State",
        legend=:topright, color=:blue, ylabel="Mean")
    p6 = plot(updates, mean_xprob_A, label="xprob_A", title="Alien A HGF Internal State",
        legend=:topright, color=:blue, ylabel="Mean")
    p7 = plot(updates, mean_xprob_B, label="xprob_B", title="Alien B HGF Internal State",
        legend=:topright, color=:blue, ylabel="Mean")
    p8 = plot(updates, mean_xprob_C, label="xprob_C", title="Alien C HGF Internal State",
        legend=:topright, color=:blue, ylabel="Mean")
    p9 = plot(updates, mean_xprob_D, label="xprob_D", title="Alien D HGF Internal State",
        legend=:topright, color=:blue, ylabel="Mean")

    # Combine into a 4-row layout
    #           a   b   c   d   e   f   g   h   i
    return plot(p1, p2, p3, p4, p5, p6, p7, p8, p9,
        layout=@layout([e; a f; b g; c h; d i]),
        size=(1800, 1000),
        plot_title=title_str,
        margin=5Plots.mm)
end

# Generate aggregate plots for each model
plt1 = plot_aggregate_experiment(results_m1, xprob_results_m1, "Model 1: 100 Runs Aggregated")
plt2 = plot_aggregate_experiment(results_m2, xprob_results_m2, "Model 2: 100 Runs Aggregated")
plt3 = plot_aggregate_experiment(results_m3, xprob_results_m3, "Model 3: 100 Runs Aggregated")
plt4 = plot_aggregate_experiment(results_m4, xprob_results_m4, "Model 4: 100 Runs Aggregated")

display(plt1)
display(plt2)
display(plt3)
display(plt4)

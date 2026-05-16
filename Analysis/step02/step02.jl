##########################################################################################
####################################### SETUP ############################################
##########################################################################################
using Pkg
using Random
using JLD2, DataFrames, Plots, Statistics
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

agents_m1 = Any[]
agents_m2 = Any[]
agents_m3 = Any[]
agents_m4 = Any[]

# Parameter tracking for sensitivity analysis
param_tracking_m1 = Dict(:β₁ => [], :β₂ => [], :p => [], :ω₂ => [], :α₂ => [])
param_tracking_m2 = Dict(:β₁ => [], :β₂ => [], :p => [], :ω₂ => [], :α₁ => [], :α₂ => [])
param_tracking_m3 = Dict(:β₁ => [], :β₂ => [], :p => [], :ω₂ => [], :α₂ => [])
param_tracking_m4 = Dict(:β₁ => [], :β₂ => [], :p => [], :ω₂ => [], :α₁ => [], :α₂ => [])

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

    # Track parameters for sensitivity analysis
    push!(param_tracking_m1[:β₁], baseParameters.β₁)
    push!(param_tracking_m1[:β₂], baseParameters.β₂)
    push!(param_tracking_m1[:p], baseParameters.p)
    push!(param_tracking_m1[:ω₂], baseParameters.ω₂)
    push!(param_tracking_m1[:α₂], baseParameters.α₂)

    push!(param_tracking_m2[:β₁], baseParameters.β₁)
    push!(param_tracking_m2[:β₂], baseParameters.β₂)
    push!(param_tracking_m2[:p], baseParameters.p)
    push!(param_tracking_m2[:ω₂], baseParameters.ω₂)
    push!(param_tracking_m2[:α₂], baseParameters.α₂)

    push!(param_tracking_m3[:β₁], baseParameters.β₁)
    push!(param_tracking_m3[:β₂], baseParameters.β₂)
    push!(param_tracking_m3[:p], baseParameters.p)
    push!(param_tracking_m3[:ω₂], baseParameters.ω₂)
    push!(param_tracking_m3[:α₂], baseParameters.α₂)

    push!(param_tracking_m4[:β₁], baseParameters.β₁)
    push!(param_tracking_m4[:β₂], baseParameters.β₂)
    push!(param_tracking_m4[:p], baseParameters.p)
    push!(param_tracking_m4[:ω₂], baseParameters.ω₂)
    push!(param_tracking_m4[:α₂], baseParameters.α₂)

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

    # Track Model 2/4 specific parameters
    push!(param_tracking_m2[:α₁], parametersS1HGF.α₁)
    push!(param_tracking_m4[:α₁], parametersS1HGF.α₁)

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

    # Save Agent Objects
    push!(agents_m1, sim_agent_1)
    push!(agents_m2, sim_agent_2)
    push!(agents_m3, sim_agent_3)
    push!(agents_m4, sim_agent_4)

end

# Extract min/max values for sensitivity analysis
param_ranges_m1 = Dict(:β₁ => (minimum(param_tracking_m1[:β₁]), maximum(param_tracking_m1[:β₁])),
    :β₂ => (minimum(param_tracking_m1[:β₂]), maximum(param_tracking_m1[:β₂])),
    :p => (minimum(param_tracking_m1[:p]), maximum(param_tracking_m1[:p])),
    :ω₂ => (minimum(param_tracking_m1[:ω₂]), maximum(param_tracking_m1[:ω₂])),
    :α₂ => (minimum(param_tracking_m1[:α₂]), maximum(param_tracking_m1[:α₂])))

param_ranges_m2 = Dict(:β₁ => (minimum(param_tracking_m2[:β₁]), maximum(param_tracking_m2[:β₁])),
    :β₂ => (minimum(param_tracking_m2[:β₂]), maximum(param_tracking_m2[:β₂])),
    :p => (minimum(param_tracking_m2[:p]), maximum(param_tracking_m2[:p])),
    :ω₂ => (minimum(param_tracking_m2[:ω₂]), maximum(param_tracking_m2[:ω₂])),
    :α₁ => (minimum(param_tracking_m2[:α₁]), maximum(param_tracking_m2[:α₁])),
    :α₂ => (minimum(param_tracking_m2[:α₂]), maximum(param_tracking_m2[:α₂])))

param_ranges_m3 = Dict(:β₁ => (minimum(param_tracking_m3[:β₁]), maximum(param_tracking_m3[:β₁])),
    :β₂ => (minimum(param_tracking_m3[:β₂]), maximum(param_tracking_m3[:β₂])),
    :p => (minimum(param_tracking_m3[:p]), maximum(param_tracking_m3[:p])),
    :ω₂ => (minimum(param_tracking_m3[:ω₂]), maximum(param_tracking_m3[:ω₂])),
    :α₂ => (minimum(param_tracking_m3[:α₂]), maximum(param_tracking_m3[:α₂])))

param_ranges_m4 = Dict(:β₁ => (minimum(param_tracking_m4[:β₁]), maximum(param_tracking_m4[:β₁])),
    :β₂ => (minimum(param_tracking_m4[:β₂]), maximum(param_tracking_m4[:β₂])),
    :p => (minimum(param_tracking_m4[:p]), maximum(param_tracking_m4[:p])),
    :ω₂ => (minimum(param_tracking_m4[:ω₂]), maximum(param_tracking_m4[:ω₂])),
    :α₁ => (minimum(param_tracking_m4[:α₁]), maximum(param_tracking_m4[:α₁])),
    :α₂ => (minimum(param_tracking_m4[:α₂]), maximum(param_tracking_m4[:α₂])))
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

"""
averaged across all simulation runs (e.g., 100 runs) for a single model.
Optional sensitivity_data parameter overlays min/max trajectories for parameter sensitivity.
"""
function plot_aggregate_experiment(results_array::Vector{DataFrame}, xprob_results_array::Vector{DataFrame}, title_str::String, sensitivity_data::Dict=Dict())
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

    # Process sensitivity data if provided
    sensitivity_overlays = Dict()
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :dash]
    if !isempty(sensitivity_data)
        for (param_idx, (param_name, (min_data_tuple, max_data_tuple))) in enumerate(sensitivity_data)
            min_results, min_xprobs = min_data_tuple
            max_results, max_xprobs = max_data_tuple
            min_n = length(min_results)
            max_n = length(max_results)

            # Extract and process min results
            min_rockets = zeros(Int, n_trials, min_n)
            min_planets = zeros(Int, n_trials, min_n)
            min_aliens = zeros(Int, n_trials, min_n)
            min_rewards = zeros(Int, n_trials, min_n)
            min_xprob_T = zeros(Real, n_updates, min_n)
            min_xprob_A = zeros(Real, n_updates, min_n)
            min_xprob_B = zeros(Real, n_updates, min_n)
            min_xprob_C = zeros(Real, n_updates, min_n)
            min_xprob_D = zeros(Real, n_updates, min_n)

            for (i, (df_result, df_xprob)) in enumerate(zip(min_results, min_xprobs))
                min_rockets[:, i] = df_result.rocket
                min_planets[:, i] = df_result.planet
                min_aliens[:, i] = [haskey(alien_map, a) ? alien_map[a] : 0 for a in df_result.alien]
                min_rewards[:, i] = df_result.reward
                min_xprob_T[:, i] = df_xprob.xprob_T
                min_xprob_A[:, i] = df_xprob.xprob_A
                min_xprob_B[:, i] = df_xprob.xprob_B
                min_xprob_C[:, i] = df_xprob.xprob_C
                min_xprob_D[:, i] = df_xprob.xprob_D
            end

            # Extract and process max results
            max_rockets = zeros(Int, n_trials, max_n)
            max_planets = zeros(Int, n_trials, max_n)
            max_aliens = zeros(Int, n_trials, max_n)
            max_rewards = zeros(Int, n_trials, max_n)
            max_xprob_T = zeros(Real, n_updates, max_n)
            max_xprob_A = zeros(Real, n_updates, max_n)
            max_xprob_B = zeros(Real, n_updates, max_n)
            max_xprob_C = zeros(Real, n_updates, max_n)
            max_xprob_D = zeros(Real, n_updates, max_n)

            for (i, (df_result, df_xprob)) in enumerate(zip(max_results, max_xprobs))
                max_rockets[:, i] = df_result.rocket
                max_planets[:, i] = df_result.planet
                max_aliens[:, i] = [haskey(alien_map, a) ? alien_map[a] : 0 for a in df_result.alien]
                max_rewards[:, i] = df_result.reward
                max_xprob_T[:, i] = df_xprob.xprob_T
                max_xprob_A[:, i] = df_xprob.xprob_A
                max_xprob_B[:, i] = df_xprob.xprob_B
                max_xprob_C[:, i] = df_xprob.xprob_C
                max_xprob_D[:, i] = df_xprob.xprob_D
            end

            # Compute averages for min and max
            min_overlay = (
                prop_rocket1_min=mean(min_rockets .== 1, dims=2),
                prop_planet1_min=mean(min_planets .== 1, dims=2),
                prop_alienA_min=mean(min_aliens .== 1, dims=2),
                prop_alienB_min=mean(min_aliens .== 2, dims=2),
                prop_alienC_min=mean(min_aliens .== 3, dims=2),
                prop_alienD_min=mean(min_aliens .== 4, dims=2),
                mean_reward_min=mean(min_rewards, dims=2),
                mean_xprob_T_min=mean(min_xprob_T, dims=2),
                mean_xprob_A_min=mean(min_xprob_A, dims=2),
                mean_xprob_B_min=mean(min_xprob_B, dims=2),
                mean_xprob_C_min=mean(min_xprob_C, dims=2),
                mean_xprob_D_min=mean(min_xprob_D, dims=2)
            )

            max_overlay = (
                prop_rocket1_max=mean(max_rockets .== 1, dims=2),
                prop_planet1_max=mean(max_planets .== 1, dims=2),
                prop_alienA_max=mean(max_aliens .== 1, dims=2),
                prop_alienB_max=mean(max_aliens .== 2, dims=2),
                prop_alienC_max=mean(max_aliens .== 3, dims=2),
                prop_alienD_max=mean(max_aliens .== 4, dims=2),
                mean_reward_max=mean(max_rewards, dims=2),
                mean_xprob_T_max=mean(max_xprob_T, dims=2),
                mean_xprob_A_max=mean(max_xprob_A, dims=2),
                mean_xprob_B_max=mean(max_xprob_B, dims=2),
                mean_xprob_C_max=mean(max_xprob_C, dims=2),
                mean_xprob_D_max=mean(max_xprob_D, dims=2)
            )

            sensitivity_overlays[param_name] = (min_overlay, max_overlay, line_styles[param_idx])
        end
    end


    trials = 1:n_trials
    updates = 1:n_updates

    # Create subplots for each variable
    p1 = plot(trials, prop_rocket1, label="Base", title="Rocket Choice Proportion",
        ylim=(0, 1), legend=:topright, color=:black, ylabel="Proportion", linewidth=2)

    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p1, trials, min_data.prop_rocket1_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p1, trials, max_data.prop_rocket1_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p2 = plot(trials, prop_planet1, label="Base", title="Planet State Proportion",
        ylim=(0, 1), legend=:topright, color=:green, ylabel="Proportion", linewidth=2)

    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p2, trials, min_data.prop_planet1_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p2, trials, max_data.prop_planet1_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p3a = plot(trials, prop_alienA, label="Base", title="Alien A Proportion",
        ylim=(0, 1), legend=:topright, color=:black, ylabel="Proportion", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p3a, trials, min_data.prop_alienA_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p3a, trials, max_data.prop_alienA_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p3b = plot(trials, prop_alienB, label="Base", title="Alien B Proportion",
        ylim=(0, 1), legend=:topright, color=:black, ylabel="Proportion", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p3b, trials, min_data.prop_alienB_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p3b, trials, max_data.prop_alienB_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p3c = plot(trials, prop_alienC, label="Base", title="Alien C Proportion",
        ylim=(0, 1), legend=:topright, color=:black, ylabel="Proportion", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p3c, trials, min_data.prop_alienC_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p3c, trials, max_data.prop_alienC_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p3d = plot(trials, prop_alienD, label="Base", title="Alien D Proportion",
        ylim=(0, 1), legend=:topright, color=:black, ylabel="Proportion", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p3d, trials, min_data.prop_alienD_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p3d, trials, max_data.prop_alienD_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p4 = plot(trials, mean_reward, label="Base", title="Average Reward Rate",
        ylim=(0, 1), legend=:topright, color=:black, xlabel="Trial", ylabel="Reward Rate", linewidth=2)

    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p4, trials, min_data.mean_reward_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p4, trials, max_data.mean_reward_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p5 = plot(updates, mean_xprob_T, label="Base", title="Rocket HGF Internal State",
        legend=:topright, color=:black, ylabel="Mean", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p5, updates, min_data.mean_xprob_T_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p5, updates, max_data.mean_xprob_T_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p6 = plot(updates, mean_xprob_A, label="Base", title="Alien A HGF Internal State",
        legend=:topright, color=:black, ylabel="Mean", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p6, updates, min_data.mean_xprob_A_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p6, updates, max_data.mean_xprob_A_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p7 = plot(updates, mean_xprob_B, label="Base", title="Alien B HGF Internal State",
        legend=:topright, color=:black, ylabel="Mean", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p7, updates, min_data.mean_xprob_B_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p7, updates, max_data.mean_xprob_B_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p8 = plot(updates, mean_xprob_C, label="Base", title="Alien C HGF Internal State",
        legend=:topright, color=:black, ylabel="Mean", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p8, updates, min_data.mean_xprob_C_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p8, updates, max_data.mean_xprob_C_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    p9 = plot(updates, mean_xprob_D, label="Base", title="Alien D HGF Internal State",
        legend=:topright, color=:black, ylabel="Mean", linewidth=2)
    if !isempty(sensitivity_overlays)
        for (param_name, (min_data, max_data, line_style)) in sensitivity_overlays
            plot!(p9, updates, min_data.mean_xprob_D_min, label="$(param_name) min",
                color=:blue, linestyle=line_style, linewidth=1.5, alpha=0.7)
            plot!(p9, updates, max_data.mean_xprob_D_max, label="$(param_name) max",
                color=:red, linestyle=line_style, linewidth=1.5, alpha=0.7)
        end
    end

    # Combine into 3×4 layout for better readability
    # Row 1: Rocket, Planet, Reward, xprob_T
    # Row 2: Alien A, B, C, D proportions
    # Row 3: xprob_A, B, C, D internal states
    return plot(
        p1, p3a, p6,
        p2, p3b, p7,
        p4, p3c, p8,
        p5, p3d, p9,
        layout=(4, 3),
        size=(3600, 1800),
        plot_title=title_str,
        margin=5Plots.mm,
        suptitle=title_str)
end

"""
Helper function to plot overlaid agent trajectories from 4 agents, separated by xprob node.
Each plot shows 4 trajectories overlaid with mean and variance.
"""
function plot_individual_agents(agents::Vector, title_str::String)
    n_agents = min(length(agents), 4)
    colors = [:blue, :red, :green, :purple, :orange, :grey,]
    node_names = [:xprob_A, :xprob_B, :xprob_C, :xprob_D, :xprob_T]
    plots_array = []

    for node in node_names
        node_title = String(node)
        p = plot(legend=:topright, title=node_title, xlabel="Update", ylabel="Mean")

        for agent_idx in 1:n_agents
            agent = agents[agent_idx]
            try
                plot!(p, agent, String(node), label="Agent $agent_idx",
                    color=colors[agent_idx], linewidth=2, alpha=0.7)
            catch
                # Fallback: extract and plot mean data directly if plot(agent, node) fails
                node_symbol = node
                if node_symbol == :xprob_T && hasproperty(agent.model_attributes.parameters, :θ_T)
                    mean_data = fill(agent.model_attributes.parameters.θ_T.value,
                        length(agent.history.xprob_A_prediction_mean))
                else
                    # Convert symbol to string for field access
                    field_name = Symbol(String(node_symbol) * "_prediction_mean")
                    mean_data = getproperty(agent.history, field_name)
                end
                updates = 1:length(mean_data)
                plot!(p, updates, mean_data, label="Agent $agent_idx",
                    color=colors[agent_idx], linewidth=2, alpha=0.7)
            end
        end
        # Restore title in case it was overwritten by agent plotting recipe
        plot!(p, title=node_title)
        push!(plots_array, p)
    end

    return plot(plots_array...,
        layout=(2, 3),
        size=(1400, 800),
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

# Select 4 random agents from each model and create individual trajectory plots
selected_indices = rand(rng, 1:100, 4)

agents_selected_m1 = agents_m1[selected_indices]
agents_selected_m2 = agents_m2[selected_indices]
agents_selected_m3 = agents_m3[selected_indices]
agents_selected_m4 = agents_m4[selected_indices]

# Generate individual agent trajectory plots
plt_agents_1 = plot_individual_agents(agents_selected_m1, "Model 1: Individual Agent Trajectories")
plt_agents_2 = plot_individual_agents(agents_selected_m2, "Model 2: Individual Agent Trajectories")
plt_agents_3 = plot_individual_agents(agents_selected_m3, "Model 3: Individual Agent Trajectories")
plt_agents_4 = plot_individual_agents(agents_selected_m4, "Model 4: Individual Agent Trajectories")

display(plt_agents_1)
display(plt_agents_2)
display(plt_agents_3)
display(plt_agents_4)

##########################################################################################
######################### PARAMETER SENSITIVITY ANALYSIS #################################
##########################################################################################

"""
Run sensitivity analysis: 20 simulations with a parameter fixed to a specific value,
all other parameters sampled from their priors.
Returns tuple of (results_array, xprob_results_array)
"""
function run_sensitivity_simulations(model_creator, fixed_param::Symbol, fixed_value::Real, n_sims::Int=20)
    sens_results = DataFrame[]
    sens_xprobs = DataFrame[]

    for run in 1:n_sims
        model = model_creator()
        agent = init_agent(model, save_history=true)

        # Sample base parameters
        base_params = (
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

        # If Model 2 or 4 (learned T), add S1HGF params
        if model_creator in [create_cognitive_model_2, create_cognitive_model_4]
            params = merge(base_params, (
                xprob_T_initial_precision=1,
                xprob_autoconnection_strength=1,
                xprob_T_drift=0,
                ω₁=-4,
                xprob_T_initial_mean=0,
                α₁=rand(rng, 6 * LogitNormal(0, 3)),
                xprob_T_autoconnection_strength=1
            ))
        else  # Model 1 or 3 (fixed T)
            params = merge(base_params, [:θ_T => 0.7])
        end

        # Override the fixed parameter
        params = merge(params, Dict(fixed_param => fixed_value))
        set_parameters!(agent, params)

        # Simulate
        result, xprobs = simulate_agent(agent, environment_lookup_table)
        push!(sens_results, result)
        push!(sens_xprobs, xprobs)
    end

    return (sens_results, sens_xprobs)
end

# Run sensitivity analysis for each model
println("Running sensitivity analysis...")

# Model 1 sensitivity analysis
println("  Model 1...")
sens_data_m1 = Dict()
for param_name in keys(param_ranges_m1)
    min_val, max_val = param_ranges_m1[param_name]
    if abs(max_val - min_val) > 1e-6  # Only analyze if there's variation
        min_results, min_xprobs = run_sensitivity_simulations(create_cognitive_model_1, param_name, min_val)
        max_results, max_xprobs = run_sensitivity_simulations(create_cognitive_model_1, param_name, max_val)
        sens_data_m1[param_name] = (
            (min_results, min_xprobs),
            (max_results, max_xprobs)
        )
    end
end

# Model 2 sensitivity analysis
println("  Model 2...")
sens_data_m2 = Dict()
for param_name in keys(param_ranges_m2)
    min_val, max_val = param_ranges_m2[param_name]
    if abs(max_val - min_val) > 1e-6
        min_results, min_xprobs = run_sensitivity_simulations(create_cognitive_model_2, param_name, min_val)
        max_results, max_xprobs = run_sensitivity_simulations(create_cognitive_model_2, param_name, max_val)
        sens_data_m2[param_name] = (
            (min_results, min_xprobs),
            (max_results, max_xprobs)
        )
    end
end

# Model 3 sensitivity analysis
println("  Model 3...")
sens_data_m3 = Dict()
for param_name in keys(param_ranges_m3)
    min_val, max_val = param_ranges_m3[param_name]
    if abs(max_val - min_val) > 1e-6
        min_results, min_xprobs = run_sensitivity_simulations(create_cognitive_model_3, param_name, min_val)
        max_results, max_xprobs = run_sensitivity_simulations(create_cognitive_model_3, param_name, max_val)
        sens_data_m3[param_name] = (
            (min_results, min_xprobs),
            (max_results, max_xprobs)
        )
    end
end

# Model 4 sensitivity analysis
println("  Model 4...")
sens_data_m4 = Dict()
for param_name in keys(param_ranges_m4)
    min_val, max_val = param_ranges_m4[param_name]
    if abs(max_val - min_val) > 1e-6
        min_results, min_xprobs = run_sensitivity_simulations(create_cognitive_model_4, param_name, min_val)
        max_results, max_xprobs = run_sensitivity_simulations(create_cognitive_model_4, param_name, max_val)
        sens_data_m4[param_name] = (
            (min_results, min_xprobs),
            (max_results, max_xprobs)
        )
    end
end

plt1_sens = plot_aggregate_experiment(results_m1, xprob_results_m1, "Model 1: 100 Runs + Sensitivity", sens_data_m1)
plt2_sens = plot_aggregate_experiment(results_m2, xprob_results_m2, "Model 2: 100 Runs + Sensitivity", sens_data_m2)
plt3_sens = plot_aggregate_experiment(results_m3, xprob_results_m3, "Model 3: 100 Runs + Sensitivity", sens_data_m3)
plt4_sens = plot_aggregate_experiment(results_m4, xprob_results_m4, "Model 4: 100 Runs + Sensitivity", sens_data_m4)

display(plt1_sens)
display(plt2_sens)
display(plt3_sens)
display(plt4_sens)

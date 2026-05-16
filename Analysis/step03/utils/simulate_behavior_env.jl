##########################################################################################
################################# SIMULATION FUNCTION #####################################
##########################################################################################
using DataFrames, Distributions, Random

# Sample N parameter vectors, simulate one fresh agent per vector through the shared
# environment lookup table, and return everything.
# Returns
# NamedTuple with fields:
#     .behavior   — DataFrame(ID, action, observation, choice) [(2n-1) × n_participants rows]
#     .parameters — DataFrame(ID, <one column per parameter>) [n_participants rows]

function simulate_synthetic_dataset(;
    create_model_fn,
    env_table::DataFrame,
    n_participants::Int,
    fixed_parameters::NamedTuple = NamedTuple(),
    param_distributions::NamedTuple,
    seed::Int = 1,
)
    # 1. Seed the global RNG once: all draws (params + actions) share this stream:
    Random.seed!(seed)

    n = nrow(env_table)
    param_names = keys(param_distributions)

    # 2. Sample all parameter vectors up front:
    sampled_params = [
        NamedTuple{param_names}(
            Tuple(rand(getfield(param_distributions, k)) for k in param_names)
        )
        for _ in 1:n_participants
    ]

    # 3. Build parameters DataFrame:
    param_df = DataFrame(ID = string.(1:n_participants))
    for k in param_names
        param_df[!, k] = [getfield(sampled_params[i], k) for i in 1:n_participants]
    end

    # 4. Simulate each participant:
    all_behavior = DataFrame[]

    for i in 1:n_participants

        # Fresh model and agent 
        cognitive_model = create_model_fn()
        agent           = init_agent(cognitive_model, save_history = true)
        set_parameters!(agent, sampled_params[i])
        if length(fixed_parameters) > 0
            set_parameters!(agent, fixed_parameters)
        end
        # --- Simulation loop  ---
        rockets = Vector{Int}(undef, n)
        planets = Vector{Int}(undef, n)
        aliens  = Vector{String}(undef, n)
        rewards = Vector{Int}(undef, n)

        rocket = rand(1:2)
        for t in 1:n
            rockets[t] = rocket
            planet = rocket == 1 ? env_table.planet_if_rocket1[t] : env_table.planet_if_rocket2[t]
            planets[t] = planet
            alien_idx  = observe!(agent, (action = rocket, observation = planet))
            alien = planet == 1 ? (alien_idx == 1 ? "A" : "B") : (alien_idx == 1 ? "C" : "D")
            aliens[t]  = alien
            reward = env_table[t, Symbol("reward_" * alien)]
            rewards[t] = reward
            rocket = observe!(agent, (action = alien, observation = reward))
        end

        # --- Convert to per-stage format (2n - 1 rows) ---
        n_rows = 2 * n - 1
        id_col = fill(string(i), n_rows)
        act_col = Vector{Union{Int, String}}(undef, n_rows)
        obs_col = Vector{Int}(undef, n_rows)
        cho_col = Vector{Int}(undef, n_rows)

        row = 1
        for t in 1:n
            # Stage 2 of trial t: agent observed (rocket, planet) and chose alien
            act_col[row] = rockets[t]
            obs_col[row] = planets[t]
            cho_col[row] = aliens[t] in ("A", "C") ? 1 : 2
            row += 1

            if t < n
                # Stage 1 of trial t+1: agent observed (alien, reward) and chose next rocket
                act_col[row] = aliens[t]
                obs_col[row] = rewards[t]
                cho_col[row] = rockets[t + 1]
                row += 1
            end
        end

        push!(all_behavior, DataFrame(
            ID   = id_col,
            action   = act_col,
            observation = obs_col,
            choice = cho_col,
        ))

    end

    behavior_df = vcat(all_behavior...)

    return (behavior = behavior_df, parameters = param_df)
end

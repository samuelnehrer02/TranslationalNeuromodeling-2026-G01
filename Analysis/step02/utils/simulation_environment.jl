##########################################################################################
############################### SIMULATION ENVIRONMENT ###################################
##########################################################################################

# Runs !one! agent through all trials of the Two-Step Task using the pre-sampled lookup table.
# Returns a DataFrame recording every action and observation in the format the agents expect.
#
# Trial flow:
#   Stage 1: agent picks rocket  → env looks up planet  → agent observes (rocket, planet)
#                                                        → agent returns alien choice (1 or 2)
#   Stage 2: agent picks alien   → env looks up reward  → agent observes (alien,  reward)
#                                                        → agent returns next-trial rocket (1 or 2)

function simulate_agent(agent, env_table::DataFrame)
    n = nrow(env_table)

    rockets = Vector{Int}(undef, n)
    planets = Vector{Int}(undef, n)
    aliens  = Vector{String}(undef, n)
    rewards = Vector{Int}(undef, n)

    rocket = 1  # initial rocket for trial 1

    for t in 1:n

        # --- Stage 1 ---
        # Record this trial's rocket choice:
        rockets[t] = rocket
        # Look up planet from the pre-sampled table:
        planet = rocket == 1 ? env_table.planet_if_rocket1[t] : env_table.planet_if_rocket2[t]
        planets[t] = planet
        # Feed transition to agent; receive alien choice (1 or 2, local to the planet):
        alien_idx = observe!(agent, (action = rocket, observation = planet))
        # Convert local alien index to global alien label:
        alien = if planet == 1
            alien_idx == 1 ? "A" : "B"
        else
            alien_idx == 1 ? "C" : "D"
        end
        aliens[t] = alien

        # --- Stage 2 ---
        # Look up reward from the pre-sampled table:
        reward    = env_table[t, Symbol("reward_" * alien)]
        rewards[t] = reward
        # Feed reward to agent; receive next trial's rocket choice:
        rocket = observe!(agent, (action = alien, observation = reward))

    end

    return DataFrame(
        trial  = collect(1:n),
        rocket = rockets,
        planet = planets,
        alien  = aliens,
        reward = rewards,
    )
end


names(train_data)
p_A = train_data[1:150,"P(R|A)"]
p_B = train_data[1:150,"P(R|B)"]
p_C = train_data[1:150,"P(R|C)"]
p_D = train_data[1:150,"P(R|D)"]



###################################################### Playground ######################################################
using Distributions
using Random

mutable struct TwoStepEnv
    n_trials::Int
    current_trial::Int

    # transition_lookup[t][rocket] -> planet
    transition_lookup::Vector{Dict{Int,Int}}

    # reward_lookup[t][alien] -> reward
    reward_lookup::Vector{Dict{String,Int}}
end

function create_environment(
    p_A,
    p_B,
    p_C,
    p_D;
    seed = 42,
)
    rng = Random.MersenneTwister(seed)

    n_trials = length(p_A)

    transition_lookup = Vector{Dict{Int,Int}}(undef, n_trials)
    reward_lookup     = Vector{Dict{String,Int}}(undef, n_trials)

    for t in 1:n_trials

        ##################################################
        # Pre-sample alien rewards
        ##################################################

        rA = rand(rng, Bernoulli(p_A[t]))
        rB = rand(rng, Bernoulli(p_B[t]))
        rC = rand(rng, Bernoulli(p_C[t]))
        rD = rand(rng, Bernoulli(p_D[t]))

        reward_lookup[t] = Dict(
            "A" => rA,
            "B" => rB,
            "C" => rC,
            "D" => rD,
        )

        ##################################################
        # Pre-sample rocket transitions
        ##################################################
        #
        # Rocket 1:
        #   70% -> planet 1
        #   30% -> planet 2
        #
        # Rocket 2:
        #   30% -> planet 1
        #   70% -> planet 2
        #
        ##################################################

        planet_if_r1 = rand(rng, Bernoulli(0.7)) == 1 ? 1 : 2
        planet_if_r2 = rand(rng, Bernoulli(0.3)) == 1 ? 1 : 2

        transition_lookup[t] = Dict(
            1 => planet_if_r1,
            2 => planet_if_r2,
        )
    end

    return TwoStepEnv(
        n_trials,
        1,
        transition_lookup,
        reward_lookup,
    )
end

function step_stage1!(env::TwoStepEnv, rocket::Int)

    t = env.current_trial

    planet = env.transition_lookup[t][rocket]

    return planet
end

function available_aliens(planet::Int)

    if planet == 1
        return ["A", "B"]
    else
        return ["C", "D"]
    end
end

function step_stage2!(env::TwoStepEnv, alien::String)

    t = env.current_trial

    reward = env.reward_lookup[t][alien]

    env.current_trial += 1

    return reward
end

function reset!(env::TwoStepEnv)
    env.current_trial = 1
end





env = create_environment(p_A, p_B, p_C, p_D, seed = 42)
cognititive_model_2 = create_cognitive_model_2()
agent_model_2 = init_agent(cognititive_model_2, save_history=true)
set_parameters!(agent_model_2, (; ω₁ = -10))
set_parameters!(agent_model_2, (; α₁ = 0.5))


rocket = 1

for trial in 1:150

    planet = step_stage1!(env, rocket)

    # Feed transition observation in;
    # receive alien choice back.
    alien_idx = observe!(
        agent_model_2,
        (
            action = rocket,
            observation = planet,
        )
    )

    ########################################
    # Convert alien index to actual alien
    ########################################

    alien =
        if planet == 1
            alien_idx == 1 ? "A" : "B"
        else
            alien_idx == 1 ? "C" : "D"
        end
    ########################################
    # Stage 2
    ########################################

    reward = step_stage2!(env, alien)

    # Feed reward observation in;
    # receive NEXT trial rocket choice back.
    rocket = observe!(
        agent_model_2,
        (
            action = alien,
            observation = reward,
        )
    )
end

plot(agent_model_2.model_attributes.submodel, ("xbin_T", "prediction_mean"))

plot(agent_model_2.model_attributes.submodel, ("xbin_A", "prediction_mean"))
plot!(agent_model_2.model_attributes.submodel, ("xbin_B", "prediction_mean"))
plot!(agent_model_2.model_attributes.submodel, ("xbin_C", "prediction_mean"))
plot!(agent_model_2.model_attributes.submodel, ("xbin_D", "prediction_mean"))

agent_model_2
###########################################################################################################################
################################## EXAMPLE HOW TO CALCULATE BIC FOR A SINGLE PARTICIPANT ##################################
###########################################################################################################################
using Pkg
using JLD2, DataFrames

# Lets try what model comparison metric we can reliably obtain:
# Include models from step01
include("step01\\step01.jl")

synthetic_dataset_model_1 = load("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_1.jld2")["synthetic_dataset_model_1"]

# Get first participant's behavior all rows where column ID has values 1
first_participant = synthetic_dataset_model_1.behavior[1:299, :]

# Load chains for first participant.
chains_m1_p1 = load("Analysis\\step03\\fitting_outputs\\model_1\\participants\\participant_1.jld2")["chains"]

# Rebuild the model from scratch.
cog_model_1 = create_cognitive_model_1()
recovery_model = create_model(
    cog_model_1,
    (
            θ_T = truncated(Normal(0.75, 0.12), 0, 1),
            ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
            α₂  = truncated(Normal(0.5, 0.2), 0, 1),
            β₁  = truncated(Normal(2.0, 2), 0, 8),
            p   = truncated(Normal(1.0, 0.5), 0, Inf),
            β₂  = truncated(Normal(2.0, 2), 0, 8),
        ),
    first_participant;
    action_cols = [:choice],
    observation_cols = [:action, :observation],
    session_cols = [:ID],
)

mean_chain = MCMCChains.Chains(
    reshape(
        [mean(vec(chains_m1_p1[s].data)) for s in names(chains_m1_p1, :parameters)],
        1, :, 1
    ),
    names(chains_m1_p1, :parameters)
)
logliks = Turing.pointwise_loglikelihoods(recovery_model.model, mean_chain) |>
          d -> sum(values(d))
loglik = logliks[1, 1]   
n_trials = 299
k = 6
bic = -2 * loglik + k * log(n_trials)
@info "Participant 1, Model 1" loglik n_trials k bic
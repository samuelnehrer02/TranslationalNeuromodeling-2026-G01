##########################################################################################
############################## Parameter Recovery Study ##################################
##########################################################################################

####################################### SETUP ##############################################
using Pkg
using JLD2, DataFrames


##########################################################################################
############################ 1. Simulate synthetic behavior ##############################
##########################################################################################

### 1. Load models from step01
include("..\\step01\\step01.jl")

### 2. Load environment lookup table from step02
environment_lookup_table = load("Analysis\\step02\\environment_lookup_table.jld2")["df_env"]

### 3. Load simulate_synthetic_dataset function
include("utils\\simulate_behavior_env.jl")

###################################################################################
################################## Simulations ####################################
###################################################################################

####### 1. Simulate synthetic dataset for MODEL I. #######

### Load Parameter distributions from step02:
param_distributions_m1 = (
    θ_T = truncated(Normal(0.75, 0.12), 0, 1),
    ω₂  = truncated(Normal(-2.0, 4.0), -Inf, 0),
    α₂  = truncated(Normal(1, 1), 0, 6),
    β₁  = LogNormal(0.3, 0.6),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = LogNormal(0.4, 0.6),
)
synthetic_dataset_model_1 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_1,
    env_table = environment_lookup_table,
    n_participants = 1,
    param_distributions = param_distributions_m1,
    seed = 1,
)

####### 2. Simulate synthetic dataset for MODEL II. #######
param_distributions_m2 = (
    α₁  = truncated(Normal(1, 1), 0, 6),
    ω₂  = truncated(Normal(-2.0, 4.0), -Inf, 0),
    α₂  = truncated(Normal(1, 1), 0, 6),
    β₁  = LogNormal(0.3, 0.6),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = LogNormal(0.4, 0.6),
)
fixed_parameters_m2 = (ω₁ = -4.0,)
synthetic_dataset_model_2 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_2,
    env_table = environment_lookup_table,
    n_participants = 10,
    fixed_parameters = fixed_parameters_m2,
    param_distributions = param_distributions_m2,
    seed = 1,
)

###### 3. Simulate synthetic dataset for MODEL III. #######
param_distributions_m3 = (
    θ_T = truncated(Normal(0.75, 0.12), 0, 1),
    ω₂  = truncated(Normal(-2.0, 4.0), -Inf, 0),
    α₂  = truncated(Normal(1, 1), 0, 6),
    β₁  = LogNormal(0.3, 0.6),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = LogNormal(0.4, 0.6),
)
synthetic_dataset_model_3 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_3,
    env_table = environment_lookup_table,
    n_participants = 10,
    param_distributions = param_distributions_m3,
    seed = 1,
)
####### 4. Simulate synthetic dataset for MODEL IV. #######
param_distributions_m4 = (
    α₁  = truncated(Normal(1, 1), 0, 6),
    ω₂  = truncated(Normal(-2.0, 4.0), -Inf, 0),
    α₂  = truncated(Normal(1, 1), 0, 6),
    β₁  = LogNormal(0.3, 0.6),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = LogNormal(0.4, 0.6),
)
fixed_parameters_m4 = (ω₁ = -4.0,)
synthetic_dataset_model_4 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_4,
    env_table = environment_lookup_table,
    n_participants = 10,
    fixed_parameters = fixed_parameters_m4,
    param_distributions = param_distributions_m4,
    seed = 1,
)

##########################################################################################
############################ 2. Fit models to synthetic data #############################
##########################################################################################


################# FIT MODEL I. #########################################################

# Instantiate a fresh cognitive model for fitting purposes 
# In model I., no need to set specific fixed parameters, we use defaults.
cog_model_1 = create_cognitive_model_1()
recovery_model_1 = create_model(
    cog_model_1,
    param_distributions_m1,
    synthetic_dataset_model_1.behavior;
    action_cols = [:choice],
    observation_cols = [:action, :observation],
    session_cols = [:ID],
)

# RUN SAMPLER: We use a No-U-Turn Sampler (NUTS) with 4 chains, 1000 samples per chain.
recovery_chains_model_1 = sample_posterior!(
    recovery_model_1,
    MCMCThreads(), # Or MCMCDistributed() ?
    n_samples = 1000,
    n_chains = 4,
    sampler = NUTS(;),
)

################# FIT MODEL II. #########################################################

# Instantiate a fresh cognitive model for fitting purposes 
# In model II., we set the fixed parameter ω₁ = -4.0.
cog_model_2 = create_cognitive_model_2()
set_parameters!(cog_model_2.submodel, :ω₁ , fixed_parameters_m2[:ω₁])
get_parameters(cog_model_2.submodel)[:ω₁] == fixed_parameters_m2[:ω₁]
recovery_model_2 = create_model(
    cog_model_2,
    param_distributions_m2,
    synthetic_dataset_model_2.behavior;
    action_cols = [:choice],
    observation_cols = [:action, :observation],
    session_cols = [:ID],
)

# RUN SAMPLER: We use a No-U-Turn Sampler (NUTS) with 4 chains, 1000 samples per chain.
recovery_chains_model_2 = sample_posterior!(
    recovery_model_2,
    MCMCThreads(), # Or MCMCDistributed() ?
    n_samples = 1000,
    n_chains = 4,
    sampler = NUTS(;),
)

################# FIT MODEL III. ########################################################

# Instantiate a fresh cognitive model for fitting purposes 
# In model III., no need to set specific fixed parameters, we use defaults.
cog_model_3 = create_cognitive_model_3()
recovery_model_3 = create_model(
    cog_model_3,
    param_distributions_m3,
    synthetic_dataset_model_3.behavior;
    action_cols = [:choice],
    observation_cols = [:action, :observation],
    session_cols = [:ID],
)

# RUN SAMPLER: We use a No-U-Turn Sampler (NUTS) with 4 chains, 1000 samples per chain.
recovery_chains_model_3 = sample_posterior!(
    recovery_model_3,
    MCMCThreads(), # Or MCMCDistributed() ?
    n_samples = 1000,
    n_chains = 4,
    sampler = NUTS(;),
)

################# FIT MODEL IV. ########################################################

# Instantiate a fresh cognitive model for fitting purposes 
# In model IV., we set the fixed parameter ω₁ = -4.0.
cog_model_4 = create_cognitive_model_4()
set_parameters!(cog_model_4.submodel, :ω₁ , fixed_parameters_m4[:ω₁])
get_parameters(cog_model_4.submodel)[:ω₁] == fixed_parameters_m4[:ω₁]
recovery_model_4 = create_model(
    cog_model_4,
    param_distributions_m4,
    synthetic_dataset_model_4.behavior;
    action_cols = [:choice],
    observation_cols = [:action, :observation],
    session_cols = [:ID],
)

# RUN SAMPLER: We use a No-U-Turn Sampler (NUTS) with 4 chains, 1000 samples per chain.
recovery_chains_model_4 = sample_posterior!(
    recovery_model_4,
    MCMCThreads(), # Or MCMCDistributed() ?
    n_samples = 1000,
    n_chains = 4,
    sampler = NUTS(;),
)

##########################################################################################
############################ 4. SAVING RESULTS ###########################################
##########################################################################################
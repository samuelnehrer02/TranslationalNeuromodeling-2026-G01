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
    ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
    α₂  = truncated(Normal(0.5, 0.2), 0, 1),
    β₁  = truncated(Normal(2.0, 2), 0, 8),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = truncated(Normal(2.0, 2), 0, 8),
)
synthetic_dataset_model_1 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_1,
    env_table = environment_lookup_table,
    n_participants = 100,
    param_distributions = param_distributions_m1,
    seed = 1,
)
synthetic_dataset_model_1.parameters

####### 2. Simulate synthetic dataset for MODEL II. #######
param_distributions_m2 = (
    α₁  = truncated(Normal(0.5, 0.2), 0, 1),
    ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
    α₂  = truncated(Normal(0.5, 0.2), 0, 1),
    β₁  = truncated(Normal(2.0, 2), 0, 8),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = truncated(Normal(2.0, 2), 0, 8),
)

fixed_parameters_m2 = (ω₁ = -4.0,)
synthetic_dataset_model_2 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_2,
    env_table = environment_lookup_table,
    n_participants = 100,
    fixed_parameters = fixed_parameters_m2,
    param_distributions = param_distributions_m2,
    seed = 1,
)

###### 3. Simulate synthetic dataset for MODEL III. #######
param_distributions_m3 = (
    θ_T = truncated(Normal(0.75, 0.12), 0, 1),
    ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
    α₂  = truncated(Normal(0.5, 0.2), 0, 1),
    β₁  = truncated(Normal(2.0, 2), 0, 8),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = truncated(Normal(2.0, 2), 0, 8),
)
synthetic_dataset_model_3 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_3,
    env_table = environment_lookup_table,
    n_participants = 100,
    param_distributions = param_distributions_m3,
    seed = 1,
)

####### 4. Simulate synthetic dataset for MODEL IV. #######
param_distributions_m4 = (
    α₁  = truncated(Normal(0.5, 0.2), 0, 1),
    ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
    α₂  = truncated(Normal(0.5, 0.2), 0, 1),
    β₁  = truncated(Normal(2.0, 2), 0, 8),
    p   = truncated(Normal(1.0, 0.5), 0, Inf),
    β₂  = truncated(Normal(2.0, 2), 0, 8),
)
fixed_parameters_m4 = (ω₁ = -4.0,)
synthetic_dataset_model_4 = simulate_synthetic_dataset(
    create_model_fn = create_cognitive_model_4,
    env_table = environment_lookup_table,
    n_participants = 100,
    fixed_parameters = fixed_parameters_m4,
    param_distributions = param_distributions_m4,
    seed = 1,
)

####################################### SAVE SYNTHETIC DATASETS ##############################################
save("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_1.jld2", "synthetic_dataset_model_1", synthetic_dataset_model_1)
save("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_2.jld2", "synthetic_dataset_model_2", synthetic_dataset_model_2)
save("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_3.jld2", "synthetic_dataset_model_3", synthetic_dataset_model_3)
save("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_4.jld2", "synthetic_dataset_model_4", synthetic_dataset_model_4)



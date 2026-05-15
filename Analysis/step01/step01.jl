##########################################################################################
####################################### SETUP ############################################
##########################################################################################
using Pkg
using HierarchicalGaussianFiltering, ActionModels, Distributions, StatsPlots
# Include update_hgf! extension for handling of multiple independent hgfs
include("utils\\update_hgf_extension.jl")

##########################################################################################
#################### Load the models and their corresponding agents ####################
##########################################################################################

include("model_1.jl")
include("model_2.jl")
include("model_3.jl")
include("model_4.jl")


# --- Model I: Fixed T, Bellman equation planning (free params: β₁, β₂, p, θ_T) ---
cognitive_model_1 = create_cognitive_model_1()
agent_model_1     = init_agent(cognitive_model_1, save_history = true)

# --- Model II: Learned T, Bellman equation planning (free params: β₁, β₂, p) ---
cognitive_model_2 = create_cognitive_model_2()
agent_model_2     = init_agent(cognitive_model_2, save_history = true)

# --- Model III: Fixed T, Bellman softmax planning (free params: β₁, β₂, p, θ_T) ---
cognitive_model_3 = create_cognitive_model_3()
agent_model_3     = init_agent(cognitive_model_3, save_history = true)

# --- Model IV: Learned T, Bellman softmax planning (free params: β₁, β₂, p) ---
cognitive_model_4 = create_cognitive_model_4()
agent_model_4     = init_agent(cognitive_model_4, save_history = true)

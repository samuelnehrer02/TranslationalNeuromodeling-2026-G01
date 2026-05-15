##########################################################################################
####################################### SETUP ############################################
##########################################################################################
using Pkg
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

cognitive_model_1 = create_cognitive_model_1()
cognitive_model_2 = create_cognitive_model_2()
cognitive_model_3 = create_cognitive_model_3()
cognitive_model_4 = create_cognitive_model_4()


sim_agent_1 = init_agent(cognitive_model_1, save_history = true)
sim_agent_2 = init_agent(cognitive_model_2, save_history = true)
sim_agent_3 = init_agent(cognitive_model_3, save_history = true)
sim_agent_4 = init_agent(cognitive_model_4, save_history = true)


include("..\\step02\\utils\\simulation_environment.jl")

# Run the full 150-trial simulation for each model, SINGLE AGENT! per model.
sim_1 = simulate_agent(sim_agent_1, environment_lookup_table)
sim_2 = simulate_agent(sim_agent_2, environment_lookup_table)
sim_3 = simulate_agent(sim_agent_3, environment_lookup_table)
sim_4 = simulate_agent(sim_agent_4, environment_lookup_table)


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
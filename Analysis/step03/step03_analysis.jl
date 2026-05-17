##########################################################################################
############################ Parameter Recovery Analysis #################################
##########################################################################################
### 1. Load models from step01
include("..\\step01\\step01.jl")


### 2. Load analysis utilities
include("utils\\analysis_utils.jl")

### 3. Load synthetic datasets
synthetic_dataset_model_1 = load("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_1.jld2")["synthetic_dataset_model_1"]
synthetic_dataset_model_2 = load("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_2.jld2")["synthetic_dataset_model_2"]
synthetic_dataset_model_3 = load("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_3.jld2")["synthetic_dataset_model_3"]
synthetic_dataset_model_4 = load("Analysis\\step03\\synthetic_data\\synthetic_dataset_model_4.jld2")["synthetic_dataset_model_4"]
##########################################################################################
# Model 1 — parameter recovery
##########################################################################################

# Load and concatenate per-participant chains (renames params → param.session[pid])
all_chains_m1         = [MCMCChains.get_sections(load_and_rename(1, pid), :parameters) for pid in 1:100]
joined_chains_model_1 = hcat(all_chains_m1...)

# Posterior medians per participant
medians_m1 = extract_medians(joined_chains_model_1, 100)
# Recovery plot: generative (synthetic_dataset_model_1.parameters) vs estimated medians
fig_m1 = plot_recovery(synthetic_dataset_model_1.parameters, medians_m1;
                       title_str = "Model 1 — Parameter Recovery")

##########################################################################################
# Model 2 — parameter recovery
##########################################################################################

# Load and concatenate per-participant chains (renames params → param.session[pid])
all_chains_m2         = [MCMCChains.get_sections(load_and_rename(2, pid), :parameters) for pid in 1:100]
joined_chains_model_2 = hcat(all_chains_m2...)
# Posterior medians per participant
medians_m2 = extract_medians(joined_chains_model_2, 100)
# Recovery plot: generative (synthetic_dataset_model_2.parameters) vs estimated medians
fig_m2 = plot_recovery(synthetic_dataset_model_2.parameters, medians_m2;
                       title_str = "Model 2 — Parameter Recovery")

##########################################################################################
# Model 3 — parameter recovery
##########################################################################################
all_chains_m3         = [MCMCChains.get_sections(load_and_rename(3, pid), :parameters) for pid in 1:100]
joined_chains_model_3 = hcat(all_chains_m3...)
medians_m3 = extract_medians(joined_chains_model_3, 100)

# Recovery plot: generative (synthetic_dataset_model_3.parameters) vs estimated medians
fig_m3 = plot_recovery(synthetic_dataset_model_3.parameters, medians_m3;
                       title_str = "Model 3 — Parameter Recovery")
##########################################################################################
# Model 4 — parameter recovery
##########################################################################################
all_chains_m4         = [MCMCChains.get_sections(load_and_rename(4, pid), :parameters) for pid in 1:100]
joined_chains_model_4 = hcat(all_chains_m4...)  
medians_m4 = extract_medians(joined_chains_model_4, 100)
# Recovery plot: generative (synthetic_dataset_model_4.parameters) vs estimated medians
fig_m4 = plot_recovery(synthetic_dataset_model_4.parameters, medians_m4; title_str = "Model 4 — Parameter Recovery")


##########################################################################################
# Save plots
##########################################################################################
savefig(fig_m1, "Analysis\\step03\\analysis_outputs\\model_1.png")
savefig(fig_m2, "Analysis\\step03\\analysis_outputs\\model_2.png")
savefig(fig_m3, "Analysis\\step03\\analysis_outputs\\model_3.png")
savefig(fig_m4, "Analysis\\step03\\analysis_outputs\\model_4.png")



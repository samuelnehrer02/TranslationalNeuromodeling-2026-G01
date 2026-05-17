################################################ Stress Test for Fitting ################################################
########################################################################################################################

using Pkg, JLD2, DataFrames
df_baseline = load("Analysis\\step04\\data\\baseline.jld2", "df_baseline")

# Test on two participants with high proportion of missing data:
#test_pids = ["Cod72A", "yLxgDp"]
#df_baseline_test = filter(:ID => in(test_pids), df_baseline)

#jldsave("Analysis\\step04\\fitting_stress_test\\df_baseline_test.jld2"; df_baseline_test)

# Load test data:
df_baseline_test = load("Analysis\\step04\\fitting_stress_test\\df_baseline_test.jld2", "df_baseline_test")
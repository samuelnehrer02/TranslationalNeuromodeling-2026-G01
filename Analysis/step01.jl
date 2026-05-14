##########################################################################################
####################################### SETUP ############################################
##########################################################################################
using Pkg
using HierarchicalGaussianFiltering, ActionModels, Distributions, StatsPlots
# Include update_hgf! extension for handling of multiple independent hgfs
include("utils\\update_hgf_extension.jl")



##########################################################################################
####################################### Model I. #########################################
##########################################################################################

#======================================================#
#          CREATE 2-LEVEL HGFs WITH 4 ALIENS           #
#======================================================#
function second_stage_hgfs(config::Dict = Dict())
    spec_defaults = Dict(
        "n_aliens" => 4,
        # --- Alien HGFs (shared parameters: ω₂, α₂, etc.) ---
        ("xprob", "volatility") => -2,
        ("xprob", "drift") => 0,
        ("xprob", "autoconnection_strength") => 1,
        ("xprob", "initial_mean") => 0,
        ("xprob", "initial_precision") => 1,
        ("xbin", "xprob", "coupling_strength") => 1,
        # --- Transition HGF (independent parameters: ω₁, α₁, etc.) ---
        ("xprob_T", "volatility") => -2,
        ("xprob_T", "drift") => 0,
        ("xprob_T", "autoconnection_strength") => 1,
        ("xprob_T", "initial_mean") => 0,
        ("xprob_T", "initial_precision") => 1,
        ("xbin_T", "xprob_T", "coupling_strength") => 1,
        "update_type" => EnhancedUpdate(),
        "save_history" => true,
    )
    config = merge(spec_defaults, config)

    nodes = HierarchicalGaussianFiltering.AbstractNodeInfo[]
    edges = Dict{Tuple{String,String},HierarchicalGaussianFiltering.CouplingType}()
    grouped_xprob_volatility = []
    grouped_xprob_drift = []
    grouped_xprob_autoconnection_strength = []
    grouped_xprob_initial_mean = []
    grouped_xprob_initial_precision = []
    grouped_xbin_xprob_coupling_strength = []

    aliens = ["_A", "_B", "_C", "_D"]

    # ===== Alien HGFs (shared parameters) =====
    for i = 1:config["n_aliens"]
        alien = aliens[i]
        push!(nodes, BinaryInput("u$alien"))
        push!(nodes, BinaryState("xbin$alien"))
        push!(nodes, ContinuousState(
            name = "xprob$alien",
            volatility = config[("xprob", "volatility")],
            drift = config[("xprob", "drift")],
            autoconnection_strength = config[("xprob", "autoconnection_strength")],
            initial_mean = config[("xprob", "initial_mean")],
            initial_precision = config[("xprob", "initial_precision")],
        ))
        push!(grouped_xprob_volatility, ("xprob$alien", "volatility"))
        push!(grouped_xprob_drift, ("xprob$alien", "drift"))
        push!(grouped_xprob_autoconnection_strength, ("xprob$alien", "autoconnection_strength"))
        push!(grouped_xprob_initial_mean, ("xprob$alien", "initial_mean"))
        push!(grouped_xprob_initial_precision, ("xprob$alien", "initial_precision"))
        push!(grouped_xbin_xprob_coupling_strength,
              ("xbin$alien", "xprob$alien", "coupling_strength"))
        push!(edges, ("u$alien", "xbin$alien") => ObservationCoupling())
        push!(edges, ("xbin$alien", "xprob$alien") =>
              ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")]))
    end

    # ===== Transition HGF (independent parameters) =====
    push!(nodes, BinaryInput("u_T"))
    push!(nodes, BinaryState("xbin_T"))
    push!(nodes, ContinuousState(
        name = "xprob_T",
        volatility = config[("xprob_T", "volatility")],
        drift = config[("xprob_T", "drift")],
        autoconnection_strength = config[("xprob_T", "autoconnection_strength")],
        initial_mean = config[("xprob_T", "initial_mean")],
        initial_precision = config[("xprob_T", "initial_precision")],
    ))
    push!(edges, ("u_T", "xbin_T") => ObservationCoupling())
    push!(edges, ("xbin_T", "xprob_T") =>
          ProbabilityCoupling(config[("xbin_T", "xprob_T", "coupling_strength")]))

    parameter_groups = [
        # --- Alien parameter groups ---
        ParameterGroup("ω₂", grouped_xprob_volatility,
                       config[("xprob", "volatility")]),
        ParameterGroup("xprob_drift", grouped_xprob_drift,
                       config[("xprob", "drift")]),
        ParameterGroup("xprob_autoconnection_strength",
                       grouped_xprob_autoconnection_strength,
                       config[("xprob", "autoconnection_strength")]),
        ParameterGroup("xprob_initial_mean", grouped_xprob_initial_mean,
                       config[("xprob", "initial_mean")]),
        ParameterGroup("xprob_initial_precision", grouped_xprob_initial_precision,
                       config[("xprob", "initial_precision")]),
        ParameterGroup("α₂", grouped_xbin_xprob_coupling_strength,
                       config[("xbin", "xprob", "coupling_strength")]),
        # --- Transition: single-element groups so you get clean ω₁/α₁ names ---
        ParameterGroup("ω₁", [("xprob_T", "volatility")],
                       config[("xprob_T", "volatility")]),
        ParameterGroup("α₁", [("xbin_T", "xprob_T", "coupling_strength")],
                       config[("xbin_T", "xprob_T", "coupling_strength")]),
    ]

    hgf = init_hgf(
        nodes = nodes,
        edges = edges,
        parameter_groups = parameter_groups,
        verbose = false,
        node_defaults = NodeDefaults(update_type = config["update_type"]),
        save_history = config["save_history"],
    )
    return hgf
end

hgf = second_stage_hgfs()

# this would be the input to the hgf is alien A was chosen and returned nothing (no reward), other hgfs receive missing
# Transition observed (common = 1); alien HGFs untouched

update_hgf!(hgf, [nothing, nothing, nothing, nothing, 1])

update_hgf!(hgf, [1, missing, missing, missing, nothing])


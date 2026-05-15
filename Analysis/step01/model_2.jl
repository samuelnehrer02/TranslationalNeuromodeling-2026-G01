##########################################################################################
####################################### Model II. ########################################
##########################################################################################

#======================================================#
#   CREATE 2-LEVEL HGFs WITH 4 ALIENS + 1 TRANSITION   #
#======================================================#
function hgfs_model_2(config::Dict = Dict())
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
        # --- Transition: single-element groups  ---
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


#======================================================#
#                ACTION MODEL FUNCTION:                #
#======================================================#

# Stage 1 Learning: Single HGF updated after Rocket->Planet transition has been observed
# Stage 1 Planning: VP1= max([μ₂A, μ₂B]), VP2= max([μ₂C, μ₂D])
#                   T = [μ₁ 1-μ₁; 1-μ₁  μ₁]
#                   Q = T*[VP1, VP2] 
#                   π₁ = softmax(β₁*(Q + p*rep)), where rep = last 1-stage action (rocket choice)
# Stage 2 Learning: Four hgfs, one tracking each alien [A,B,C,D], updated after Alien -> Reward observation.
# Stage 2 Planning: If on planet 1: π₂(P1) = softmax(β₂ * [μ₂A, μ₂B])
#                   If on planet 2: π₂(P2) = softmax(β₂ * [μ₂C, μ₂D])                 

function planning_model_2(
    attributes::ModelAttributes,
    action::Union{Int, AbstractString, Missing},
    observation::Union{Int, Missing},
)
    # roughly 1.6% of trials have missing observation,
    # therefore we simply return a uniform here and move on:
    if ismissing(action) || ismissing(observation)
        return Categorical([0.5, 0.5])
    end

    # 1. Extract the hgf:
    hgf = attributes.submodel
    # Pre-allocate the input for the hgf:
    hgf_input = Vector{Union{Int, Missing, Nothing}}(nothing, length(hgf.input_nodes))
    # 2. Find out what stage of the task we are in (transition or alien choice):

    #   If we have just observed the transition, we need to:
    #   1. Save Stage-1 action that was just taken.
    #   2. Update the transition hgf:
    #   3. Return the choice probabilities for the aliens:
    if action isa Int
        # If we have just observed Stage 1 transition, save the action
        # to be used to determine "rep" at next stage.
        update_state!(attributes, :last_S1_action, action)

        #  Update the transition hgf based on the outcome
        is_common = (action == observation) ? 1 : 0
        hgf_input[5] = is_common
        update_hgf!(hgf, hgf_input)

        # Return choice probabilities for the aliens depending on the planet:
        β₂ = load_parameters(attributes).β₂
        # If we are on planet 1:
        if observation == 1
            μ₂A = get_states(hgf, ("xbin_A", "prediction_mean"))|> x -> x === missing ? 0.5 : x
            μ₂B = get_states(hgf, ("xbin_B", "prediction_mean"))|> x -> x === missing ? 0.5 : x
            return Categorical(softmax(β₂ * [μ₂A, μ₂B]))
         # If we are on planet 2:
        else
            μ₂C = get_states(hgf, ("xbin_C", "prediction_mean"))|> x -> x === missing ? 0.5 : x
            μ₂D = get_states(hgf, ("xbin_D", "prediction_mean"))|> x -> x === missing ? 0.5 : x
            return Categorical(softmax(β₂ * [μ₂C, μ₂D]))
        end

    #   If we have just observed a reward given our choice of alien:
    #   1. First update the alien HGF's
    #   2. Calculate 1-stage action:
    else
        hgf_input[1:4] .= missing
        alien_idx = Dict("A" => 1, "B" => 2, "C" => 3, "D" => 4)[action]
        hgf_input[alien_idx] = observation
        update_hgf!(hgf, hgf_input)

        # Now we are deciding which planet to choose:
        # 1. Form the transition matrix T:
        μ₁ = get_states(hgf, ("xbin_T", "prediction_mean"))|> x -> x === missing ? 0.5 : x
        T = [μ₁ 1-μ₁;
            1-μ₁  μ₁]
        # 2. Calculate value of each planet (max reward):
        μ₂A = get_states(hgf, ("xbin_A", "prediction_mean"))|> x -> x === missing ? 0.5 : x
        μ₂B = get_states(hgf, ("xbin_B", "prediction_mean"))|> x -> x === missing ? 0.5 : x
        VP1 = maximum([μ₂A, μ₂B])

        μ₂C = get_states(hgf, ("xbin_C", "prediction_mean"))|> x -> x === missing ? 0.5 : x
        μ₂D = get_states(hgf, ("xbin_D", "prediction_mean"))|> x -> x === missing ? 0.5 : x
        VP2 = maximum([μ₂C, μ₂D])

        # 3. Calculate Q:
        Q = T*[VP1, VP2]
        β₁ = load_parameters(attributes).β₁
        p = load_parameters(attributes).p
        rep = load_states(attributes).last_S1_action == 1 ? [1, 0] : [0, 1]

        return Categorical(softmax(β₁*(Q + p*rep)))
        
    end

end


#======================================================#
#                  ACTION MODEL AGENT                  #
#======================================================#

# Free Parameters:
# β₁, β₂, p

function create_cognitive_model_2(
)
    ### CREATE HGF ###
    hgf = hgfs_model_2()
    parameters = (β₁ = Parameter(1.0), β₂ = Parameter(1.0), p = Parameter(1.0))
    
    ### INPUT STRUCTURE ###
    observations = (;
        action = Observation(Union{Missing, Int, String}),
        observation = Observation(Union{Missing, Int}),       
    )
    states = (;
        last_S1_action = State(Union{Missing, Int}),
    )
    ###  OUTPUT STRUCTURE ###
    actions = (; choice = Action(Categorical),)
    return ActionModel(
        planning_model_2,
        parameters = parameters,
        observations = observations,
        states=states,
        actions = actions,
        submodel = hgf,
    )
end



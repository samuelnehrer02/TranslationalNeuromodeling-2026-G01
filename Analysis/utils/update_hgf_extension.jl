"""
    update_hgf!(hgf::HGF, inputs::Vector{<:Union{Real,Missing,Nothing}}; stepsize::Real = 1)

Extend the `update_hgf!` function from HierarchicalGaussianFiltering to handle input vectors that may contain `nothing` in addition to `Real` or `Missing`.

When an input is `nothing`, the corresponding input node and its ancestors are not updated or "aged" for this timestep.
This allows updating only the subset of HGF nodes corresponding to non-`nothing` elements of `inputs` (i.e., updating a subset of parallel hierarchical Gaussian filters),
making it possible to update separate HGFs in parallel without advancing (aging) the states of nodes corresponding to `nothing` values. 
"""

import HierarchicalGaussianFiltering:
    HGF,
    update_hgf!,
    update_node_prediction!,
    update_node_posterior!,
    update_node_value_prediction_error!,
    update_node_precision_prediction_error!,
    update_node_input!


function update_hgf!(
    hgf::HGF,
    inputs::Vector{<:Union{Real,Missing,Nothing}};
    stepsize::Real = 1,
)
    active_inputs = Dict{String,Union{Real,Missing}}()
    active_states = Set{String}()
    for (input_node, val) in zip(hgf.ordered_nodes.input_nodes, inputs)
        val === nothing && continue
        active_inputs[input_node.name] = val
        union!(active_states, _ancestors(input_node))
    end

    in_state(n) = n.name in active_states
    in_input(n) = haskey(active_inputs, n.name)

    for node in reverse(hgf.ordered_nodes.all_state_nodes)
        in_state(node) && update_node_prediction!(node, stepsize)
    end
    for node in reverse(hgf.ordered_nodes.input_nodes)
        in_input(node) && update_node_prediction!(node, stepsize)
    end

    for (name, val) in active_inputs
        update_node_input!(hgf.input_nodes[name], val)
    end

    for node in hgf.ordered_nodes.input_nodes
        in_input(node) && update_node_value_prediction_error!(node)
    end

    for node in hgf.ordered_nodes.early_update_state_nodes
        if in_state(node)
            update_node_posterior!(node, node.update_type)
            update_node_value_prediction_error!(node)
            update_node_precision_prediction_error!(node)
        end
    end

    for node in hgf.ordered_nodes.input_nodes
        in_input(node) && update_node_precision_prediction_error!(node)
    end

    for node in hgf.ordered_nodes.late_update_state_nodes
        if in_state(node)
            update_node_posterior!(node, node.update_type)
            update_node_value_prediction_error!(node)
            update_node_precision_prediction_error!(node)
        end
    end

    if hgf.save_history
        push!(hgf.timesteps, hgf.timesteps[end] + stepsize)
        for node in hgf.ordered_nodes.all_nodes
            for state_name in fieldnames(typeof(node.states))
                push!(getfield(node.history, state_name),
                      getfield(node.states, state_name))
            end
        end
    end

    return nothing
end

function _ancestors(node, visited::Set{String} = Set{String}())
    for prop in propertynames(node.edges)
        endswith(string(prop), "_parents") || continue
        for parent in getproperty(node.edges, prop)
            if !(parent.name in visited)
                push!(visited, parent.name)
                _ancestors(parent, visited)
            end
        end
    end
    return visited
end
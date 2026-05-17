using JLD2, DataFrames, StatsPlots, Plots, Statistics, GLM, Printf
function load_and_rename(model_id::Int, pid::Int)
    path = joinpath("Analysis", "step03", "fitting_outputs",
                    "model_$(model_id)", "participants",
                    "participant_$(pid).jld2")
    chains = load(path)["chains"]

    rename_map = Dict(
        p => Symbol(replace(string(p), "session[1]" => "session[$(pid)]"))
        for p in names(chains, :parameters)
    )
    return replacenames(chains, rename_map)
end



function extract_medians(joined_chains, n_participants::Int)
    # Get all parameter names, e.g. "θ_T.session[1]"
    all_params = string.(names(joined_chains, :parameters))

    # Extract unique parameter base names (everything before ".session[")
    param_bases = unique([split(p, ".session[")[1] for p in all_params])

    rows = []
    for pid in 1:n_participants
        row = Dict{Symbol, Any}(:id => pid)
        for base in param_bases
            sym = Symbol("$(base).session[$(pid)]")
            row[Symbol(base)] = median(vec(joined_chains[sym].data))
        end
        push!(rows, row)
    end

    return DataFrame(rows)
end


function plot_recovery(truth_df::DataFrame, medians_df::DataFrame; title_str="Model 1")
    truth = copy(truth_df)
    truth.id = parse.(Int, truth.ID)
    select!(truth, Not(:ID))

    merged = innerjoin(truth, medians_df, on = :id, makeunique = true)

    param_names = [n for n in names(truth) if n != "id"]

    subplots = []
    for pname in param_names
        true_vals = merged[!, pname]
        est_vals  = merged[!, Symbol(pname, :_1)]

        lo = min(minimum(true_vals), minimum(est_vals))
        hi = max(maximum(true_vals), maximum(est_vals))
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)

        r = cor(true_vals, est_vals)
        m = lm(@formula(y ~ x), DataFrame(x = true_vals, y = est_vals))
        β₀, β₁ = coef(m)
        xs = range(lims[1], lims[2]; length = 100)
        ys = β₀ .+ β₁ .* xs

        p = StatsPlots.scatter(true_vals, est_vals;
            xlabel = "Generative",
            ylabel = "Estimated",
            title  = string(pname),
            xlims  = lims,
            ylims  = lims,
            aspect_ratio = :equal,
            markersize = 4,
            markercolor = :black,
            markerstrokecolor = :black,
            markerstrokewidth = 0.5,
            label = "",
            legend = false,
            grid = true,
        )
        Plots.plot!(p, [lims[1], lims[2]], [lims[1], lims[2]];
            linestyle = :dash, color = :grey, label = "")

        Plots.plot!(p, xs, ys; color = :teal, linewidth = 2, label = "")
        Plots.annotate!(p,
            lims[1] + 0.05 * (lims[2] - lims[1]),
            lims[2] - 0.08 * (lims[2] - lims[1]),
            Plots.text(@sprintf("r = %.2f", r), 10, :left, :top, :bold),
        )
        push!(subplots, p)
    end

    return Plots.plot(subplots...;
        layout = (2, 3),
        size = (1100, 750),
        plot_title = title_str,
    )
end
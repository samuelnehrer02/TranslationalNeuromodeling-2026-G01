##########################################################################################
#  STEP 03 — PARAMETER RECOVERY: FITTING
#  ─────────────────────────────────────────────────────────────────────────────────────
#  HOW TO RUN (from the project root directory):
#
#      julia --project=. Analysis\step03\step03_fitting.jl
#
#  WHAT HAPPENS:
#    · 20 Distributed workers are launched.
#    · 400 jobs (4 models × 100 participants) are shuffled for load balance and
#      dispatched dynamically across workers via pmap.
#    · Each job fits exactly 1 participant → 6-dimensional NUTS posterior, keeping
#      gradient cost and trajectory length tractable. Fitting participants jointly
#      (even when they are independent) incurs 20–100× extra cost from NUTS's
#      superlinear scaling on the joint geometry.
#    · BLAS is pinned to 1 thread per worker (LinearAlgebra.BLAS.set_num_threads(1))
#      to prevent BLAS from spawning competing threads across the 20 workers.
#    · Each fit is wrapped in a retry loop (≤ 4 attempts, 15 s × attempt backoff).
#      The cognitive model is rebuilt from scratch on every attempt so no corrupted
#      HGF mutable state carries over.
#    · Participant chains are saved immediately on success:
#        Analysis/step03/fitting_outputs/model_N/participants/participant_P.jld2
#    · After all workers finish, participant chains for each model are merged with
#      chainscat and the final chain is saved:
#        Analysis/step03/fitting_outputs/model_N/recovery_chains_model_N.jld2
#    · A per-model pass/fail summary is printed.
#
#  PREREQUISITES:
#    · Run Analysis\step03\step03_sim_data.jl first to generate synthetic datasets.
#    · All packages must be installed:
#        julia --project=. -e "using Pkg; Pkg.instantiate()"
##########################################################################################
using Pkg
using Distributed, Random


addprocs(20)

##########################################################################################
# Everything in this block runs on ALL workers AND the main process.
##########################################################################################
@everywhere begin

    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))   # project root

    # Pin BLAS to 1 thread per worker so 20 workers don't compete for CPU cores.
    using LinearAlgebra
    BLAS.set_num_threads(1)

    using HierarchicalGaussianFiltering, ActionModels, Distributions,
          JLD2, Turing, DataFrames, MCMCChains

    include(joinpath(@__DIR__, "..", "step01", "step01.jl"))

    # ── MCMC hyperparameters ──────────────────────────────────────────────────────────
    N_SAMPLES      = 1000
    N_CHAINS       = 4
    N_PARTICIPANTS = 100

    # ── Fixed HGF parameters ─────────────────────────────────────────────────────────
    # Models II and IV fix ω₁ = −4.0 (matches step03_sim_data.jl).
    FIXED_PARAMS = Dict{Int, Any}(
        1 => nothing,
        2 => (ω₁ = -4.0,),
        3 => nothing,
        4 => (ω₁ = -4.0,),
    )

    # ── Prior distributions ───────────────────────────────────────────────────────────
    PARAM_DIST = Dict(
        1 => (
            θ_T = truncated(Normal(0.75, 0.12), 0, 1),
            ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
            α₂  = truncated(Normal(0.5, 0.2), 0, 1),
            β₁  = truncated(Normal(2.0, 2), 0, 8),
            p   = truncated(Normal(1.0, 0.5), 0, Inf),
            β₂  = truncated(Normal(2.0, 2), 0, 8),
        ),
        2 => (
            α₁  = truncated(Normal(0.5, 0.2), 0, 1),
            ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
            α₂  = truncated(Normal(0.5, 0.2), 0, 1),
            β₁  = truncated(Normal(2.0, 2), 0, 8),
            p   = truncated(Normal(1.0, 0.5), 0, Inf),
            β₂  = truncated(Normal(2.0, 2), 0, 8),
        ),
        3 => (
            θ_T = truncated(Normal(0.75, 0.12), 0, 1),
            ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
            α₂  = truncated(Normal(0.5, 0.2), 0, 1),
            β₁  = truncated(Normal(2.0, 2), 0, 8),
            p   = truncated(Normal(1.0, 0.5), 0, Inf),
            β₂  = truncated(Normal(2.0, 2), 0, 8),
        ),
        4 => (
            α₁  = truncated(Normal(0.5, 0.2), 0, 1),
            ω₂  = truncated(Normal(-4.0, 2.5), -15, -0.5),
            α₂  = truncated(Normal(0.5, 0.2), 0, 1),
            β₁  = truncated(Normal(2.0, 2), 0, 8),
            p   = truncated(Normal(1.0, 0.5), 0, Inf),
            β₂  = truncated(Normal(2.0, 2), 0, 8),
        ),
    )

    # ── Cognitive model constructors ──────────────────────────────────────────────────
    CREATE_FN = Dict(
        1 => create_cognitive_model_1,
        2 => create_cognitive_model_2,
        3 => create_cognitive_model_3,
        4 => create_cognitive_model_4,
    )

    # ─────────────────────────────────────────────────────────────────────────────────
    # fit_with_retry(model_id, participant_id)
    #
    # Filters the single participant from the full dataset and fits a 6-dimensional
    # NUTS posterior. Rebuilds the cognitive model from scratch on every retry to
    # guarantee a clean HGF state (defence against InterruptException /
    # ForwardDiff leapfrog errors).
    # ─────────────────────────────────────────────────────────────────────────────────
    function fit_with_retry(
        model_id::Int,
        participant_id::Int;
        max_attempts::Int     = 4,
        backoff_seconds::Real = 15.0,
    )
        part_dir = joinpath(@__DIR__, "fitting_outputs", "model_$(model_id)", "participants")
        mkpath(part_dir)
        out_path = joinpath(part_dir, "participant_$(participant_id).jld2")

        # Load the full synthetic dataset and extract this participant's rows.
        data_key  = "synthetic_dataset_model_$(model_id)"
        data_path = joinpath(@__DIR__, "synthetic_data", "$(data_key).jld2")
        if !isfile(data_path)
            @error "[M$(model_id) P$(participant_id)] Synthetic data not found: $(data_path)" *
                   "\nRun Analysis/step03/step03_sim_data.jl first."
            return nothing
        end
        dataset              = load(data_path)[data_key]
        behavior_participant = filter(row -> row.ID == string(participant_id), dataset.behavior)

        for attempt in 1:max_attempts
            @info "[M$(model_id) P$(participant_id)] Attempt $(attempt)/$(max_attempts)…"
            try
                # Rebuild cognitive model from scratch — clears all HGF mutable state.
                cog_model = CREATE_FN[model_id]()

                # Apply fixed HGF parameters where needed (models II and IV: ω₁).
                fixed = FIXED_PARAMS[model_id]
                if !isnothing(fixed)
                    for (k, v) in pairs(fixed)
                        set_parameters!(cog_model.submodel, k, v)
                    end
                end

                recovery_model = create_model(
                    cog_model,
                    PARAM_DIST[model_id],
                    behavior_participant;
                    action_cols      = [:choice],
                    observation_cols = [:action, :observation],
                    session_cols     = [:ID],
                )

                # MCMCSerial: 4 chains run sequentially within this single-thread worker.
                chains = sample_posterior!(
                    recovery_model,
                    MCMCSerial(),
                    n_samples = N_SAMPLES,
                    n_chains  = N_CHAINS,
                    sampler   = NUTS(;),
                )

                # Save immediately — other workers are unaffected.
                save(out_path, "chains", chains)
                @info "[M$(model_id) P$(participant_id)] Done → $(out_path)"
                return true

            catch e
                @warn "[M$(model_id) P$(participant_id)] Attempt $(attempt) failed" exception = (e, catch_backtrace())
                if attempt < max_attempts
                    wait_s = backoff_seconds * attempt
                    @info "[M$(model_id) P$(participant_id)] Retrying in $(wait_s)s…"
                    sleep(wait_s)
                end
            end
        end

        @error "[M$(model_id) P$(participant_id)] All $(max_attempts) attempts exhausted — skipping."
        return nothing
    end

end  # @everywhere

##########################################################################################
# Main process: create dirs, dispatch 400 workers, merge participants, print summary.
##########################################################################################

for model_id in 1:4
    mkpath(joinpath(@__DIR__, "fitting_outputs", "model_$(model_id)", "participants"))
end

# 400 jobs shuffled: slow model-4 jobs are interleaved throughout the queue so fast
# workers always have something to pick up near the end rather than stalling on a
# last wave of slow jobs.
jobs = [(model_id = m, participant_id = p) for m in 1:4 for p in 1:100]
shuffle!(jobs)

@info "Launching 400 jobs across 20 workers (4 models × 100 participants, shuffled)…"
t_start = time()
pmap(j -> fit_with_retry(j.model_id, j.participant_id), jobs)
elapsed = round(time() - t_start; digits = 1)

##########################################################################################
# Merge 100 participant chains per model into one final chain (chainscat).
# Workers saved to disk; main process loads and merges here.
##########################################################################################
@info "Merging participant chains…"

merge_status = Dict{Int, String}()

for model_id in 1:4
    part_chains   = []
    missing_parts = Int[]

    for pid in 1:N_PARTICIPANTS
        path = joinpath(
            @__DIR__, "fitting_outputs", "model_$(model_id)",
            "participants", "participant_$(pid).jld2",
        )
        if isfile(path)
            push!(part_chains, load(path)["chains"])
        else
            push!(missing_parts, pid)
        end
    end

    if isempty(part_chains)
        merge_status[model_id] = "FAILED — no participants succeeded"
        continue
    end

    merged   = chainscat(part_chains...)
    out_name = isempty(missing_parts) ?
               "recovery_chains_model_$(model_id).jld2" :
               "recovery_chains_model_$(model_id)_partial.jld2"
    merged_path = joinpath(@__DIR__, "fitting_outputs", "model_$(model_id)", out_name)
    save(merged_path, "chains", merged)

    n_ok = length(part_chains)
    merge_status[model_id] = isempty(missing_parts) ?
        "OK ($(n_ok)/100) → $(out_name)" :
        "PARTIAL ($(n_ok)/100, missing: $(missing_parts)) → $(out_name)"
    @info "Model $(model_id): merged chain saved → $(merged_path)"
end

##########################################################################################
# Summary
##########################################################################################
println("\n" * "─"^60)
println("  FITTING SUMMARY  ($(elapsed)s total wall time)")
println("─"^60)
for model_id in 1:4
    println("  Model $(model_id): $(get(merge_status, model_id, "unknown"))")
end
println("─"^60 * "\n")

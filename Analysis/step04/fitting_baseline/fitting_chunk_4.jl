##########################################################################################
#  STEP 04 — BASELINE FITTING: CHUNK 4
#  ─────────────────────────────────────────────────────────────────────────────────────
#  Run on LAPTOP (8 workers) from the project root:
#
#      julia --project=. Analysis\step04\fitting_baseline\fitting_chunk_4.jl
#
#  Input:  fitting_baseline/baseline_chunk_4.jld2  (key: "df_baseline")
#  Output: fitting_baseline/model_N/participant_<ID>.jld2
#          (model_1–model_4 folders already exist; chains saved directly inside)
#
#  · 8 workers, one job per (model × participant), shuffled for load balance.
#  · Each job fits 1 participant: 6-dimensional NUTS posterior, MCMCSerial, 3 chains.
#  · BLAS pinned to 1 thread per worker.
#  · Retry loop: ≤ 4 attempts, 15 s × attempt backoff, fresh cognitive model each try.
#  · No merge step — chains combined at analysis time.
##########################################################################################

using Distributed, Random

addprocs(8)

##########################################################################################
@everywhere begin

    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

    using LinearAlgebra
    BLAS.set_num_threads(1)

    using HierarchicalGaussianFiltering, ActionModels, Distributions,
          JLD2, Turing, DataFrames, MCMCChains

    Turing.setprogress!(false)

    include(joinpath(@__DIR__, "..", "..", "step01", "step01.jl"))

    N_SAMPLES = 1000
    N_CHAINS  = 3

    FIXED_PARAMS = Dict{Int, Any}(
        1 => nothing,
        2 => (ω₁ = -4.0,),
        3 => nothing,
        4 => (ω₁ = -4.0,),
    )

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

    CREATE_FN = Dict(
        1 => create_cognitive_model_1,
        2 => create_cognitive_model_2,
        3 => create_cognitive_model_3,
        4 => create_cognitive_model_4,
    )

    function fit_with_retry(
        model_id::Int,
        participant_id::String;
        max_attempts::Int     = 4,
        backoff_seconds::Real = 15.0,
    )
        out_path = joinpath(@__DIR__, "model_$(model_id)", "participant_$(participant_id).jld2")

        data_path = joinpath(@__DIR__, "baseline_chunk_4.jld2")
        df                   = load(data_path, "df_baseline")
        behavior_participant = filter(:ID => ==(participant_id), df)

        for attempt in 1:max_attempts
            @info "[M$(model_id) P$(participant_id)] Attempt $(attempt)/$(max_attempts)…"
            try
                cog_model = CREATE_FN[model_id]()

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

                chains = sample_posterior!(
                    recovery_model,
                    MCMCSerial(),
                    n_samples = N_SAMPLES,
                    n_chains  = N_CHAINS,
                    sampler   = NUTS(;),
                    progress  = false,
                )

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
df_chunk   = load(joinpath(@__DIR__, "baseline_chunk_4.jld2"), "df_baseline")
chunk_pids = unique(df_chunk.ID)

jobs = [(model_id = m, participant_id = p) for m in 1:4 for p in chunk_pids]
shuffle!(jobs)

@info "Chunk 4: $(length(jobs)) jobs (4 models × $(length(chunk_pids)) participants), 8 workers…"
t_start = time()
results  = pmap(j -> fit_with_retry(j.model_id, j.participant_id), jobs)
elapsed  = round(time() - t_start; digits = 1)

n_ok   = count(!isnothing, results)
n_fail = count(isnothing, results)
println("\n" * "─"^60)
println("  CHUNK 4 SUMMARY  ($(elapsed)s total wall time)")
println("─"^60)
println("  OK:     $(n_ok)/$(length(jobs))")
println("  Failed: $(n_fail)/$(length(jobs))")
if n_fail > 0
    failed = [jobs[i] for i in eachindex(jobs) if isnothing(results[i])]
    for f in failed
        println("    ✗  Model $(f.model_id) / $(f.participant_id)")
    end
end
println("─"^60 * "\n")

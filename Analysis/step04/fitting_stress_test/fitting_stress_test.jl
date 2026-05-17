##########################################################################################
#  STEP 04 — REAL DATA FITTING: STRESS-TEST PILOT
#  ─────────────────────────────────────────────────────────────────────────────────────
#  HOW TO RUN (from the project root directory):
#
#      julia --project=. Analysis\step04\fitting_stress_test\fitting_stress_test.jl
#
#  WHAT HAPPENS:
#    · 8 Distributed workers are launched (4 models × 2 pilot participants).
#    · All 8 jobs run simultaneously, one per worker.
#    · Each job fits exactly 1 participant with NUTS (MCMCSerial, 3 chains).
#    · BLAS is pinned to 1 thread per worker.
#    · Each fit is wrapped in a retry loop (≤ 4 attempts, 15 s × attempt backoff).
#      The cognitive model is rebuilt from scratch on every attempt.
#    · Participant chains are saved immediately on success:
#        fitting_stress_test/fitting_outputs/model_N/participants/participant_<ID>.jld2
#    · No merge step — chains are combined at analysis time, not here.
#    · A per-job pass/fail summary is printed.
#
#  PREREQUISITES:
#    · df_baseline_test.jld2 must be present in this folder (2 pilot participants).
#    · All packages must be installed:
#        julia --project=. -e "using Pkg; Pkg.instantiate()"
##########################################################################################


using Distributed, Random

addprocs(8)

##########################################################################################
# Everything in this block runs on ALL workers AND the main process.
##########################################################################################
@everywhere begin

    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))   # project root

    # Pin BLAS to 1 thread per worker.
    using LinearAlgebra
    BLAS.set_num_threads(1)

    using HierarchicalGaussianFiltering, ActionModels, Distributions,
          JLD2, Turing, DataFrames, MCMCChains

    # Suppress per-chain progress bars across all workers.
    Turing.setprogress!(false)

    include(joinpath(@__DIR__, "..", "..", "step01", "step01.jl"))

    # ── MCMC hyperparameters ──────────────────────────────────────────────────────────
    N_SAMPLES = 1000
    N_CHAINS  = 3

    # ── Fixed HGF parameters ─────────────────────────────────────────────────────────
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
    # Loads df_baseline_test, filters the single participant's rows, and fits a
    # 6-dimensional NUTS posterior. Rebuilds the cognitive model from scratch on every
    # retry to guarantee a clean HGF state.
    # ─────────────────────────────────────────────────────────────────────────────────
    function fit_with_retry(
        model_id::Int,
        participant_id::String;
        max_attempts::Int     = 4,
        backoff_seconds::Real = 15.0,
    )
        part_dir = joinpath(@__DIR__, "fitting_outputs", "model_$(model_id)", "participants")
        mkpath(part_dir)
        out_path = joinpath(part_dir, "participant_$(participant_id).jld2")

        # Load data and filter to this participant.
        data_path = joinpath(@__DIR__, "df_baseline_test.jld2")
        if !isfile(data_path)
            @error "[M$(model_id) P$(participant_id)] Data not found: $(data_path)"
            return nothing
        end
        df                   = load(data_path, "df_baseline_test")
        behavior_participant = filter(:ID => ==(participant_id), df)

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
# Main process: extract participant IDs, create dirs, dispatch 8 workers.
##########################################################################################

# Extract pilot participant IDs from the test dataset.
pilot_df   = load(joinpath(@__DIR__, "df_baseline_test.jld2"), "df_baseline_test")
pilot_pids = unique(pilot_df.ID)   # ["Cod72A", "yLxgDp"]

for model_id in 1:4
    mkpath(joinpath(@__DIR__, "fitting_outputs", "model_$(model_id)", "participants"))
end

# 8 jobs: 4 models × 2 participants (no shuffle needed for 8 jobs)
jobs = [(model_id = m, participant_id = p) for m in 1:4 for p in pilot_pids]

@info "Launching $(length(jobs)) pilot jobs (4 models × $(length(pilot_pids)) participants)…"
@info "Participants: $(pilot_pids)"
t_start = time()
results  = pmap(j -> fit_with_retry(j.model_id, j.participant_id), jobs)
elapsed  = round(time() - t_start; digits = 1)

##########################################################################################
# Summary — no merge step; chains are combined at analysis time.
##########################################################################################
println("\n" * "─"^60)
println("  PILOT FITTING SUMMARY  ($(elapsed)s total wall time)")
println("─"^60)
for (j, r) in zip(jobs, results)
    tag = isnothing(r) ? "FAILED" : "OK"
    println("  Model $(j.model_id) / $(j.participant_id): $(tag)")
end
println("─"^60 * "\n")

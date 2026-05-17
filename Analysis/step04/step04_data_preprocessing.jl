using Pkg, CSV, DataFrames

# Load RAW training data:
DATA_TRAIN = CSV.read("DataTrain\\train_data_long.csv", DataFrame)

# ── 1. Keep only what we need for behavioral fitting ─────────────────────────────
keep_cols = [:ID, :baseline, :trial, :rocket_choice, :planet, :alien_left_right,
             :reward, :alien]
df = select(DATA_TRAIN, keep_cols)

# ── 2. Recode -1 → missing across all behavioral columns ─────────────────────────
# (rocket_choice, planet, alien_left_right, reward, alien can all be -1)
recode_cols = [:rocket_choice, :planet, :alien_left_right, :reward, :alien]
for col in recode_cols
    df[!, col] = [ismissing(v) ? missing :
                  (v == -1 || v == "-1") ? missing : v
                  for v in df[!, col]]
end


function expand_participant(sub::AbstractDataFrame)
    n = nrow(sub)
    rows = []
    for t in 1:n
        r = sub[t, :]
        next_rocket = t < n ? sub[t+1, :rocket_choice] : missing

        # Row 1 — rocket stage
        push!(rows, (
            ID = r.ID,
            action = r.rocket_choice,
            observation = r.planet,
            choice = r.alien_left_right,
        ))

        # Row 2 — alien stage. Skip on the very last trial.
        t == n && continue

        push!(rows, (
            ID          = r.ID,
            action      = r.alien,
            observation = r.reward,
            choice      = next_rocket,
        ))
    end
    return DataFrame(rows)
end

# ── 4. Split baseline vs followup, then expand each ─────────────────────────────
df_baseline_raw = filter(:baseline => ==(true),  df)
df_followup_raw = filter(:baseline => ==(false), df)

# Sort by ID and trial to guarantee chronological order before expansion
sort!(df_baseline_raw, [:ID, :trial])
sort!(df_followup_raw, [:ID, :trial])

# ── 5. Expand baseline and followup ────────────────────────────────────────────────
df_baseline = combine(groupby(df_baseline_raw, :ID), expand_participant)
df_followup = combine(groupby(df_followup_raw, :ID), expand_participant)

# ── 6. Normalize column types ──────────────────────────────────────────────────────
for d in (df_baseline, df_followup)
    d.ID = String.(string.(d.ID))   # handles String7, String, anything → String

    d.action = [
        ismissing(a)  ? missing  :
        a isa Integer ? Int64(a) :
                        String(a)
        for a in d.action
    ]
    d.action = convert(Vector{Union{Missing, Int64, String}}, d.action)
end

# ── 7. Save the expanded data as separate jld2 files for baseline and followup ─────
using JLD2
jldsave("Analysis\\step04\\data\\baseline.jld2"; df_baseline)
jldsave("Analysis\\step04\\data\\followup.jld2"; df_followup)


# ── 8. Split baseline data into chunks for workstation and laptop ──────────────────
df_baseline = load("Analysis\\step04\\data\\baseline.jld2", "df_baseline")
all_ids = sort(unique(df_baseline.ID))
n       = length(all_ids)
# Workstation gets 3/4, laptop gets 1/4
split_at = floor(Int, 3 * n / 4)   # 477 for n = 637
ws_ids     = Set(all_ids[1:split_at])
laptop_ids = Set(all_ids[split_at+1:end])
df_baseline_chunk_1_3 = filter(:ID => in(ws_ids),     df_baseline)
df_baseline_chunk_4   = filter(:ID => in(laptop_ids), df_baseline)

#jldsave("Analysis\\step04\\fitting_baseline\\baseline_chunk_1_3.jld2"; df_baseline = df_baseline_chunk_1_3)
#jldsave("Analysis\\step04\\fitting_baseline\\baseline_chunk_4.jld2";   df_baseline = df_baseline_chunk_4)
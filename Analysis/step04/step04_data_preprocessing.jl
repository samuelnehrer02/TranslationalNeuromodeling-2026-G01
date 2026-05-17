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

# ── 6. Convert ID and action to String ─────────────────────────────────────────────
for d in (df_baseline, df_followup)
    d.ID     = String.(d.ID)
    d.action = [ismissing(a) ? missing : (a isa Char ? string(a) : a) for a in d.action]
end



# ── 7. Save the expanded data as separate jld2 files for baseline and followup ─────
using JLD2
jldsave("Analysis\\step04\\data\\baseline.jld2"; df_baseline)
jldsave("Analysis\\step04\\data\\followup.jld2"; df_followup)


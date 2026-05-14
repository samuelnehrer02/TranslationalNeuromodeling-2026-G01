using Pkg
using CSV, DataFrames
using Random

"""
recodeData(rawData)

Recode data by picking columns of interest, renaming to more intuitive names and recoding certain columns from raw data.

# Arguments
- `rawData`: DataFrame with raw data from study.

# Returns
- `DataFrame`: dataframe containing selected and recoded data
"""
function recodeData(rawData::DataFrame)::DataFrame

    # List of columns from data to include
    includeCols::Vector{String} = ["pid", "age", "sex", "education", "treatment_group", "AD", "session", "trial_num", "S1c", "step_two_state", "step_two_chosen_stim", "step_two_hit", "p1", "p2", "p3", "p4"]

    # Rename columns to more intuitive names
    cleanData::DataFrame = rename!(rawData[!, includeCols], [
        "pid" => "ID",
        "education" => "education_level",
        "treatment_group" => "treatment",
        "AD" => "anxious_depression_score",
        "session" => "baseline",
        "trial_num" => "trial",
        "S1c" => "rocket_choice",
        "step_two_state" => "planet",
        "step_two_chosen_stim" => "alien_left_right",
        "step_two_hit" => "reward",
        "p1" => "P(R|A)",
        "p2" => "P(R|B)",
        "p3" => "P(R|C)",
        "p4" => "P(R|D)"
    ])

    # Recode some columns to more intuitive codings
    cleanData.baseline .= cleanData.baseline .== "baseline"
    cleanData.rocket_choice = map(x -> get(Dict("NA" => -1), x, x), cleanData.rocket_choice)
    cleanData.planet = map(x -> get(Dict(2 => 1, 3 => 2), x, x), cleanData.planet)

    # Add global alien choice
    cleanData.alien .= ifelse.(cleanData.planet .== 1 .&& cleanData.alien_left_right .== 1, "A",
        ifelse.(cleanData.planet .== 1 .&& cleanData.alien_left_right .== 2, "B",
            ifelse.(cleanData.planet .== 2 .&& cleanData.alien_left_right .== 1, "C",
                ifelse.(cleanData.planet .== 2 .&& cleanData.alien_left_right .== 2, "D", ""))))

    return cleanData
end

"""
splitData(testSetPercentage=20.0)

Imports raw data and splits it into training and test sets while keeping proportions of treatment groups the same and then exports said datasets.
Makes use of recodeData function to clean raw data.

# Arguments
- `testSetPercentage`: A number giving the proportion of indiviuals to split into the test set in percentages (default=20%).
"""
function splitData(testSetPercentage::Float64=20.0)

    # set RNG
    rng = MersenneTwister(hash("Group_01"))

    # Import data
    rawData::DataFrame = DataFrame(CSV.File("./DataRaw/all_data_long.csv"))

    # recode dataa
    cleanData::DataFrame = recodeData(rawData)

    # get list of indiviuals and their treatment group
    treatmentsByID::DataFrame = unique(cleanData[:, ["ID", "treatment"]])

    # Determine the number of people per treatment group
    treatments::Vector{String} = unique(cleanData[!, "treatment"])
    treatmentGroupNbs::Vector{Integer} = [size(treatmentsByID[treatmentsByID[!, "treatment"].==treatment, :])[1] for treatment in treatments]

    # Randomly pick test sets per treatments...
    testSetNbs::Vector{Integer} = [round(Int, treatmentGroupNb * testSetPercentage / 100.0) for treatmentGroupNb in treatmentGroupNbs]
    testIDs::Vector{String} = vcat([rand(rng, treatmentsByID[treatmentsByID[!, "treatment"].==treatment, "ID"], testSetNbs[i]) for (i, treatment) in enumerate(treatments)]...)
    # ... and move others to training set
    trainIDs::Vector{String} = treatmentsByID.ID[treatmentsByID.ID.∉Ref(testIDs)]

    # Create test and training data from chosen IDs
    testData::DataFrame = cleanData[cleanData[!, "ID"].∈Ref(testIDs), :]
    trainData::DataFrame = cleanData[cleanData[!, "ID"].∈Ref(trainIDs), :]

    # Export test and training data
    CSV.write("./DataTest/test_data_long.csv", testData)
    CSV.write("./DataTrain/train_data_long.csv", trainData)
end

function main()

    splitData()

end

main()
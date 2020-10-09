module ParallelExp

using POMDPs
using POMDPPolicies # For function policy and random policy
using POMDPSimulators # For parallel simulator

using Statistics
using DataFrames
using CSV
using Printf
using Dates

export
    parallel_experiment,
    FuncSolver,
    CSV

function parallel_experiment(pomdp::Union{POMDP, Function},
                             number_of_episodes::Int,
                             max_steps::Int,
                             solver_list::Array;
                             belief_updater::Union{Updater,Nothing}     = nothing,
                             initial_belief::Any                        = nothing,
                             initial_state::Any                          = nothing,
                             show_progress::Bool                        = true,
                             max_queue_length::Int                      = 300,
                             full_factorial_design::Bool                = true,
                             solver_labels::Union{Array, Nothing}       = nothing,
                             solver_list_labels::Union{Array, Nothing}  = nothing,
                             experiment_label::String                   = Dates.format(now(), "mmddHHMMSS"))

    if solver_labels === nothing
        solver_labels = unique(collect(string(solver) for (solver, params) in solver_list))
    end

    if solver_list_labels === nothing
        solver_list_labels = []
        for (solver, params) in solver_list
            labels = []
            for (param, values) in params
                push!(labels, [string(value) for value in values])
            end
            push!(solver_list_labels, labels)
        end
    end

    @assert length(solver_list) == length(solver_list_labels)

    println("Generating experimental design")
    solvers = []
    param_labels = []
    for i in 1:length(solver_list)
        solver = solver_list[i].first
        param_list = solver_list[i].second
        param_list_labels = solver_list_labels[i]
        if full_factorial_design
            params = [[]]
            labels = [[]]
            for j in 1:length(param_list)
                param = param_list[j].first
                values = param_list[j].second
                value_labels = param_list_labels[j]
                params = [[a...,param=>b] for a in params for b in values]
                labels = [[a...,param=>b] for a in labels for b in value_labels]
            end
        else
            default_param = []
            default_param_labels = []
            for j in 1:length(param_list)
                param = param_list[j].first
                value = param_list[j].second[1]
                value_label = param_list_labels[j][1]
                push!(default_param, param=>value)
                push!(default_param_labels, param=>value_label)
            end
            params = []
            labels = []
            for j in 1:length(param_list)
                param = param_list[j].first
                values = param_list[j].second
                value_labels = param_list_labels[j]
                for k in 1:length(values)
                    push!(params, [default_param[1:j-1]..., param=>values[k], default_param[j+1:end]...])
                    push!(labels, [default_param_labels[1:j-1]..., param=>value_labels[k], default_param_labels[j+1:end]...])
                end
            end
            params = unique(params)
            labels = unique(labels)
        end 
        # Different sets of parameters for same solver will be merged
        isexist = false
        for i in 1:length(solvers)
            if solvers[i].first == solver
                solvers[i] = solver=>unique(vcat(solvers[i].second, params))
                param_labels[i] = solver=>unique(vcat(param_labels[i].second, labels))
                isexist = true
                break
            end
        end
        if !isexist
            push!(solvers, solver=>params)
            push!(param_labels, solver=>labels)
        end 
    end

    labels = []
    for i in 1:length(solver_labels)
        push!(labels, [[:Solver=>solver_labels[i], label...] for label in param_labels[i].second])
    end
    
    println("Simulations begin")
    queue = []
    raw_data = DataFrame()
    for i in 1:number_of_episodes
        println("Simulating the $(i)-th episode.")
        m = typeof(pomdp) <: Function ? pomdp() : pomdp
        for j in 1:length(solvers)
            solver, params = solvers[j]
            for k in 1:length(params)
                planner = solve(solver(;params[k]...), m)
                if belief_updater === nothing
                    belief_updater = updater(planner)
                end
                push!(queue, Sim(m,
                                planner,
                                belief_updater,
                                initialstate(m),
                                initial_state,
                                max_steps=max_steps,
                                metadata=Dict(:Epsiode=>i, :Solver=>j, :Param=>k)))
                if length(queue) >= max_queue_length
                    raw_data = process_queue!(queue, raw_data, labels, experiment_label, show_progress)
                end
            end
        end
    end
    process_queue!(queue, raw_data, labels, experiment_label, show_progress)
    return nothing::Nothing
end

function process_queue!(queue::Array, raw_data::DataFrame, labels::Array, experiment_label::String, show_progress::Bool)
    if length(queue) == 0
        return raw_data
    end

    println("Solving")
    data = run_parallel(queue, show_progress=show_progress) do sim, hist
        return (reward=discounted_reward(hist),) # Discounted reward is used. An alternative can be undiscounted_reward()
    end
    empty!(queue) # clear out queue for next sets of parameters
    raw_data = DataFrame([raw_data;data])

    for data in groupby(raw_data, :Solver)
        df = DataFrame()
        for subdata in groupby(data, :Param)
            rewards = subdata[!,:reward]
            label = labels[subdata[!,:Solver][1]][subdata[!,:Param][1]]
            push!(df, (label..., Mean=mean(rewards), SEM=std(rewards)/sqrt(nrow(subdata)), Confidence_Interval="(" * @sprintf("%.2f", quantile(rewards, 0.025)) * "," * @sprintf("%.2f", quantile(rewards, 0.975)) * ")"))
        end
        CSV.write("$(experiment_label)-$(labels[data[!,:Solver][1]][1][1].second).csv", df)
    end
    return raw_data
end

FuncSolver(;func::Union{Nothing, Function}=nothing) = FunctionSolver(func)

# end of module
end
module ParallelExperiment

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
    CSV,
    init_param

function parallel_experiment(pomdp::Union{POMDP, Function},
                             number_of_episodes::Int,
                             max_steps::Int,
                             solver_list::Array;
                             belief_updater::Any                        = nothing,
                             initial_belief::Any                        = nothing,
                             initial_state::Any                         = nothing,
                             show_progress::Bool                        = true,
                             max_queue_length::Int                      = 300,
                             full_factorial_design::Bool                = true,
                             solver_labels::Union{Array, Nothing}       = nothing,
                             solver_list_labels::Union{Array, Nothing}  = nothing,
                             proc_warn::Bool                            = true,
                             experiment_label::String                   = Dates.format(now(), "mmddHHMMSS"))

    # If no solver_labels are provided, then take the struct name of solvers as solver_labels.
    if solver_labels === nothing
        solver_labels = unique(collect(string(solver) for (solver, params) in solver_list))
    end

    # If no solver_list_labels are provided, then directly transform the values to string.
    if solver_list_labels === nothing
        solver_list_labels = []
        for (solver, params) in solver_list
            labels = []
            for (param, values) in params
                push!(labels, [string(value) for value in values])
            end
            push!(solver_list_labels, labels)
        end
    else
        # solver is expected to be of identical structure with solver_list_labels
        @assert length(solver_list) == length(solver_list_labels)
        for i in 1:length(solver_list)
            @assert length(solver_list[i].second) == length(solver_list_labels[i])
        end
    end

    println("Generating experimental design")

    # Generating the labels of param set.
    param_set_labels = []
    for i in 1:length(solver_list_labels)
        param_list_labels = solver_list_labels[i]
        param_list = solver_list[i].second
        if full_factorial_design
            labels = [[]]
            for j in 1:length(param_list_labels)
                value_labels = param_list_labels[j]
                param = param_list[j].first
                labels = [[a...,param=>b] for a in labels for b in value_labels]
            end
        else
            default_param_labels = []
            for j in 1:length(param_list_labels)
                value_labels = param_list_labels[j]
                param = param_list[j].first
                push!(default_param_labels, param=>value_labels[1])
            end
            labels = []
            for j in 1:length(param_list_labels)
                value_labels = param_list_labels[j]
                param = param_list[j].first
                for value_label in value_labels
                    push!(labels, [default_param_labels[1:j-1]..., param=>value_label, default_param_labels[j+1:end]...])
                end
            end
            labels = unique(labels)
        end 
        # Different sets of parameters for same solver will be merged
        isexist = false
        solver = solver_list[i].first
        for j in 1:length(param_set_labels)
            if param_set_labels[j].first == solver
                param_set_labels[j] = solver=>unique(vcat(param_set_labels[j].second, labels))
                isexist = true
                break
            end
        end
        if !isexist
            push!(param_set_labels, solver=>labels)
        end 
    end

    # Fuse solver_labels and param_set_labels for final output
    labels = []
    for i in 1:length(solver_labels)
        push!(labels, [[:Solver=>solver_labels[i], label...] for label in param_set_labels[i].second])
    end
    
    println("Simulations begin")
    queue = [] # simualtor queue
    raw_data = DataFrame() # stores the discounted_reward for each combination of Epsiode, Solver and Param.
    solved_solver_list = deepcopy(solver_list)
    if typeof(pomdp) <: POMDP 
        m = pomdp
        init_param_list!(solved_solver_list, solver_list, m)
        param_set = gen_param_set(solved_solver_list, full_factorial_design)
    end
    for i in 1:number_of_episodes
        println("Generating solvers for the $(i)-th episode.")
        # Generate a POMDP model if a generator is provided. This model is shared across all available parameter settings.
        if typeof(pomdp) <: Function 
            m = pomdp()
            init_param_list!(solved_solver_list, solver_list, m)
            param_set = gen_param_set(solved_solver_list, full_factorial_design)
        end
        for j in 1:length(param_set)
            solver, params = param_set[j]
            for k in 1:length(params)
                # Generate planners and belief_updater.
                planner = solve(solver(;params[k]...), m)
                if belief_updater === nothing
                    up = updater(planner)
                elseif typeof(belief_updater) <: Function
                    up = belief_updater(m)
                else
                    up = belief_updater
                end
                # Push a simulator into the queue.
                push!(queue, Sim(m,
                                planner,
                                up,
                                initialstate(m),
                                initial_state,
                                max_steps=max_steps,
                                metadata=Dict(:Epsiode=>i, :Solver=>j, :Param=>k)))
                # If the lenght of the queue surpass the max_queue_length, then start simulating.
                if length(queue) >= max_queue_length
                    raw_data = process_queue!(queue, raw_data, labels, experiment_label, show_progress, proc_warn)
                end
            end
        end
    end
    if length(queue) != 0
        # Perform a simulation at the end, if the queue is not empty.
        process_queue!(queue, raw_data, labels, experiment_label, show_progress, proc_warn)
    end
    return nothing::Nothing
end

init_param(m, param) = param

function init_param_list!(solved_solver_list, solver_list, pomdp)
    for i in 1:length(solver_list)
        param_list = solver_list[i].second
        solved_param_list = solved_solver_list[i].second
        for j in 1:length(param_list)
            values = param_list[j].second
            solved_values = solved_param_list[j].second
            for k in 1:length(values)
                solved_values[k] = init_param(pomdp, values[k])
            end
        end
    end
end

function gen_param_set(solver_list::Array, full_factorial_design::Bool)
    param_set = []
    for (solver, param_list) in solver_list
        if full_factorial_design
            params = [[]]
            for (param, values) in param_list
                params = [[a...,param=>b] for a in params for b in values]
            end
        else
            default_param = []
            for (param, values) in param_list
                push!(default_param, param=>values[1])
            end
            params = []
            for i in 1:length(param_list)
                param = param_list[i].first
                values = param_list[i].second
                for value in values
                    push!(params, [default_param[1:i-1]..., param=>value, default_param[i+1:end]...])
                end
            end
            params = unique(params)
        end 
        # Different sets of parameters for same solver will be merged
        isexist = false
        for i in 1:length(param_set)
            if param_set[i].first == solver
                param_set[i] = solver=>unique(vcat(param_set[j].second, params))
                isexist = true
                break
            end
        end
        if !isexist
            push!(param_set, solver=>params)
        end 
    end
    return param_set
end

function process_queue!(queue::Array, raw_data::DataFrame, labels::Array, experiment_label::String, show_progress::Bool, proc_warn::Bool)
    println("Solving")
    data = run_parallel(queue, show_progress=show_progress, proc_warn=proc_warn) do sim, hist
        return (reward=discounted_reward(hist),) # Discounted reward is used. An alternative can be undiscounted_reward()
    end
    empty!(queue) # clear out queue for next sets of parameters
    raw_data = DataFrame([raw_data;data]) # Merge newly generated data into raw_data

    for data in groupby(raw_data, :Solver)
        df = DataFrame()
        for subdata in groupby(data, :Param)
            rewards = subdata[!,:reward]
            # Within each subdata, the value of :Solver and :Param are identical, hence the first one is chosen.
            label = labels[subdata[!,:Solver][1]][subdata[!,:Param][1]]
            push!(df, (label..., Mean=mean(rewards), SEM=std(rewards)/sqrt(nrow(subdata)), Confidence_Interval="(" * @sprintf("%.2f", quantile(rewards, 0.025)) * "," * @sprintf("%.2f", quantile(rewards, 0.975)) * ")"))
        end
        # Save the results to the present work directory with the prefix of experiment_label.
        CSV.write("$(experiment_label)-$(labels[data[!,:Solver][1]][1][1].second).csv", df)
    end
    return raw_data
end

FuncSolver(;func::Union{Nothing, Function}=nothing) = FunctionSolver(func)

# end of module
end
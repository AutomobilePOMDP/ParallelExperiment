module ParallelExperiment

using POMDPs
using POMDPPolicies # For function policy and random policy
using POMDPSimulators # For parallel simulator

using Statistics
using DataFrames
using CSV
using Printf
using Dates
using Distributed
using ProgressMeter

export
    parallel_experiment,
    FuncSolver,
    CSV,
    init_param

function parallel_experiment(pomdp::Union{POMDP, Function},
                             episodes_per_domain::Int,
                             max_steps::Int,
                             solver_list::Array;
                             num_of_domains::Int                        = 1,
                             belief_updater::Any                        = nothing,
                             initial_belief::Any                        = nothing,
                             initial_state::Any                         = nothing,
                             show_progress::Bool                        = true,
                             max_queue_length::Int                      = 300,
                             domain_queue_length::Int                   = 5,
                             full_factorial_design::Bool                = true,
                             solver_labels::Union{Array, Nothing}       = nothing,
                             solver_list_labels::Union{Array, Nothing}  = nothing,
                             proc_warn::Bool                            = true,
                             discount::Bool                             = true,
                             experiment_label::String                   = Dates.format(now(), "mmddHHMMSS"))
    
    if typeof(pomdp) <: POMDP 
        if num_of_domains != 1
            error("If you intend to use multiple domains, you should provide a function for generating domains.")
        end
        m = pomdp
        pomdp = ()->m
    end

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

    println("Generating experimental designs.")

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
    
    println("Experiments begin.")
    sim_queue = Array{Any, 1}[]
    raw_data = DataFrame() # stores the discounted_reward for each combination of Epsiode, Solver and Param.
    for i in 1:domain_queue_length:num_of_domains
        println("Initializing domains $(i) to $(min(i+domain_queue_length-1, num_of_domains)).")
        params = Any[]
        domain_queue = Any[]
        for j in 1:min(domain_queue_length, num_of_domains-i+1)
            m = pomdp()
            push!(domain_queue, m)
            for (solver, param_list) in solver_list
                for (param, values) in param_list
                    for value in values
                        push!(params, (m, value))
                    end
                end
            end
        end
        map_function(args...) = (show_progress ? progress_pmap(args..., progress=Progress(length(params), desc="Initializing parameters...")) : pmap(args...))
        initialized_params = map_function((args)->init_param(args...), params)
        initialized_solver_list = Pair{Any, Array{Pair{Symbol, Array{Any, 1}}, 1}}[]
        for j in 1:min(domain_queue_length, num_of_domains-i+1)
            empty!(initialized_solver_list)
            for (solver, param_list) in solver_list
                initialized_param_list = Pair{Symbol, Array{Any, 1}}[]
                for (param, values) in param_list
                    initialized_values = Any[]
                    for value in values
                        push!(initialized_values, initialized_params[1])
                        initialized_params = initialized_params[2:end]
                    end
                    push!(initialized_param_list, param=>initialized_values)
                end
                push!(initialized_solver_list, solver=>initialized_param_list)
            end
            param_set = gen_param_set(initialized_solver_list, full_factorial_design)
            m = domain_queue[1]
            domain_queue = domain_queue[2:end]
            b0 = initial_belief === nothing ? initialstate(m) : (typeof(initial_belief) <: Function ? initial_belief(m) : initial_belief)
            for u in 1:episodes_per_domain
                s0 = initial_state === nothing ? rand(b0) : (typeof(initial_state) <: Function ? initial_state(m) : initial_state)
                No_domain = (i-1+j-1)*episodes_per_domain+u
                println("Preparing for the $(No_domain)-th episode.")
                for (v, (solver, params)) in enumerate(param_set)
                    for (w, param) in enumerate(params)
                        push!(sim_queue, [m, solver, belief_updater, b0, s0, max_steps, No_domain, v, w, param])
                        # If the lenght of the sim_queue surpass the max_queue_length, then start simulating.
                        if length(sim_queue) >= max_queue_length
                            raw_data = process_queue!(sim_queue, raw_data, labels, experiment_label, show_progress, proc_warn, discount)
                        end
                    end
                end
            end
        end
    end
    if length(sim_queue) != 0
        # Perform a simulation at the end, if the sim_queue is not empty.
        process_queue!(sim_queue, raw_data, labels, experiment_label, show_progress, proc_warn, discount)
    end
    return nothing::Nothing
end

init_param(m, param) = param
init_param(m, param::S) where S <: Solver = solve(param, m)

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
                param_set[i] = solver=>unique(vcat(param_set[i].second, params))
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

function solve_sim(m::POMDP, solver::Any, belief_updater::Any, b0::Any, s0::Any, max_steps::Int, i::Int, j::Int, k::Int, param::Array{P,1}) where P <: Pair
    # Generate planners and belief_updater.
    planner = solve(solver(;param...), m)
    if belief_updater === nothing
        up = updater(planner)
    elseif typeof(belief_updater) <: Function
        up = belief_updater(m)
    else
        up = belief_updater
    end

    Sim(m,
        planner,
        up,
        b0,
        s0,
        max_steps=max_steps,
        metadata=Dict(:Epsiode=>i, :Solver=>j, :Param=>k))
end

function process_queue!(sim_queue::Array{Array{Any,1},1}, raw_data::DataFrame, labels::Array, experiment_label::String, show_progress::Bool, proc_warn::Bool, discount::Bool)
    map_function(args...) = (show_progress ? progress_pmap(args..., progress=Progress(length(sim_queue), desc="Generating simulators...")) : pmap(args...))
    solved_sim_queue = map_function((sim)->solve_sim(sim...), sim_queue)
    data = run_parallel(solved_sim_queue, show_progress=show_progress, proc_warn=proc_warn) do sim, hist
        return (reward=discount ? discounted_reward(hist) : undiscounted_reward(hist),)
    end
    empty!(sim_queue) # clear out queue for next sets of parameters
    raw_data = DataFrame([raw_data;data]) # Merge newly generated data into raw_data

    for data in groupby(raw_data, :Solver)
        df = DataFrame()
        for subdata in groupby(data, :Param)
            rewards = subdata[!,:reward]
            # Within each subdata, the value of :Solver and :Param are identical, hence the first one is chosen.
            label = labels[subdata[!,:Solver][1]][subdata[!,:Param][1]]
            push!(df, (label..., AvgReturn=mean(rewards), SEM=std(rewards)/sqrt(nrow(subdata)), CI95="(" * @sprintf("%.2f", quantile(rewards, 0.025)) * "," * @sprintf("%.2f", quantile(rewards, 0.975)) * ")"))
        end
        # Save the results to the present work directory with the prefix of experiment_label.
        CSV.write("$(experiment_label)-$(labels[data[!,:Solver][1]][1][1].second).csv", df)
    end
    return raw_data
end

struct FuncSolver <: Solver
    func::Function
end
FuncSolver(;func::Function) = FuncSolver(func)
function POMDPs.solve(solver::FuncSolver, pomdp::POMDP)
    FunctionPolicy(b->(solver.func(pomdp, b)))
end

# end of module
end
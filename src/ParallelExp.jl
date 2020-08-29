module ParallelExp

using POMDPs
using POMDPPolicies # For function policy and random policy
using POMDPSimulators # For parallel simulator

using Statistics
using DataFrames
using CSV
using Printf

export
    parallel_experiment,
    FuncSolver,
    CSV

function parallel_experiment(pomdp::POMDP,
                             number_of_episodes::Int,
                             max_steps::Int,
                             solver_list::Array;
                             belief_updater::Union{Updater,Nothing}     = nothing,
                             initial_belief::Any                        = initialstate_distribution(pomdp),
                             initialstate::Any                          = nothing,
                             show_progress::Bool                        = true,
                             full_factorial_design::Bool                = true,
                             auto_save::Bool                            = true)

    println("Generating experimental design")
    solvers = []
    for (solver, param_dict) in solver_list
        params = [Dict()]
        if full_factorial_design
            for param in param_dict
                params = [Dict{Symbol, Any}(a...,param.first=>b) for a in params for b in param.second]
            end
        else
            default_param = Dict(k=>v[1] for (k,v) in param_dict)
            params = unique(vcat([[Dict{Symbol, Any}(default_param..., k=>value) for value in v] for (k, v) in param_dict]...))
        end 
        # Different sets of parameters for same solver will be merged
        isexist = false
        for i in 1:length(solvers)
            if solvers[i].first == solver
                solvers[i] = solver=>unique(vcat(solvers[i].second, params))
                isexist = true
                break
            end
        end
        if !isexist
            push!(solvers, solver=>params)
        end 
    end
    
    println("Simulations begin")
    dfs = []
    for (solver, params) in solvers
        rewards = []
        queue = []
        df = DataFrame()
        for i = 1:length(params)
            println("Preparing simulators for the $(i)-th set of parameters of the $(string(solver))")
            if belief_updater === nothing
                planner = solve(solver(;params[i]...), pomdp)
                belief_updater = updater(planner)
            end
            for j = 1:number_of_episodes
                planner = solve(solver(;params[i]...), pomdp)
                push!(queue, Sim(pomdp, planner, belief_updater, initial_belief, initialstate, max_steps=max_steps, metadata=Dict(:No=>i)))
            end
            if length(queue) < 500 && i != length(params) # queue should be large so that CPU is always busy
                continue
            end
            println("Solving")
            data = run_parallel(queue, show_progress=show_progress) do sim, hist
                return (reward=discounted_reward(hist),) # Discounted reward is used. An alternative can be undiscounted_reward()
            end
            queue = [] # clear out queue for next sets of parameters
            for i in unique(data[!, :No])
                push!(rewards, data[data[!, :No] .== i,:reward])
                params[i] = Dict(:solver=>string(solver), [k=>string(v) for (k,v) in params[i]]...)
                push!(df, (params[i]..., mean=mean(rewards[i]), std=std(rewards[i]), confidence_interval="(" * @sprintf("%.2f", quantile(rewards[i], 0.025)) * "," * @sprintf("%.2f", quantile(rewards[i], 0.975)) * ")"))
            end
            if auto_save
                CSV.write("$(string(solver)).csv", df)
            end
        end
        push!(dfs, df)
    end
    return dfs
end

FuncSolver(;func::Union{Nothing, Function}=nothing) = FunctionSolver(func)




# end of module
end
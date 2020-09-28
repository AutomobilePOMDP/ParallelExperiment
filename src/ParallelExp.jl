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
                             max_queue_length::Int                      = 300,
                             full_factorial_design::Bool                = true,
                             auto_save::Bool                            = true)

    println("Generating experimental design")
    solvers = []
    for i in 1:length(solver_list)
        solver = solver_list[i].first
        param_list = solver_list[i].second
        if full_factorial_design
            params = [[]]
            for param in param_list
                params = [[a...,param.first=>b] for a in params for b in param.second]
            end
        else
            default_param = [k=>v[1] for (k,v) in param_list]
            params = []
            for k in 1:length(param_list)
                for value in param_list[k].second
                    push!(params, [default_param[1:k-1]..., param_list[k].first=>value, default_param[k+1:end]...])
                end
            end
            params = unique(params)
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
            if length(queue) < max_queue_length && i != length(params) # queue should be large so that CPU is always busy
                continue
            end
            println("Solving")
            data = run_parallel(queue, show_progress=show_progress) do sim, hist
                return (reward=discounted_reward(hist),) # Discounted reward is used. An alternative can be undiscounted_reward()
            end
            queue = [] # clear out queue for next sets of parameters
            for i in unique(data[!, :No])
                push!(rewards, data[data[!, :No] .== i,:reward])
                params[i] = [:solver=>string(solver), [k=>string(v) for (k,v) in params[i]]...]
                push!(df, (params[i]..., Mean=mean(rewards[i]), SEM=std(rewards[i])/sqrt(number_of_episodes), Confidence_Interval="(" * @sprintf("%.2f", quantile(rewards[i], 0.025)) * "," * @sprintf("%.2f", quantile(rewards[i], 0.975)) * ")"))
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
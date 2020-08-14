# initialize DESPOT env
import Pkg
Pkg.cd("ARDESPOT.jl/notebook/")
Pkg.activate("..")

# Initialize workers
using Distributed
addprocs(exeflags="--project") # initial workers with the project env in current work directory

# POMCP
@everywhere using BasicPOMCP

# DESPOT
@everywhere using ARDESPOT # ARDESPOT pkg

# UCT-DESPOT
@everywhere push!(LOAD_PATH, "../../UCT-DESPOT")
@everywhere using UCTDESPOT # UCT-DESPOT pkg

# QMDP
using QMDP

# POMDP related pkgs
@everywhere using POMDPs # Basic POMDP framework
@everywhere using POMDPSimulators # For parallel simulator
using POMDPPolicies # For function policy and random policy
@everywhere using ParticleFilters # For simple particle filter
using BeliefUpdaters # For roomba and BabyPOMDP belief updater

# For visualization
using D3Trees
using POMDPModelTools
using POMDPGifs

# For data processing and storing
using Statistics
using DataFrames
using CSV
using Random
using Printf

# It's essential to activate multiple workers with following codes so as to enable the parallel computation.
# using Distributed
# addprocs()
function parallel_experiment(pomdp, number_of_episodes, max_steps, solver_list;
        belief_updater=nothing, show_progress=true, full_factorial_design=true, auto_save=true)
    println("Generating experimental design")
    solvers = []
    for (solver, param_dict) in solver_list
        params = [Dict()]
        if full_factorial_design
            for param in param_dict
                params = [Dict{Symbol, Any}(a...,param.first=>b) for a in params for b in param.second]
            end
        else
            default_param = Dict(k=>rand(v) for (k,v) in param_dict)
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
            println("Preparing solvers for the $(i)-th set of parameters of the $(string(solver))")
            for j = 1:number_of_episodes
                planner = solve(solver(;params[i]...), pomdp)
                if(belief_updater==nothing)
                    push!(queue, Sim(pomdp, planner, max_steps=max_steps, metadata=Dict(:No=>i)))
                else
                    push!(queue, Sim(pomdp, planner, belief_updater, max_steps=max_steps, metadata=Dict(:No=>i)))
                end
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
                push!(df, (params[i]..., mean=mean(rewards[i]), std=std(rewards[i]), confidence_interval=[@sprintf("%.2f", i) for i in quantile(rewards[i], [0.025,0.975])]))
            end
            if auto_save
                CSV.write("$(string(solver)).csv", df)
            end
        end
        push!(dfs, df)
    end
    return dfs
end
FuncSolver(;func=nothing) = FunctionSolver(func)

# Roomba related pkgs
# Roomba need ParticleFilters = "0.2" for compatibility
@everywhere push!(LOAD_PATH, "../../AA228FinalProject")
@everywhere using AA228FinalProject # For Roomba Env

max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0

cut_points =  exp10.(range(-.5, stop=1.3, length=10))
sensor = DiscreteLidar(cut_points)

num_particles = 5000 # number of particles in belief

pos_noise_coeff = 0.3
ori_noise_coeff = 0.1

# POMDP problem
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
pomdp = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=3, aspace=action_space));

# Belief updater
resampler = LidarResampler(num_particles, pomdp, pos_noise_coeff, ori_noise_coeff)
belief_updater = BasicParticleFilter(pomdp, resampler, num_particles)

# Running policy
running_policy = FunctionPolicy() do b
    # s = typeof(b) == RoombaState ? b : typeof(b) <: AA228FinalProject.RoombaInitialDistribution ? rand(b) : mean(b)
    # The statement is computational inefficient.
    s = typeof(b) == RoombaState ? b : rand(b)
    # compute the difference between our current heading and one that would
    # point to the goal
    goal_x, goal_y = get_goal_xy(pomdp)
    x,y,th = s[1:3]
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal - th)
    
    # apply proportional control to compute the turn-rate
    Kprop = 1.0
    om = Kprop * del_angle
    # find the closest option in action space
    _,ind = findmin(abs.(om .- (-max_turn_rate:turn_rate_interval:max_turn_rate)))
    om = (-max_turn_rate:turn_rate_interval:max_turn_rate)[ind]
    # always travel at some fixed velocity
    v = max_speed
    
    return RoombaAct(v, om)
end

# For DESPOT
bounds = IndependentBounds(DefaultPolicyLB(running_policy), 10.0, check_terminal=true)
random_bounds = IndependentBounds(DefaultPolicyLB(RandomPolicy(pomdp)), 10.0, check_terminal=true)
despot_dict = Dict(:default_action=>[running_policy,], 
                    :bounds=>[random_bounds,],
                    :lambda=>[0.0, 0.01, 0.1, 1.0],
                    :T_max=>[10.0],
                    :K=>[100, 300],
                    :beta=>[0., 0.5, 1., 5.])

# For UCT-DESPOT
rollout_policy = running_policy
random_rollout_policy = RandomPolicy(pomdp)
uctdespot_dict = Dict(:rollout_policy=>[rollout_policy, random_rollout_policy],
                        :K=>[100, 300],
                        :T_max=>[10.0],
                        :m=>[10, 30],
                        :c=>[1., 10., 100.])

# For POMCP
value_estimator = FORollout(running_policy)
random_value_estimator = FORollout(RandomPolicy(pomdp))
pomcp_dict = Dict(:estimate_value=>[value_estimator],
                    :tree_queries=>[100000,], 
                    :max_time=>[10.0,], 
                    :c=>[10., 100.])

# Solver list
solver_list = [DESPOTSolver=>despot_dict, 
                UCT_DESPOTSolver=>uctdespot_dict, 
                POMCPSolver=>pomcp_dict]

                
number_of_episodes = 100
max_steps = 100

dfs = parallel_experiment(pomdp, number_of_episodes, max_steps, solver_list, belief_updater=belief_updater, full_factorial_design=false)

CSV.write("DiscreteLidarRoomba_DESPOT.csv", dfs[1])
CSV.write("DiscreteLidarRoomba_UCT_DESPOT.csv", dfs[2])
CSV.write("DiscreteLidarRoomba_POMCP.csv", dfs[3])
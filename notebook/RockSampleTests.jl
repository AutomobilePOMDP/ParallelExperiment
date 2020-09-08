# load the packages
import Pkg
Pkg.cd("..")
Pkg.activate(".")

# global variables
max_workers = 10
rock_nums = [15, 11, 8]

# create n workers
using Distributed
addprocs(max_workers, exeflags="--project")

# the algorithms tested in experiments
@everywhere[
    # POMDP related pkgs
    using POMDPs
    using QMDP
    # POMCP
    using POMCPOW
    using BasicPOMCP
    # DESPOT
    push!(LOAD_PATH, "../LB-DESPOT")
    using LBDESPOT # LBDESPOT pkg
    # UCT-DESPOT
    push!(LOAD_PATH, "../UCT-DESPOT")
    using UCTDESPOT # UCT-DESPOT pkg
    # Rocksample pkg
    using RockSample

    using ParallelExp
    using BasicPOMCP
    using POMDPPolicies # For function policy and random policy
    using ParticleFilters
    using BeliefUpdaters # For belief updater 
    using Random
]

Random.seed!(0)
# set up the environment
maps = [(7, 7), (11, 11), (15, 15)]
pomdps = []
for map in maps
    current_rocks = []
    possible_ps = [(i, j) for i in range(1, length=map[1]), j in range(1, length=map[2])]
    selected = rand(1:map[1]*map[2], pop!(rock_nums))
    for pos in selected
        rock = possible_ps[pos]
        push!(current_rocks, rock)
    end
    pomdp = RockSamplePOMDP(map_size=map, rocks_positions=current_rocks)
    push!(pomdps, pomdp)
end

# test algorithms on environments
let 
    k = 1
    for pomdp in pomdps
        println(k)
        # QMDP upper bound
        qmdp_policy = solve(QMDPSolver(), pomdp)
        @everywhere function qmdp_upper_bound(pomdp, b)
            return value($qmdp_policy, b)
        end

        # default policy
        move_east = FunctionPolicy() do b
            return 2
        end

        # better default policy
        to_best = FunctionPolicy() do b 
            if typeof(b) <: RSState 
                s = b 
                val, ind = findmax(s.rocks) 
            else 
                s = rand(b) 
                good_count = zeros(Int, length(s.rocks)) 
                for state in particles(b) 
                    good_count += state.rocks 
                end 
                val, ind = findmax(good_count) 
            end 
            if val/length(s.rocks) < 0.5 
                return 2 
            end 
            rock_pos = pomdp.rocks_positions[ind]
            diff = rock_pos - s.pos 
            if diff[2] != 0 
                if sign(diff[2]) == 1 
                    return 1 # to north 
                else 
                    return 3 # to south 
                end 
            else 
                if sign(diff[1]) == 1 
                    return 2 # to east 
                elseif sign(diff[1]) == -1 
                    return 4 # to west 
                else 
                    return 5 # sample 
                end 
            end 
        end

        # For LB-DESPOT
            random_bounds = IndependentBounds(DefaultPolicyLB(RandomPolicy(pomdp)), 40.0, check_terminal=true)
            bounds = IndependentBounds(DefaultPolicyLB(to_best), 40.0, check_terminal=true)
            bounds_hub = IndependentBounds(DefaultPolicyLB(to_best), qmdp_upper_bound, check_terminal=true)
            lbdespot_dict1 = Dict(:default_action=>[to_best,], 
                                :bounds=>[bounds],
                                :K=>[100],
                                :beta=>[0.5])
            lbdespot_dict2 = Dict(:default_action=>[to_best,], 
                                :bounds=>[random_bounds],
                                :K=>[100],
                                :beta=>[0.3])
            lbdespot_dict3 = Dict(:default_action=>[to_best,], 
                                :bounds=>[bounds_hub],
                                :K=>[100],
                                :beta=>[0.5])
        # For UCT-DESPOT
            # random_rollout_policy = RandomPolicy(pomdp)
            # rollout_policy = to_best
            # uctdespot_dict1 = Dict(:default_action=>[RandomPolicy(pomdp),],
            #                     :rollout_policy=>[random_rollout_policy],
            #                     :max_trials=>[100000,],
            #                     :K=>[1000, 2000],
            #                     :m=>[50, 100],
            #                     :c=>[1, 10.])
            # uctdespot_dict = Dict(:default_action=>[RandomPolicy(pomdp),],
            #                     :rollout_policy=>[random_rollout_policy],
            #                     :max_trials=>[100000,],
            #                     :K=>[300, 100, 500],
            #                     :m=>[50, 30],
            #                     :c=>[1.,10,])
        # For POMCPOW
            # random_value_estimator = FORollout(RandomPolicy(pomdp))
            # value_estimator = FORollout(to_best)
            # pomcpow_dict = Dict(:default_action=>[RandomPolicy(pomdp),],
            #                     :estimate_value=>[random_value_estimator],
            #                     :tree_queries=>[200000,], 
            #                     :max_time=>[1.0,],
            #                     :criterion=>[MaxUCB(10.),])

        # Solver list
            solver_list = [
                LB_DESPOTSolver=>lbdespot_dict1, 
                LB_DESPOTSolver=>lbdespot_dict2, 
                LB_DESPOTSolver=>lbdespot_dict3, 
                # UCT_DESPOTSolver=>uctdespot_dict, 
                # POMCPOWSolver=>pomcpow_dict
            ]

        number_of_episodes = 100
        max_steps = 300
        # Pkg.cd("notebook")
        dfs = parallel_experiment(pomdp,
                                number_of_episodes,
                                max_steps, solver_list,
                                full_factorial_design=false)
        CSV.write("RockSample_DESPOT_$k.csv", dfs[1])
        # CSV.write("RockSample_DESPOT2_$k.csv", dfs[2])
        # CSV.write("RockSample_DESPOT3_$k.csv", dfs[3])
        # CSV.write("RockSample_UCT_DESPOT_$k.csv", dfs[2])
        # CSV.write("RockSample_POMCP_$k.csv", dfs[3])
        k += 1
    end
end
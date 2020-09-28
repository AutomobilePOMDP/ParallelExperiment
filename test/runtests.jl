num_of_procs = 10 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs)

using ParallelExp

@everywhere using POMDPs # Basic POMDP framework
@everywhere using RockSample
@everywhere using POMCPOW
using BasicPOMCP
using POMDPPolicies
using Random

# POMDP problem
pomdp = RockSamplePOMDP()

# For POMCP
random_value_estimator = FORollout(RandomPolicy(pomdp))
pomcpow_list = [:estimate_value=>[random_value_estimator],
                    :tree_queries=>[100000,], 
                    :max_time=>[1.0,], 
                    :criterion=>[MaxUCB(0.1), MaxUCB(1.0), MaxUCB(10.), MaxUCB(100.), MaxUCB(1000.)]]

# Solver list
solver_list = [POMCPOWSolver=>pomcpow_list,]

number_of_episodes = 2
max_steps = 5

dfs = parallel_experiment(pomdp,
                          number_of_episodes,
                          max_steps,
                          solver_list,
                          full_factorial_design=false)
for i in 1:length(dfs)
    CSV.write("BumperRoomba$(i).csv", dfs[i])
end

num_of_procs = 10 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs)


@everywhere using POMDPs # Basic POMDP framework
@everywhere using RockSample
@everywhere using POMCPOW
@everywhere using DiscreteValueIteration
@everywhere using POMDPModelTools
using BasicPOMCP
using POMDPPolicies
using Random
using ParticleFilters

@everywhere using ParallelExperiment

# POMDP problem
pomdp = RockSamplePOMDP()

# For POMCP
random_value_estimator = FORollout(RandomPolicy(pomdp))
@everywhere mutable struct MDPEstimator
    solver::ValueIterationSolver
    policy::Union{Nothing, ValueIterationPolicy}
end
MDPEstimator(solver) = MDPEstimator(solver, nothing)
@everywhere function ParallelExperiment.init_param(m, param::MDPEstimator)
    m = UnderlyingMDP(m)
    param.policy = solve(param.solver, m)
end
@everywhere function POMCPOW.estimate_value(o::MDPEstimator, pomdp, s, h, steps)
    value(o.policy, s)
end
pomcpow_list = [:estimate_value=>[random_value_estimator, MDPEstimator(ValueIterationSolver())],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :criterion=>[MaxUCB(0.1), MaxUCB(1.0), MaxUCB(10.), MaxUCB(100.), MaxUCB(1000.)]]
pomcpow_list_labels = [["Random Rollout", "MDP"],
                    [100000,], 
                    [1.0,], 
                    [0.1, 1.0, 10., 100., 1000.]]

# Solver list
solver_list = [POMCPOWSolver=>pomcpow_list,]
solver_list_labels = [pomcpow_list_labels,]

number_of_episodes = 2
max_steps = 5

parallel_experiment(pomdp,
                    number_of_episodes,
                    max_steps,
                    solver_list,
                    experiment_label="test",
                    full_factorial_design=false)

parallel_experiment(pomdp,
                    number_of_episodes,
                    max_steps,
                    solver_list,
                    solver_labels=["POMCP",],
                    full_factorial_design=false)

maps = [(7, 8), (11, 11)]
for map in maps
    parallel_experiment(number_of_episodes,
                        max_steps,
                        solver_list,
                        solver_list_labels=solver_list_labels,
                        belief_updater=(m)->SIRParticleFilter(m, 20000),
                        full_factorial_design=false) do

        possible_ps = [(i, j) for i in 1:map[1], j in 1:map[1]]
        selected = unique(rand(possible_ps, map[2]))
        while length(selected) != map[2]
            push!(selected, rand(possible_ps))
            selected = unique!(selected)
        end
        return RockSamplePOMDP(map_size=(map[1],map[1]), rocks_positions=selected)
    end
end
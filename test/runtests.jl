num_of_procs = 6 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs)
if isdir("results")
    rm("results", recursive=true)
end
mkdir("results")
cd("results")

@everywhere using POMDPs # Basic POMDP framework
@everywhere using RockSample
@everywhere using POMCPOW
@everywhere using DiscreteValueIteration
@everywhere using POMDPModelTools
@everywhere using ParticleFilters
using BasicPOMCP
using POMDPPolicies
using Random

@everywhere using ParallelExperiment

# POMDP problem
pomdp = RockSamplePOMDP()

# For POMCP
random_value_estimator = FORollout(RandomPolicy(pomdp))
@everywhere function ParallelExperiment.init_param(m, param::FORollout)
    return typeof(param.solver) <: POMDPs.Solver ? FORollout(solve(param.solver, UnderlyingMDP(m))) : param
end
pomcpow_list = [:estimate_value=>[random_value_estimator, FORollout(ValueIterationSolver())],
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

episodes_per_domain = 2
max_steps = 5

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    experiment_label="test",
                    full_factorial_design=false)

# parallel_experiment(pomdp,
#                     episodes_per_domain,
#                     max_steps,
#                     solver_list,
#                     solver_labels=["POMCP",],
#                     belief_updater=SIRParticleFilter(pomdp, 20000),
#                     full_factorial_design=false)

maps = [(5, 5)]
max_steps = 1
episodes_per_domain = 10
for map in maps
    parallel_experiment(episodes_per_domain,
                        max_steps,
                        solver_list,
                        num_of_domains=15,
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
cd("..")
rm("results", recursive=true)
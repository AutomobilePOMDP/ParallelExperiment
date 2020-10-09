# Parallel Experiment
ParallelExp is a module providing necessary tools for parallel experiments. It features experiments with different POMDP solvers and different sets of parameters for each solver.
## Update
The pomdp can now be a Function for generating environment. For example,
```julia
maps = [(7, 8), (11, 11), (15, 15)]
for map in maps
    dfs = parallel_experiment(number_of_episodes,
                            max_steps,
                            solver_list,
                            full_factorial_design=false) do

        possible_ps = [(i, j) for i in 1:map[1], j in 1:map[1]]
        selected = unique(rand(possible_ps, map[2]))
        while length(selected) !== map[2]
            push!(selected, rand(possible_ps))
            selected = unique!(selected)
        end
        return RockSamplePOMDP(map_size=map, rocks_positions=selected)
    end
end
```
Or you can explicitly input a function, for example,
```julia
function rsgen()
    possible_ps = [(i, j) for i in 1:map[1], j in 1:map[1]]
    selected = unique(rand(possible_ps, map[2]))
    while length(selected) !== map[2]
        push!(selected, rand(possible_ps))
        selected = unique!(selected)
    end
    return RockSamplePOMDP(map_size=map, rocks_positions=selected)
end
dfs = parallel_experiment(rsgen,
                        number_of_episodes,
                        max_steps,
                        solver_list,
                        full_factorial_design=false)
```

## Installation
```bash
add https://github.com/AutomobilePOMDP/ParallelExperiment
```
## Usage
```julia
# Initialize multiple workers for parallel experiment
num_of_procs = 10 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs) # initial workers with the project env in current work directory

# using Pkg
# Pkg.add("https://github.com/AutomobilePOMDP/ParallelExperiment")
using ParallelExp

# Make sure all your solvers are loaded in every procs
@everywhere using POMCPOW
using BasicPOMCP

# Pkg.add("https://github.com/AutomobilePOMDP/PL-DESPOT")
@everywhere using PL_DESPOT # PL-DESPOT pkg

# Make sure these pkgs are loaded in every procs
@everywhere using POMDPs # Basic POMDP framework
@everywhere using ParticleFilters # For simple particle filter

### Setting up an POMDP problem ###

# The following codes is quoted from tests.ipynb, you can check the the detail there.

pldespot_list = [:default_action=>[rush_policy,], 
                    :bounds=>[bounds, random_bounds],
                    :K=>[100, 300, 500],
                    :beta=>[0., 0.1, 1., 10., 100.]]
pomcpow_list = [:estimate_value=>[random_value_estimator],
                    :tree_queries=>[100000,], 
                    :max_time=>[1.0,], 
                    :criterion=>[MaxUCB(0.1), MaxUCB(1.0), MaxUCB(10.), MaxUCB(100.), MaxUCB(1000.)]]
solver_list = [ PL_DESPOTSolver=>pldespot_list, 
                POMCPOWSolver=>pomcpow_list,
                FuncSolver=>[:func=>[rush_policy,],]

number_of_episodes = 100
max_steps = 100

dfs = parallel_experiment(pomdp,
                          number_of_episodes,
                          max_steps,
                          solver_list,
                          belief_updater=belief_updater,
                          full_factorial_design=false)
for i in 1:length(dfs)
    CSV.write("BumperRoomba$(i).csv", dfs[i])
end
```
## Key parameters
- belief_updater: Customized belief updater for specific POMDP problem
- initial_belief: Customized initial belief for specific POMDP problem
- initialstate: Customized initial state for specific POMDP problem
- show_progress: Whether to show progress. Default to true.
- full_factorial_design: Whether to enable full factorial design. Default to true. Full factorial design means every possible combination of parameters will be tested. For example, ```paramlist = [:a=>[1,2], :b=>[1,2]]```, then four sets of experiments will be done ```[[:a=>1,:b=>1], [:a=>1,:b=>2], [:a=>2, :b=>1], [:a=>2,:b=>2]]```. If the full factorial design is set to false, then changes will be made to the first set of parameter one at a time. For example, ```paramlist = [:a=>[1,2,3], :b=>[1,2,3]]```, then five sets of experiments will be done, ```[[:a=>1,:b=>1], [:a=>2,:b=>1], [:a=>3,:b=>1], [:a=>1,:b=>2], [:a=>1,:b=>3]]```.
- auto_save: Automatically save the result in "SolverName.csv" every 500 episodes. This will have minor influence on performance.

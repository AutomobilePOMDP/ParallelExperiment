# Parallel Experiment
ParallelExp is a module providing necessary tools for parallel experiments. It features experiments with different POMDP solvers and different sets of parameters for each solver.
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
                FuncSolver=>Dict(:func=>[rush_policy,])]

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
- full_factorial_design: Whether to enable full factorial design. Default to true. Full factorial design means every possible combination of parameters will be tested. For example, ```paramdict = Dict(:a=>[1,2], :b=>[1,2])```, then four sets of experiments will be done ```[Dict(:a=>1,:b=>1), Dict(:a=>1,:b=>2), Dict(:a=>2, :b=>1), Dict(:a=>2,:b=>2)]```. If the full factorial design is set to false, then changes will be made to the first set of parameter one at a time. For example, ```paramdict = Dict(:a=>[1,2,3], :b=>[1,2,3])```, then five sets of experiments will be done, ```[Dict(:a=>1,:b=>1), Dict(:a=>2,:b=>1), Dict(:a=>3,:b=>1), Dict(:a=>1,:b=>2), Dict(:a=>1,:b=>3)]```.
- auto_save: Automatically save the result in "SolverName.csv" every 500 episodes. This will have minor influence on performance.

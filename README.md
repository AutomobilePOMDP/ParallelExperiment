# Parallel Experiment
ParallelExp is a module providing necessary tools for parallel experiments. It features experiments with different POMDP solvers and different sets of parameters for each solver.
## Installation
```bash
git clone https://github.com/AutomobilePOMDP/ParallelExperiment
cd ParallelExperiment
julia
```
```julia
Pkg> add POMDPs
Pkg> registry add https://github.com/JuliaPOMDP/Registry
Pkg> activate .
Pkg> instantiate
Pkg> precompile
```
## Usage
```julia
# Initialize multiple workers for parallel experiment
num_of_procs = 10 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs) # initial workers with the project env in current work directory

@everywhere push!(LOAD_PATH, "../ParallelExp")
using ParallelExp

# Make sure all your solvers are loaded in every procs
@everywhere using POMCPOW
using BasicPOMCP
@everywhere push!(LOAD_PATH, "../LB-DESPOT/")
@everywhere using LBDESPOT # LB-DESPOT pkg
@everywhere push!(LOAD_PATH, "../UCT-DESPOT/")
@everywhere using UCTDESPOT # UCT-DESPOT pkg

# Make sure these pkgs are loaded in every procs
@everywhere using POMDPs # Basic POMDP framework
@everywhere using ParticleFilters # For simple particle filter

# Make sure your POMDP model is loaded in every procs
@everywhere push!(LOAD_PATH, "../Roomba")
@everywhere using Roomba # For Roomba Env

### Setting up an POMDP problem ###

# The following codes is quoted from tests.ipynb, you can check the the detail there.

lbdespot_dict = Dict(:default_action=>[rush_policy,], 
                    :bounds=>[bounds, random_bounds],
                    :K=>[100, 300, 500],
                    :beta=>[0., 0.1, 1., 10., 100.])
solver_list = [LB_DESPOTSolver=>lbdespot_dict, 
                UCT_DESPOTSolver=>uctdespot_dict, 
                POMCPOWSolver=>pomcpow_dict,
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
- show_progress: Whether to show progress. Default to true.
- full_factorial_design: Whether to enable full factorial design. Default to true. Full factorial design means every possible combination of parameters will be tested. For example, ```paramdict = Dict(:a=>[1,2], :b=>[1,2])```, then four sets of experiments will be done ```[Dict(:a=>1,:b=>1), Dict(:a=>1,:b=>2), Dict(:a=>2, :b=>1), Dict(:a=>2,:b=>2)]```. If the full factorial design is set to false, then changes will be made to the first set of parameter one at a time. For example, ```paramdict = Dict(:a=>[1,2,3], :b=>[1,2,3])```, then five sets of experiments will be done, ```[Dict(:a=>1,:b=>1), Dict(:a=>2,:b=>1), Dict(:a=>3,:b=>1), Dict(:a=>1,:b=>2), Dict(:a=>1,:b=>3)]```.
- auto_save: Automatically save the result in "SolverName.csv" every 500 episodes. This will have minor influence on performance.
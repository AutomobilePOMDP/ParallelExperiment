num_of_procs = 10 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs)

using ParallelExp

@everywhere using POMCPOW
using BasicPOMCP
@everywhere push!(LOAD_PATH, "../../LB-DESPOT/")
@everywhere using LBDESPOT # LB-DESPOT pkg
@everywhere push!(LOAD_PATH, "../../UCT-DESPOT/")
@everywhere using UCTDESPOT # UCT-DESPOT pkg
@everywhere using POMDPs # Basic POMDP framework
using ParticleFilters # For simple particle filter
using POMDPPolicies
using BeliefUpdaters

@everywhere push!(LOAD_PATH, "../../Roomba")
@everywhere using Roomba # For Roomba Env

max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0

sensor = Bumper()
num_particles = 10000 # number of particles in belief

pos_noise_coeff = 0.3
ori_noise_coeff = 0.1

# POMDP problem
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
pomdp = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(aspace=action_space));

# Belief updater
resampler = BumperResampler(num_particles, pomdp, pos_noise_coeff, ori_noise_coeff)
belief_updater = BasicParticleFilter(pomdp, resampler, num_particles)

# Rush Policy
rush_policy = FunctionPolicy() do b
    if !(typeof(b) <: ParticleFilters.ParticleCollection) &&
        !(typeof(b) <: Roomba.RoombaInitialDistribution) &&
        b !== nothing &&
        typeof(b) == Bool ? b : (typeof(currentobs(b)) == Bool ? currentobs(b) : false)

        [max_speed, max_turn_rate]
    else
        [max_speed, 0.0]
    end
end

# For LB-DESPOT
bounds = IndependentBounds(DefaultPolicyLB(rush_policy), 10.0, check_terminal=true)
random_bounds = IndependentBounds(DefaultPolicyLB(RandomPolicy(pomdp)), 10.0, check_terminal=true)
lbdespot_dict = Dict(:default_action=>[rush_policy,], 
                    :bounds=>[bounds, random_bounds],
                    :K=>[100, 300, 500],
                    :beta=>[0., 0.1, 1., 10., 100.])

# For UCT-DESPOT
rollout_policy = rush_policy
random_rollout_policy = RandomPolicy(pomdp)
uctdespot_dict = Dict(:rollout_policy=>[rollout_policy, random_rollout_policy],
                        :K=>[100, 300, 500],
                        :m=>[5, 10, 20, 30],
                        :c=>[0.1, 1., 10., 100., 1000., 10000.])

# For POMCP
value_estimator = PORollout(rush_policy, PreviousObservationUpdater())
random_value_estimator = FORollout(RandomPolicy(pomdp))
pomcpow_dict = Dict(:estimate_value=>[value_estimator, random_value_estimator],
                    :tree_queries=>[100000,], 
                    :max_time=>[1.0,], 
                    :criterion=>[MaxUCB(0.1), MaxUCB(1.0), MaxUCB(10.), MaxUCB(100.), MaxUCB(1000.)])

# Solver list
solver_list = [#LB_DESPOTSolver=>lbdespot_dict, 
                #UCT_DESPOTSolver=>uctdespot_dict, 
                POMCPOWSolver=>pomcpow_dict]

number_of_episodes = 1
max_steps = 1

dfs = parallel_experiment(pomdp, number_of_episodes, max_steps, solver_list, belief_updater=belief_updater, full_factorial_design=false)
for i in 1:length(dfs)
    CSV.write("BumperRoomba$(i).csv", dfs[i])
end

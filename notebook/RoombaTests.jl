# initialize DESPOT env
import Pkg
Pkg.cd("ParallelExperiment/")
Pkg.activate(".")

# Initialize workers
using Distributed
addprocs(10, exeflags="--project") # initial workers with the project env in current work directory

using ParallelExp

# POMCP
@everywhere using POMCPOW
using BasicPOMCP

# DESPOT
@everywhere push!(LOAD_PATH, "../LB-DESPOT")
@everywhere using LBDESPOT # LBDESPOT pkg

# UCT-DESPOT
@everywhere push!(LOAD_PATH, "../UCT-DESPOT")
@everywhere using UCTDESPOT # UCT-DESPOT pkg

# POMDP related pkgs
@everywhere using POMDPs # Basic POMDP framework
using POMDPPolicies # For function policy and random policy
using ParticleFilters
using BeliefUpdaters # For roomba and BabyPOMDP belief updater

# Roomba related pkgs
# Roomba need ParticleFilters = "0.2" for compatibility
@everywhere push!(LOAD_PATH, "../Roomba")
@everywhere using Roomba # For Roomba Env

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
lbdespot_dict = Dict(:default_action=>[running_policy,], 
                    :bounds=>[random_bounds,],
                    :lambda=>[0.1, 0.0, 0.01, 1.0],
                    :T_max=>[10.0],
                    :K=>[300, 100],
                    :beta=>[0.5, 0, 1., 5.])

# For UCT-DESPOT
rollout_policy = running_policy
random_rollout_policy = RandomPolicy(pomdp)
uctdespot_dict = Dict(:rollout_policy=>[rollout_policy,],
                        :K=>[300, 100],
                        :T_max=>[10.0],
                        :m=>[30, 10],
                        :c=>[10., 1., 100.])

# For POMCP
value_estimator = FORollout(running_policy)
random_value_estimator = FORollout(RandomPolicy(pomdp))
pomcpow_dict = Dict(:estimate_value=>[value_estimator],
                    :tree_queries=>[100000,], 
                    :max_time=>[10.0,], 
                    :criterion=>[MaxUCB(10.), MaxUCB(100.)])

# Solver list
solver_list = [LB_DESPOTSolver=>lbdespot_dict, 
                UCT_DESPOTSolver=>uctdespot_dict, 
                POMCPOWSolver=>pomcpow_dict]

                
number_of_episodes = 100
max_steps = 100

dfs = parallel_experiment(pomdp,
                          number_of_episodes,
                          max_steps, solver_list,
                          belief_updater=belief_updater,
                          full_factorial_design=false)

CSV.write("notebook/DiscreteLidarRoomba_DESPOT.csv", dfs[1])
CSV.write("notebook/DiscreteLidarRoomba_UCT_DESPOT.csv", dfs[2])
CSV.write("notebook/DiscreteLidarRoomba_POMCP.csv", dfs[3])
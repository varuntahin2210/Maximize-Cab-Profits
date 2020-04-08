# Maximize-Cab-Profits

The Need for Choosing the 'Right' Requests
Most drivers get a healthy number of ride requests from customers throughout the day. But with the recent hikes in electricity prices (all cabs are electric), many drivers complain that although their revenues are gradually increasing, their profits are almost flat. Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to maximise the total profit earned by the driver that day. 


Using 2 architectures of deep Q network - Maximize the profits of cab drivers

###

Goals:

Create the environment: 
 Env.py file:This is the "environment class" - each method (function) of the class has a specific purpose. 

Building an agent that learns to pick the best request using DQN. We can choose the hyperparameters (epsilon (decay rate), learning-rate, discount factor etc.) of our choice.

Training depends purely on the epsilon-function you choose. If the ϵ
 decays fast, it won’t let your model explore much and the Q-values will converge early but to suboptimal values. If 
ϵ decays slowly, your model will converge slowly. We recommend that you try converging the Q-values in 4-6 hrs.  We’ve created a sample 
ϵ-decay function at the end of the Agent file (Jupyter notebook) that will converge your Q-values in ~5 hrs. Try building a similar one for your Q-network.

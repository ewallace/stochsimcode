function [times, updates, new_states, network_state] = ...
    run_gillespie(response_fn, W, input, alpha, beta, ...
    init_state, t_min, t_max)
% [times, updates, new_states, network_state] = ...
%     run_gillespie(response_fn, W, input, alpha, beta, ...
%     init_state, t_min, t_max)
% Simulates the 2-state Cowan model with the gillespie algorithm and
% arbitrary weight matrix, input, and transition rates, between times
% t_min and t_max.
% response_fn is handle for response function (sigmoid, hyptan or
% heaviside), see end of file.
% W is the weight matrix, n_neurons*n_neurons: W(i,j) is synaptic
% weight TO the ith neuron FROM the jth
% input is the net input, 1*n_neurons
% alpha is the rate at which active neurons decay to being
% quiescent, 1*n_neurons
% beta is the height of the response function, i.e. the rate at
% which saturated-input quiescent neurons become active, 1*n_neurons
% init_state is the initial state vector, 2*n_neurons
%
% calls: gillespie, sigmoid, hyptan, heaviside, binarysearch,
% all included in this file
%
% version 3.0, May 2010
%  Edward Wallace, ewallace a.t uchicago dot edu,
% Marc Benayoun, marcb a.t uchicago dot edu,
%%

%% Setup stuff
% Seed random number generator
rand('state', sum(100*clock));

n_neurons = length(init_state);

% Calculate expected number of events
factor=10;
expected_events=n_neurons*(t_max-t_min)*factor;

% Initialize vectors for update times, label of updated neuron,
% and new state of updated neuron
times = zeros(1, expected_events);
updates = times;
new_states = zeros(2, expected_events);

% Set event counter to 0 and simulation time to initial time.
event_no = 0;
curr_time = t_min;
dt = 0;

% initialize network state vector - in fact we'll keep one vector
% for the active neurons and another for the quiescent ones
network_state = init_state;
active = init_state(1,:)';
quiescent = init_state(2,:)';

% Calculate vector of transition rates at initial time
currents = W*active + input;
trans = beta .* (active==0) .*feval(response_fn,currents) + ...
    alpha.*(active==1);
cum_trans=cumsum(trans);

%%
% Main loop: update according to Gillespie algorithm, with rates
% specified by trans, until time t_max is exceeded.

while (curr_time < t_max)
    curr_time = curr_time + dt;
    
    % Call gillespie to pick update time, neuron updated, and new state
    [dt, n_update, new_state] = ...
        gillespie(network_state, cum_trans);
    active(n_update) = new_state(1);
    quiescent(n_update) = new_state(2);
    
    % change transition rates of neurons affected by spike
    if(new_state(1) == 1)
        currents = currents + W(:,n_update);
    elseif(new_state(1)==0)
        currents = currents - W(:,n_update);
    end;
    trans = beta .* (active==0) .*feval(response_fn,currents) + ...
        alpha.*(active==1);
    
    % %% Use code below if sparse connectivity, i.e. many W(i,j)=0
    % % find postsynaptic neurons, i.e. those affected by spike
    % postsyn = find(W(:,n_update)~=0);
    % % change transition rate of the update neuron appropriately
    % if(new_state(1) == 1)
    %     trans(n_update)=alpha(n_update);
    %     currents(postsyn) = currents(postsyn) + W(postsyn,n_update);
    % elseif(new_state(1)==0)
    %     trans(n_update)= beta(n_update) * ... 
    %         feval(response_fn,currents(n_update));
    %     currents(postsyn) = currents(postsyn) - W(postsyn,n_update);
    % end;
    %
    % % change transition rates of neurons affected by spike
    % trans(postsyn) = beta(postsyn) .* (active(postsyn)==0).* ...
    %     feval(response_fn,currents(postsyn)) + ...
    %     alpha(postsyn) .*(active(postsyn)==1);
    % %%
    
    % calculate cumulative sum of transition probabities
    cum_trans=cumsum(trans);
    
    
    event_no=event_no+1;
    times(event_no)=curr_time+dt;
    updates(event_no)=n_update;
    new_states(:, event_no)=new_state;
    network_state = [ active' ; quiescent' ];
    
    % If network can make no further transitions, end simulation.
    if(cum_trans(n_neurons)==0)
        break
    end;
end;
%% End of main simulation loop

% Throw out unused parts of result vectors. The last simulation
% event happened just after t_max, so we throw that out too.
event_no = event_no - 1;
times=times(1:event_no);
updates=updates(1:event_no);
new_states=new_states(:, 1:event_no);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions for particular transition rates follow.
%% simple implementation of heaviside function
function out = heaviside(x)
out = (x>=0);

%% sigmoid function with mean 1/2, taking values between 0 and 1
%% with 1/2 at x=0.
function y = sigmoid(x)
y = 1./(1+exp(-x));

%% hyperbolic tangent function for non-negative arguments
function out = hyptan(x)
out=tanh(x).*(x>0);


function [dt, i_update, new_state] = ...
    gillespie(network_state, cum_trans)
% [dt, i_update, new_state] = ...
%     gillespie(network_state, trans, cum_trans)
% Update rule for Gillespie algorithm in 2-state model.
% This is called by run_gillespie.m
% network_state is current network configuration
% trans is vector of transition rates
% cum_trans is cumulative sum of trans.
% all should be of the same length, the number of neurons


% Calculates total network transition rate, as sum of transition
% rates of all neurons, i.e. last element of cum_trans.
total_trans = cum_trans(end);

% timestep is exponential R.V. with parameter total_trans
dt = -log(rand)/total_trans;

% pick random variable uniform on (0, total_trans), and select
%  neuron with number i_update as least i with
% test_variable < cum_trans(i)
test_variable = total_trans*rand;
i_update=binarysearch(cum_trans, test_variable);

% Pick new state for neuron i_update
if(network_state(1,i_update)==0) new_state= [1 0];
elseif(network_state(1,i_update)== 1) new_state= [0 1];
end

function next_entry = binarysearch(table,value)
% next_entry = binarysearch(table,value)
% Returns the index of the next biggest entry to value in
% increasing vector "table", via the binary search algorithm
% Returns 1 if value is less than table(1);
% Returns 0 if value is greater than table(end)=max(table)
% Edward Wallace, ewallace at uchicago dot edu, 2009.

nrows=length(table);

if(value>table(nrows))
    next_entry=0;
    return;
end;

% Binary search squeezes the desired entry between the
% variables low and next_entry (= high), performing a
% bisection at each iteration until the value is found.
low=1;
next_entry=nrows;
while (low < next_entry)
    mid = floor((low+next_entry)/2);
    if ( table(mid) < value)
        low = mid+1;
    elseif ( table(mid) >= value)
        next_entry = mid;
    end;
end;
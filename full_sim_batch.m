function [spike_times,spike_ids] = ...
    full_sim_batch(filename, response_fn, W, input, alpha, beta, ...
    init_state, total_secs, subsave)
% [spike_times,spike_ids] = ...
%     full_sim_batch(filename, response_fn, W, input, alpha, beta, ...
%     init_state, total_secs, subsave)
% This program simulates a stochastic neural network, one second 
% at a time.
% Version 1.0, 17th June 2010
% Edward Wallace, ewallace a.t uchicago dot edu
% Marc Benayoun, marcb a.t uchicago dot edu,
%
% It calls the following as subroutines:
% run_gillespie


%% Toggle to switch saving details of each second
if(nargin<9) 
    subsave=1; 
end;

% Makes a directory for all output files to go into
if(subsave)
    mkdir(filename);
    longfilename = [filename '/' filename];
else
    longfilename = filename;
end;

%% Setup for loop
% Variables to keep track of time in seconds and simulation 
% time in milliseconds
start_state=init_state;
start_time=0;
stop_sec=1;
stop_time=1000;

% Variables to hold vectors of all spike times and neuron numbers
spike_times = [];
spike_ids = [];


%% Loop to run the simulation one second at a time for 
%  total_time seconds
while (stop_sec<=total_secs)
    % run simulation for one second, output time to screen
    [times, updates, new_states, stop_state] = ...
        run_gillespie(response_fn, W, input, alpha, beta, ...
        start_state, start_time, stop_time);

    [next_times, next_ids] = ...
        find_spike_times(times, updates, new_states);
    
    % append outputs to global output variables
    spike_times = [spike_times next_times] ;
    spike_ids = [spike_ids next_ids ];
    
    % if subsave is on, save details of each second of 
    % simulation in separate file
    if(subsave)
        % make appropriate filename
        varfilename = [filename '/second_' num2str(stop_sec)];
        save(varfilename, 'times', 'updates', 'new_states', ...
            'stop_state', 'start_state')
    end
    
    % increment time variables
    start_time = stop_time;
    disp(['finished simulating second ' num2str(stop_sec)]);
    stop_sec = stop_sec + 1;
    stop_time = stop_time + 1000;
    start_state = stop_state;
    save(longfilename);
end;


% Clear unneeded variables and save the rest
clear tvar next_times next_ids varfilename stop_time ... 
   start_sec start_state
save(longfilename);
display('everything complete and saved');



function [spike_times, spike_ids] = ...
    find_spike_times(times, ids, new_states)
% Takes vectors of all times, updates, and new states, outputs 
% reduced vector containing only spike times and info, i.e. 
% transitions from quiescent to active.
spikes = find(new_states(1,:));

spike_times = times(spikes);
spike_ids = ids(spikes);

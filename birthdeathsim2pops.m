function [times, states] = ...
 birthdeathsim2pops(transitionprobs,sizes,params,init, t_max, filename)
% [times, states] = 
% birthdeathsim2pops(transitionprobs,sizes,params,init, t_max)
% or birthdeathsim2pops(transitionprobs,sizes,params,init, ...
%        t_max, filename)
% 
% Simulates a two-variable birth-death process with transition
% probabilities given by transitionprobs with  given parameters
% transitionprobs can be a function defined below, or a function 
% handle pointing to something of your own devising.
% Uses Gillespie Algorithm.
% Edward Wallace, 12th June 2010
% ewallace a.t uchicago dot edu

% matrix of possible state transitions
% = [Efire, Ifire, Edecay, Idecay]
transitions = [ 1 0 -1 0 ; 0 1 0 -1];

%% initialize variables
% factor and est_events: adjust appropriately to the expected 
% number of transitions in model, for memory allocation.
factor = 0.1;
est_events = ceil(t_max*sum(sizes)*factor);
states=zeros(2,est_events);
times=zeros(1,est_events);
states(:,1) = init;
event_no=0;
state=states(:,1);
curr_time = 0;

% Loop for Gillespie algorithm simulation
while (curr_time < t_max)
    event_no=event_no+1;
    trans = feval(transitionprobs,sizes,params,state);
    cum_trans=cumsum(trans);
    lambda =cum_trans(end);
    % if no transitions possible, end simulation
    if(lambda<=0), 
        break, 
    end,
    
    dt = - log(rand)/lambda;
    curr_time = curr_time+dt;
    
    test_var=lambda*rand;
    % if-else loop tests random variable to choose transition
    if(test_var<=cum_trans(2))
        if(test_var<=cum_trans(1))
            i_update=1;
        else
            i_update=2;
        end;
    else
        if(test_var<=cum_trans(3)) 
            i_update = 3;
        else
            i_update = 4;
        end;
    end;
    state = state + transitions(:,i_update);
    states(:,event_no)=state;
    times(event_no)=curr_time;
end;
states=states(:,1:event_no);
times=times(1:event_no);
save(filename)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particular transition functions for Wilson-Cowan model 
% and bounded 2d random walk follow

function trans = twopopCowan(sizes,params,state)
% params = [wEE,wEI,wIE,wII,hE, hI,alphaE,alphaI]
% sizes = [NE,NI]
% state = [E,I]
trans = zeros(1,4);
sE=params(1)*state(1)-params(2)*state(2)+params(5);
sI=params(3)*state(1)-params(4)*state(2)+params(6);
trans(4)=params(8)*state(2);
trans(3)=params(7)*state(1);
trans(2)=(sizes(2)-state(2))*hyptan(sI);
trans(1)=(sizes(1)-state(1))*hyptan(sE);
end

function trans = twopopsigmoid(sizes,params,state)
% params = [wEE,wEI,wIE,wII,hE, hI,alphaE,alphaI,betaE,betaI]
% sizes = [NE,NI]
% state = [E,I]
trans = zeros(1,4);
sE=params(1)*state(1)-params(2)*state(2)+params(5);
sI=params(3)*state(1)-params(4)*state(2)+params(6);
trans(4)=params(8)*state(2);
trans(3)=params(7)*state(1);
trans(2)=(sizes(2)-state(2))*params(10)*sigmoid(sI);
trans(1)=(sizes(1)-state(1))*params(9)*sigmoid(sE);
end

function trans = twopopsym(sizes,params,state)
% params = [wE,wI,h,alpha]
trans = zeros(1,4);
f=hyptan(params(1)*state(1)-params(2)*state(2)+params(3));
trans(1)=(sizes(1)-state(1))*f;
trans(2)=(sizes(2)-state(2))*f;
trans(3)=params(4)*state(1);
trans(4)=params(4)*state(2);
end

function trans = randomwalk(sizes,params,state)
% params = [rate]
rate = params(1);
trans = zeros(1,4);
if(state(1) < sizes(1)) trans(1) = rate; end
if(state(2) < sizes(2)) trans(2) = rate; end
if(state(1) > 0) trans(3) = rate; end
if(state(2) > 0) trans(4) = rate; end
end



%% simple implementation of heaviside function
function out = heaviside(x)
out = (x>=0);
end

%% sigmoid function with mean 1/2, taking values between 0 and 1 
% with 1/2 at x= 0.
function y = sigmoid(x)
y = 1./(1+exp(-x));
end

%% hyperbolic tangent function for non-negative arguments
function out = hyptan(x)
out=tanh(x).*(x>0);
end
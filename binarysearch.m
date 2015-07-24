function [next_entry, next_value] = binarysearch(table,value)
% [next_entry, next_value] = binarysearch(table,value)
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
next_entry=length(table);
while (low < next_entry)
    mid = floor((low+next_entry)/2);
    if ( table(mid) < value)
        low = mid+1;
    elseif ( table(mid) >= value)
        next_entry = mid;
    end;
end;

next_value = table(next_entry);
#!/usr/bin/env bash

# This is a hacky way to restart the script to randomize the rocks.
# There are some bugs with the mjviewer and it was a hack to get it
# to work, so I didn't want to mess with and this is the solution

# launcher.sh: This script starts the program, checks every second if it is 
# finished. If it is, this will restart it.


# Respond to CTRL+C correctly
control_c() {
    kill $last_pid
    exit
}
trap control_c SIGINT


signal=KILL

shift

if [[ "$@" != *"super_batch"* ]]; then
	args="$@ --super_batch 1000"
else
	args="$@"
fi

while true; do
	# pass through cmd line args and set super batch default
	./rock_detector.py $args

	# Save PID of command just launched:
	last_pid=$!

	wait $last_pid
	exit_status=$?
	echo "EXIT STATE: $exit_status"
	
	if [[ $exit_status != 0 ]]; then
		echo "EXIT"
		exit;
	fi
done

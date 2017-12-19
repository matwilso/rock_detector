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
sleep_a_while () {
    sleep 1s
}
shift

while true; do
	./rock_detector.py "$@"  # pass through cmd line args
	# Save PID of command just launched:
	last_pid=$!
	while true; do
	    # See if the command is still running
	    if ps -p $last_pid -o comm= | grep -qs '^rock_detector.py$'; then
	        sleep_a_while
		else
			# Go back to the beginning and launch the command again
			break
	    fi
	done
done

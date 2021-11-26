#!/bin/bash
set -e

# setup ros environment
#source "/root/catkin_ws/devel/setup.bash"

#pip3 install -e /home/improbable/isaac_loco

export PYTHONPATH=/root/bandu_v1_full_clean:${PYTHONPATH}

eval "bash"

exec "$@"
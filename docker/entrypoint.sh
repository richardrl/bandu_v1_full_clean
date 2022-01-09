#!/bin/bash
set -e

# setup ros environment
#source "/root/catkin_ws/devel/setup.bash"

#pip3 install -e /home/improbable/isaac_loco

export PYTHONPATH=/home/docker/bandu_v1_full_clean:${PYTHONPATH}

# gosu magic start
# If "-e uid={custom/local user id}" flag is not set for "docker run" command, use 9999 as default
CURRENT_UID=${uid:-9999}

# Notify user about the UID selected
echo "Current UID : $CURRENT_UID"

usermod -u $CURRENT_UID docker

# Create user called "docker" with selected UID
#useradd --shell /bin/bash -u $CURRENT_UID -o -c "" -m docker

# Execute process
exec gosu docker "$@"
# gosu magic


eval "bash"

exec "$@"
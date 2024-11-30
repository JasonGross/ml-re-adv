#!/bin/bash

# Set up iptables rules to allow specific access
iptables -A OUTPUT -p tcp -d 34.117.59.81 -j ACCEPT  # huggingface.co IP
iptables -A OUTPUT -p tcp -d 54.88.227.179 -j ACCEPT  # pypi.org IP
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT  # Allow DNS resolution
iptables -P OUTPUT DROP  # Block all other outgoing traffic

# Drop NET_ADMIN capability to prevent further changes
# Use prctl to drop capabilities for the current process and its children
prctl --capbset=-CAP_NET_ADMIN

# Continue with the container's main command
exec "$@"

#!/usr/bin/env bash
set -ex
# add missing google pub key
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C0BA5CE6DC6315A3
# Add Docker's official GPG key:
sudo apt-get update -y --allow-releaseinfo-change
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin jq

sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Function to safely update or create the daemon.json file
# update_daemon_json() {
#     local tmp_file=$(mktemp)
#     if [ -f /etc/docker/daemon.json ]; then
#         sudo cat /etc/docker/daemon.json | \
#         jq '
#             if has("hosts") then
#                 .hosts += ["tcp://127.0.0.1:2375"] | .hosts |= unique
#             else
#                 . + {"hosts": ["unix:///var/run/docker.sock", "tcp://127.0.0.1:2375"]}
#             end
#         ' > "$tmp_file"
#     else
#         echo '{"hosts": ["unix:///var/run/docker.sock", "tcp://127.0.0.1:2375"]}' > "$tmp_file"
#     fi
#     sudo mv "$tmp_file" /etc/docker/daemon.json
#     sudo chown root:root /etc/docker/daemon.json
#     sudo chmod 644 /etc/docker/daemon.json
# }

# Update or create daemon.json
# echo "Updating /etc/docker/daemon.json..."
# update_daemon_json

# enable docker remote API
if ! grep -q -- '-H tcp://127.0.0.1:2375' /lib/systemd/system/docker.service; then
  sudo sed -i 's|ExecStart=/usr/bin/dockerd -H fd://|ExecStart=/usr/bin/dockerd -H fd:// -H tcp://127.0.0.1:2375|' /lib/systemd/system/docker.service
fi

# # Restart Docker service
# echo "Restarting Docker service..."
# sudo systemctl restart docker

# # Verify Docker is running
# if sudo systemctl is-active --quiet docker; then
#     echo "Docker service is running."
# else
#     echo "Failed to start Docker service. Please check the logs with 'sudo journalctl -u docker'."
#     exit 1
# fi

# # Print instructions for the user
# echo "
# Docker daemon has been configured to listen on 127.0.0.1:2375.

# To access Docker remotely:

# 1. On your local machine, run:
#    gcloud compute ssh [VM_NAME] --zone=[ZONE] -- -L 2375:localhost:2375 -N -f

# 2. Replace [VM_NAME] and [ZONE] with your VM's name and zone.

# 3. You can now use Docker commands on your local machine like this:
#    docker -H localhost:2375 ps

# 4. When finished, find and kill the SSH process:
#    ps aux | grep \"ssh -L 2375:localhost:2375\"
#    kill [PID]

# Remember, always use this method over a secure network connection."

# echo "Setup complete!"


sudo systemctl daemon-reload
sudo systemctl restart docker

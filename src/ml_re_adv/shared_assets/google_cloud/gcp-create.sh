#!/usr/bin/env bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

while ! test -f .env; do
  cd .. || { >&2 echo "No .env file found"; exit 1; }
done
source .env
zone_args=()
test -n "${GCLOUD_ZONE}" && zone_args=(--zone="${GCLOUD_ZONE}")
if gcloud compute tpus tpu-vm list "${zone_args[@]}" | cut -f1 -d' ' | grep -q "^${GCLOUD_INSTANCE}$"; then
  echo "Instance ${GCLOUD_INSTANCE} already exists"
else
  if [ -z "${GCLOUD_ZONE}" ]; then
    GCLOUD_ZONE="$(gcloud config get-value compute/zone)"
  fi
  case "${GCLOUD_ZONE}" in
    "us-central2-b")
      if [ -z "${GCLOUD_NCHIPS}" ]; then
        GCLOUD_NCHIPS=32
      fi
      accelerator_type="v4-${GCLOUD_NCHIPS}"
      ;;
    "us-central1-f")
      accelerator_type="v2-8"
      ;;
    "europe-west4-a")
      accelerator_type="v3-8"
      ;;
    *)
      >&2 echo "Unsupported zone: ${GCLOUD_ZONE}"
      exit 1
      ;;
  esac

  gcloud compute tpus tpu-vm create "${GCLOUD_INSTANCE}" \
    "${zone_args[@]}" \
    --accelerator-type="${accelerator_type}" \
    --version=v2-alpha \
    --preemptible
fi
# copy over ${DIR}/gcp-init-remote.sh to the remote instance
gcloud compute tpus tpu-vm scp "${DIR}/gcp-init-remote.sh" "${GCLOUD_INSTANCE}:/tmp/gcp-init-remote.sh" "${zone_args[@]}"
gcloud compute tpus tpu-vm ssh "${GCLOUD_INSTANCE}" "${zone_args[@]}" --command "bash /tmp/gcp-init-remote.sh"

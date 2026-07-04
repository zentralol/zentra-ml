pipeline {
  agent {
    kubernetes {
      cloud 'kubernetes'
      defaultContainer 'docker'
      yaml '''
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins/label: zentra-ml-deploy
spec:
  containers:
    - name: docker
      image: docker:27-cli
      command:
        - sleep
      args:
        - "9999999"
      tty: true
      workingDir: /home/jenkins/agent
      volumeMounts:
        - mountPath: /var/run/docker.sock
          name: docker-sock
  volumes:
    - name: docker-sock
      hostPath:
        path: /var/run/docker.sock
'''
    }
  }

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  parameters {
    string(name: 'IMAGE_NAME', defaultValue: 'ghcr.io/zentralol/zentra-ml', description: 'GHCR image repository to deploy', trim: true)
    string(name: 'IMAGE_TAG', defaultValue: 'latest', description: 'GHCR image tag to deploy', trim: true)
    string(name: 'GIT_COMMIT_SHA', defaultValue: '', description: 'Commit SHA that produced the image', trim: true)
    string(name: 'GIT_BRANCH', defaultValue: 'main', description: 'Git branch that produced the image', trim: true)
  }

  environment {
    CONTAINER_NAME = 'zentra-ml'
    NETWORK_NAME = 'zentra'
    HOST_PORT = '8003'
    CONTAINER_PORT = '8000'
    HEALTH_PATH = '/'
    HEALTH_RETRIES = '30'
    HEALTH_INTERVAL_SEC = '2'
  }

  stages {
    stage('Deploy') {
      steps {
        sh(label: 'Install deploy tools', script: 'apk add --no-cache bash curl openssh-client')

        withCredentials([
          string(credentialsId: 'hel-host', variable: 'DEPLOY_HOST'),
          sshUserPrivateKey(credentialsId: 'server-ssh-key', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')
        ]) {
          sh(label: 'Deploy ML container', script: '''#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
[ -n "${DEPLOY_HOST}" ] || { echo "DEPLOY_HOST is empty"; exit 1; }
[ -n "${SSH_USER}" ] || { echo "SSH_USER is empty"; exit 1; }
[ -n "${IMAGE_NAME}" ] || { echo "IMAGE_NAME is empty"; exit 1; }
[ -n "${IMAGE_TAG}" ] || { echo "IMAGE_TAG is empty"; exit 1; }

chmod 600 "${SSH_KEY}"

remote_env=(
  "IMAGE=$(printf '%q' "${IMAGE}")"
  "CONTAINER_NAME=$(printf '%q' "${CONTAINER_NAME}")"
  "NETWORK_NAME=$(printf '%q' "${NETWORK_NAME}")"
  "HOST_PORT=$(printf '%q' "${HOST_PORT}")"
  "CONTAINER_PORT=$(printf '%q' "${CONTAINER_PORT}")"
  "HEALTH_PATH=$(printf '%q' "${HEALTH_PATH}")"
  "HEALTH_RETRIES=$(printf '%q' "${HEALTH_RETRIES}")"
  "HEALTH_INTERVAL_SEC=$(printf '%q' "${HEALTH_INTERVAL_SEC}")"
)

ssh -i "${SSH_KEY}" \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  "${SSH_USER}@${DEPLOY_HOST}" \
  "${remote_env[*]} bash -s" <<'REMOTE'
set -euo pipefail

log() { printf '[deploy] %s\\n' "$*"; }
die() { printf '[deploy] ERROR: %s\\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found on target server"
command -v curl >/dev/null 2>&1 || die "curl not found on target server"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  log "Removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

if ! docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1; then
  log "Creating Docker network: ${NETWORK_NAME}"
  docker network create "${NETWORK_NAME}" >/dev/null
fi

log "Pulling image: ${IMAGE}"
docker pull "${IMAGE}"

log "Starting container: ${CONTAINER_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network "${NETWORK_NAME}" \
  --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "${IMAGE}"

HEALTH_URL="http://127.0.0.1:${HOST_PORT}${HEALTH_PATH}"
log "Waiting for health check: ${HEALTH_URL}"

for attempt in $(seq 1 "${HEALTH_RETRIES}"); do
  if response="$(curl -fsS "${HEALTH_URL}" 2>/dev/null)"; then
    if echo "${response}" | grep -q 'Zentra Crowd Prediction API is running'; then
      log "Health check passed"
      docker ps --filter "name=${CONTAINER_NAME}"
      exit 0
    fi
    log "Health endpoint reachable but response unexpected (${attempt}/${HEALTH_RETRIES})"
  else
    log "Health check pending (${attempt}/${HEALTH_RETRIES})"
  fi
  sleep "${HEALTH_INTERVAL_SEC}"
done

log "Deployment failed health check. Recent container logs:"
docker logs --tail 50 "${CONTAINER_NAME}" || true
die "health check did not pass"
REMOTE
''')
        }
      }
    }
  }
}

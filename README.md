# Setup docker

1. Log in to lambda machine ```ssh avante-lambda-10```
2. Build docker ```cd /local/mnt3/workspace3/luobwang/planning-intern/docker && ./build.sh```
3. Start docker container  ```/local/mnt3/workspace3/luobwang/planning-intern/docker/run-docker -w /local/mnt3/workspace3/luobwang```
4. Attach to the container ```docker attach unitraj ```
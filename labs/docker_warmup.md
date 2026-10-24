# Task 6.1 - Docker warm-up (ready-made image, before touching the project)

Run each command in PowerShell and screenshot the output. Docker Desktop must be running.

```powershell
docker version                                           # client + engine are both up
docker pull nginx:1.27-alpine                            # download an image
docker images                                            # list images

# run a container in the background, map host port 8080 -> container port 80,
# and mount a host folder as nginx's web root (a bind-mount volume)
mkdir $HOME\warmup-site
"<h1>Hello from my volume - $env:USERNAME $(Get-Date)</h1>" | Set-Content $HOME\warmup-site\index.html
docker run -d --name warmup -p 8080:80 -v "$HOME\warmup-site:/usr/share/nginx/html:ro" nginx:1.27-alpine
# open http://localhost:8080 in the browser -> shows the file from the volume

docker ps                                                # running containers
docker logs warmup                                       # its stdout/stderr (the access log)
docker exec -it warmup sh                                # shell INSIDE the container
#   inside: whoami; ls /usr/share/nginx/html; cat /etc/os-release; exit

# named volume (managed by Docker rather than a host folder)
docker volume create warmup-data
docker run --rm -v warmup-data:/data alpine sh -c "echo persisted > /data/note.txt"
docker run --rm -v warmup-data:/data alpine cat /data/note.txt   # data survived the first container

docker stop warmup
docker ps -a                                             # stopped containers still exist
docker rm warmup                                         # remove the container
docker rmi nginx:1.27-alpine alpine                      # remove the images
docker volume rm warmup-data
docker ps -a; docker images                              # confirm they are gone
```

Definitions to write **in your own words** in the report: image, container, layer, volume, port mapping.

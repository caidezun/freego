# freego

A minimal Go HTTP service used as the starting point for the `freego` project.

## Requirements

- Go 1.22+

## Getting started

```bash
# Download dependencies and build
go mod download
go build ./...

# Run the server (defaults to port 8080, override with PORT)
make run
# or
go run .

# Run the tests
make test
```

Once running, the service is available at http://localhost:8080.

## Endpoints

| Method | Path                | Description                                  |
| ------ | ------------------- | -------------------------------------------- |
| GET    | `/`                 | HTML landing page                            |
| GET    | `/healthz`          | JSON health check (`{"status":"ok",...}`)    |
| GET    | `/api/hello?name=X` | JSON greeting (`{"message":"hello, X!"}`)    |

Example:

```bash
curl localhost:8080/healthz
curl "localhost:8080/api/hello?name=freego"
```

## Development

Common tasks are available via the `Makefile`:

| Command      | Description                       |
| ------------ | --------------------------------- |
| `make build` | Build the binary into `bin/`      |
| `make run`   | Run the server                    |
| `make test`  | Run the test suite                |
| `make vet`   | Run `go vet`                      |
| `make fmt`   | Format the source                 |
| `make tidy`  | Sync module dependencies          |

## Cloud Agent environment

This repository includes a `.cursor/environment.json` so Cursor Cloud Agents can
build and run the service automatically. The `install` step downloads
dependencies and compiles the project, and a `server` terminal runs `go run .`.

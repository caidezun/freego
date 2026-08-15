.PHONY: build run test vet fmt tidy clean

# Build the freego binary into ./bin/freego
build:
	go build -o bin/freego .

# Run the server (honours the PORT env var, defaults to 8080)
run:
	go run .

# Run the test suite
test:
	go test ./...

# Static analysis
vet:
	go vet ./...

# Format all Go source
fmt:
	go fmt ./...

# Sync module dependencies
tidy:
	go mod tidy

clean:
	rm -rf bin

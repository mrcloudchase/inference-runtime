.PHONY: all rust go clean test

all: rust go

rust:
	cargo build --release

go: rust
	cd go-wrapper && CGO_ENABLED=1 GOARCH=arm64 \
		CGO_LDFLAGS="-L../target/release -lir_ffi" \
		CGO_CFLAGS="-I./bindings" \
		go build -o ../target/release/ir .

clean:
	cargo clean
	rm -f target/release/ir

test:
	cargo test --workspace
	cd go-wrapper && go test ./...

fmt:
	cargo fmt --all
	cd go-wrapper && gofmt -w .

check:
	cargo clippy --workspace -- -D warnings

# Default task runs when you just type `just`
default:
  just --list

alias cov := coverage

coverage: _check-cargo-llvm-cov _check-cargo-nextest
  cargo llvm-cov nextest --lcov --output-path target/lcov.info
  @echo ""
  @echo "(If you're using Visual Studio Code's \"Coverage Gutters\" extension, you can run it now.)"

_check-cargo-nextest:
  @if ! command -v cargo nextest -V > /dev/null; then \
    echo "❌ cargo-nextest not found."; \
    echo "  Please install: cargo install cargo-nextest --locked"; \
    exit 1; \
  fi

_check-cargo-llvm-cov:
  @if ! command -v cargo llvm-cov -V > /dev/null; then \
    echo "❌ cargo-llvm-cov not found."; \
    echo "  Please install: cargo +stable install cargo-llvm-cov --locked"; \
    exit 1; \
  fi

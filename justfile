# Default task runs when you just type `just`
default:
  just --list

################################################################################
# test
################################################################################

alias cov-open := coverage-open
alias covo := coverage-open
alias cov := coverage

[group('test')]
coverage-open:
  cargo llvm-cov nextest --open

# Use REMAINDER to specify a test or tests, e.g. tree::node::leaf
[group('test')]
coverage +REMAINDER='': _check-cargo-llvm-cov _check-cargo-nextest
  cargo llvm-cov nextest --lcov --output-path target/lcov.info {{REMAINDER}}
  @echo ""
  @echo "(If you're using Visual Studio Code's \"Coverage Gutters\" extension, you can run it now.)"

[group('test')]
_check-cargo-nextest:
  @if ! command -v cargo nextest -V > /dev/null; then \
    echo "❌ cargo-nextest not found."; \
    echo "  Please install: cargo install --locked cargo-nextest"; \
    exit 1; \
  fi

[group('test')]
_check-cargo-llvm-cov:
  @if ! command -v cargo llvm-cov -V > /dev/null; then \
    echo "❌ cargo-llvm-cov not found."; \
    echo "  Please install: cargo +stable install --locked cargo-llvm-cov"; \
    exit 1; \
  fi

################################################################################
# profile
################################################################################

alias prof := cpu-profile

[group('profile')]
cpu-profile: _check-samply
  cargo bench --profile=profiling --no-run
  samply record $(ls target/profiling/deps/txn-* | grep -E '^target/profiling/deps/txn-[^.]+$')

[group('profile')]
_check-samply:
  @if ! command -v samply > /dev/null; then \
    echo "❌ samply not found."; \
    echo "  Please install: cargo install --locked samply"; \
    exit 1; \
  fi

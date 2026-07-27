# Vendored Boost

This directory contains a **pruned subset** of the Boost C++ Libraries (version
1.89.0), vendored directly in the repository so that the SweRV-ISS-1 oracle
build is fully self-contained and does not require a system-installed Boost.

## What is included

Only the parts of Boost actually used by the SweRV-ISS-1 simulator are
vendored:

- `boost/` — 640 header files (header-only libraries) pulled from the Boost
  1.89.0 release. These cover the following Boost components used by SweRV:
  - `boost/program_options.hpp` (plus its transitive dependencies)
  - `boost/format.hpp`
  - `boost/lexical_cast.hpp`
  - `boost/algorithm/string.hpp`
  - `boost/multiprecision/cpp_int.hpp`

- `boost_program_options/src/` — the source files for `Boost.program_options`,
  the only non-header-only Boost component used. These are compiled into
  `libboost_program_options.a` on demand (see below).

- `GNUmakefile.vendor` — a makefile that compiles the program_options sources
  into a static library at `vendor/stage/lib/libboost_program_options.a`.

## Building the program_options library

The static library is **not** committed; it is built on demand by the test
harness (or manually):

```sh
make -f vendor/GNUmakefile.vendor
```

This produces `vendor/stage/lib/libboost_program_options.a`.

## How the SweRV oracle uses it

The SweRV-ISS-1 `GNUmakefile.wdc` is invoked with `BOOST_DIR` pointing at the
absolute path of this `vendor/` directory. The makefile finds:

- Headers via `-isystem <vendor>` (so `#include <boost/...>` resolves correctly).
- The static library via `$(BOOST_DIR)/stage/lib/libboost_program_options.a`.

## License

Boost is distributed under the Boost Software License, Version 1.0.
See `LICENSE_1_0.txt` for the full text.

Source: https://www.boost.org/users/history/version_1_89_0.html

# engine_ext NOTICE

`engine_ext/` holds StrataShift's own files that are copied into a copy of the pinned PolyStrata tree and built
there as `strata-shift-poly` (see [README.md](README.md)). The PolyStrata sources themselves are not in this
repository; they stay in `.local/polystrata/` (docs/GITIGNORE_POLICY.md).

StrataShift's files here:

- `src/poly/mixtral.hpp`, `src/poly/mixtral.cpp` — the `Mixtral` architecture class. It derives from PolyStrata's
  `Engine`, calls its kernels, and follows the structure of PolyStrata's `src/poly/qwen35.cpp`.
- `cmake/strata_shift_poly.cmake` — the `strata-shift-poly` target.
- `overlay.json` — the lines added to the copy of PolyStrata's `src/poly/engine.cpp` and `CMakeLists.txt`.
- `pin_files.json`, `README.md`, this file.

The executable built from them contains PolyStrata. PolyStrata is distributed under the MIT License, reproduced
below from the pinned tree (`LICENSE`, pin `c566ff4`):

```text
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

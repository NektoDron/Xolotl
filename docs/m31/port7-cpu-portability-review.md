# CPU pair header / batched dense: независимое review

27октября2026. Production-код при этом review не менялся.

## CPU portability

`cpu_batched_dot.hpp` помещает immintrin, CPU FP16 LUT extern, ISA attributes
и SIMD функции под условие x86 + `__GNUC__`. Native GCC и Clang попадают в
эту ветку; `target("avx2,fma")` разрешает сборку без глобального -march и
runtime CPUID проверяется перед вызовом. MSVC/Clang-cl (нет __GNUC__) и
non-x86 остаются на available=false / dot_pair=false, без intrinsics.

Фактическая проверка `port7-cpu-portability-v1/receipt.json`:6/6 PASS,3,27s.
GCC с принудительно false CPUID собрался и исполнился: dot_pair вернул false,
выходные ссылки не изменились. Native Clang собрался и повторил1966080
real-GGUF dot comparisons побитно, maxabs0. Clang target MSVCx64/AArch64
прошёл fallback-branch syntax с минимальными std/ggml stubs. Это последняя
проверка является проверкой guards, **не** полным SDK/backend build.

Ограничение: FP16 LUT ссылается на внутренний `ggml_table_f32_f16`.
Canonical Xolotl принудительно использует static GGML, там символ проверен.
Совместимость с shared/DLL export policy и иной версией ggml не заявляется.

## Batchdense graph

QKV offsets Q0/KQrows/V(Qrows+Krows), indexQK offsets0/IQrows, gate/up
split0/ffwidth соответствуют старым n1 bundles. dspark_column использует
column*matrix.nb1 + row_offset*sizeof(F32), затем cont; выход mul_mat F32.
RMS, Q/K norm, RoPE, sigmoid/top-k/route weights, MSA score и FA остаются
индивидуальными n1 операциями. ProjectedAttention переносит уже вычисленные
q/k/v/iq/ik и defer_output переносит только attn_output projection.
Индексер не получает marked MMV; mark вызывает только dspark_projection
для уже загруженных dense weight tensors. Per-query nk/positions/masks
входят в persistent geometry/value update; zero-filled KV предотвращает
NaN из незаполненных future rows.

Найден gap API v2: actual 2D MSA score CAST(indexKV)[128,nk] × IQ[128,4]
подходит по форме/type. До allocation у CAST нет buffer, а setter v2
проверял usage только при buffer!=null; он мог принять такой compute-node
и позже упасть на runtime WEIGHTS assertion. Нынешний DSpark callsite
такой setter не вызывает, поэтому frozen v2 pilots этим не испорчены.
Dense owner подтвердил исправление API v3: source opNONE и уже bound
WEIGHTS buffer, plus actual2DCAST/nullbuffer rejection tests. Live review
не вносил patch самостоятельно.

Marked n1 vs K2/3/4 MMV equivalence требует probes и target parity на
реальных projection types; название marker не заменяет численную проверку.
На16K/32K и128/256 границах нужен model greedy parity, особенно после
реального MSA block selection. CPU component proof его не подменяет.

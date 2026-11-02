#!/usr/bin/env bash
# Compile-only, own-fork hosted runner. No AMD device, model downloads or inference.
set -euo pipefail
export LC_ALL=C
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONDONTWRITEBYTECODE=1
[[ ${GITHUB_ACTIONS:-} == true && ${GITHUB_REPOSITORY:-} == NektoDron/Xolotl && ${RUNNER_OS:-} == Linux ]]
[[ ${PR99_SOURCE_SHA:-} == 5ffbd05448cdff041dbb1d43ff2fdbd2b76e3c34 ]]
[[ ${PR99_LLAMA_SHA:-} == 6050b467c07a95c7b87debd4ddac2f8ba8d9b807 ]]
pr99_source=$(realpath "$1")
pr99_evidence=$(realpath -m "$2")
pr99_harness=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$pr99_evidence"
exec > >(tee "$pr99_evidence/qualification.log") 2>&1
pr99_build="$GITHUB_WORKSPACE/pr99-build-hip"
pr99_finish() {
  pr99_status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$pr99_status" > "$pr99_evidence/exit-code.txt"
  df -B1 "$pr99_evidence" > "$pr99_evidence/disk-final.txt"
  free -b > "$pr99_evidence/memory-final.txt"
  for pr99_diag in CMakeCache.txt compile_commands.json CMakeFiles/CMakeConfigureLog.yaml Testing/Temporary/LastTest.log; do
    [[ ! -f "$pr99_build/$pr99_diag" ]] || cp "$pr99_build/$pr99_diag" "$pr99_evidence/$(basename "$pr99_diag")"
  done
  exit "$pr99_status"
}
trap pr99_finish EXIT
[[ $(git -C "$pr99_source" rev-parse HEAD) == "$PR99_SOURCE_SHA" ]]
[[ -z $(git -C "$pr99_source" status --porcelain --untracked-files=no) ]]
[[ ! -e /dev/kfd ]] # This job deliberately has no AMD device access.
printf '%s\n' "$PR99_SOURCE_SHA" > "$pr99_evidence/source-head.txt"
printf '%s\n' "$GITHUB_SHA" > "$pr99_evidence/harness-head.txt"
printf '%s\n' 'Compile/link only. No model or device execution. AMD GPU hardware qualification NOT RUN.' > "$pr99_evidence/SCOPE.txt"
uname -a > "$pr99_evidence/host.txt"
df -B1 "$pr99_evidence" > "$pr99_evidence/disk-before.txt"
free -b > "$pr99_evidence/memory-before.txt"
python3 "$pr99_harness/pr99_rocm_plan.py" --out "$pr99_evidence/rocm-metadata.json" --packages "$pr99_evidence/sdk-packages.txt"

# Official AMD 7.0.2/noble repository, authenticated by its published signing key.
# This is an ephemeral hosted runner; no amdgpu/dkms or model packages are requested.
curl -fL --retry 3 https://repo.radeon.com/rocm/rocm.gpg.key -o "$pr99_evidence/amd-repository-key.asc"
printf '%s  %s\n' '2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b' "$pr99_evidence/amd-repository-key.asc" | sha256sum --check
# Primary fingerprint observed from AMD's official URL; fail closed on key rotation.
gpg --show-keys --with-colons "$pr99_evidence/amd-repository-key.asc" > "$pr99_evidence/key-identity.txt"
[[ $(awk -F: '$1=="fpr"{print $10;exit}' "$pr99_evidence/key-identity.txt") == CA8BB4727A47B4D09B4EE8969386B48A1A693C5C ]]
gpg --dearmor --output "$pr99_evidence/amd-repository-key.gpg" "$pr99_evidence/amd-repository-key.asc"
sudo install -m 0644 "$pr99_evidence/amd-repository-key.gpg" /etc/apt/keyrings/pr99-rocm.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/pr99-rocm.gpg] https://repo.radeon.com/rocm/apt/7.0.2 noble main' | sudo tee /etc/apt/sources.list.d/pr99-rocm.list
sudo apt-get update | tee "$pr99_evidence/apt-update.log"
mapfile -t pr99_sdk < "$pr99_evidence/sdk-packages.txt"
pr99_host=(build-essential cmake ninja-build libnuma-dev python3 git)
sudo apt-get -s --no-install-recommends install "${pr99_sdk[@]}" "${pr99_host[@]}" | tee "$pr99_evidence/apt-simulate.log"
sudo apt-get --print-uris --download-only --yes --no-install-recommends install "${pr99_sdk[@]}" "${pr99_host[@]}" | tee "$pr99_evidence/apt-uri-plan.log"
python3 - "$pr99_evidence" <<'PY'
import json, pathlib, re, shlex, shutil, subprocess, sys
out = pathlib.Path(sys.argv[1]); plan = json.loads((out/'rocm-metadata.json').read_text())
archives = 0
for line in (out/'apt-uri-plan.log').read_text().splitlines():
    if line.startswith("'"):
        fields = shlex.split(line)
        if len(fields) >= 3 and fields[2].isdigit():
            archives += int(fields[2])
assert archives > 0, 'No bounded apt archive plan; do not start SDK download'
pins = {record['Package']: record for record in plan['packages']}
verified = {}
def package_fields(name, version):
    text = subprocess.check_output(['apt-cache', 'show', name+'='+version], text=True)
    records = [{line.split(':',1)[0].lower(): line.split(':',1)[1].strip()
                for line in block.splitlines() if line and not line.startswith(' ') and ':' in line}
               for block in text.split('\n\n') if block.strip()]
    published = [record for record in records if record.get('size') and record.get('sha256')]
    assert published, ('No authenticated apt archive metadata', name, version)
    pin = pins.get(name.split(':')[0])
    if pin:
        # apt acquired its own signed index after the initial AMD metadata fetch.
        # Verify this actual resolver view too, before any SDK package download.
        assert version == pin['Version']
        assert all(record.get('version') == pin['Version'] and record['size'] == pin['Size']
                   and record['sha256'] == pin['SHA256'] for record in published), ('APT SDK pin mismatch', name)
        verified[name.split(':')[0]] = published[0]
    return published[0]
for name, pin in pins.items():
    package_fields(name, pin['Version'])
(out/'apt-pinned-metadata.json').write_text(json.dumps(verified, indent=2)+'\n')
selected = []
for line in (out/'apt-simulate.log').read_text().splitlines():
    match = re.match(r'^Inst (\S+)(?: \[[^]]+\])? \((\S+)', line)
    if not match:
        continue
    name, version = match.groups()
    fields = package_fields(name, version)
    installed = fields.get('installed-size')
    compressed = int(fields['size'])
    selected.append(dict(package=name, version=version, archive_bytes=compressed,
                         installed_bytes=int(installed)*1024 if installed else None,
                         unknown_installed_allowance_bytes=0 if installed else 4*compressed))
assert selected, 'No resolved apt installation plan'
(out/'apt-resolved-resources.json').write_text(json.dumps(selected, indent=2)+'\n')
known = sum(package['installed_bytes'] or 0 for package in selected)
unknown = sum(package['unknown_installed_allowance_bytes'] for package in selected)
mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}
root_free = shutil.disk_usage('/').free
workspace_free = shutil.disk_usage(out).free
free = min(root_free, workspace_free)
# The reserve is a planning guard, not measured compiler demand. No broad RAM
# requirement: one compile worker; record actual available memory and outcomes.
# The local CUDA build directory was observed at 6.1 GiB. Reserve 8 GiB
# for a build plus headroom; this is deliberately not called measured HIP use.
reserve = 8 << 30
needed = archives + known + unknown + reserve
receipt = dict(free_bytes=free, root_free_bytes=root_free, workspace_free_bytes=workspace_free,
               apt_archive_bytes=archives, known_selected_installed_bytes=known,
               unknown_installed_allowance_bytes=unknown,
               build_reserve_bytes=reserve, build_reserve_basis='8GiB allowance from observed 6.1GiB local CUDA tree; HIP footprint unmeasured', planned_peak_bytes=needed, mem_available_bytes=mem['MemAvailable'], jobs=1,
               scope='Resolved apt plan including Ubuntu deps; installed-size metadata plus 4x archive allowance for missing fields. Not measured peak RAM/build size.')
(out/'resource-preflight.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert free >= needed, 'Insufficient measured disk for conservative SDK/archive plan; no SDK download/install started'
assert mem['MemAvailable'] >= 1 << 30, 'Less than 1GiB observed memory headroom; no SDK download/install started'
PY
sudo apt-get --yes --no-install-recommends -o APT::Keep-Downloaded-Packages=false -o Binary::apt-get::APT::Keep-Downloaded-Packages=false install "${pr99_sdk[@]}" "${pr99_host[@]}" | tee "$pr99_evidence/apt-install.log"
dpkg-query -W -f='${Package}\t${Version}\t${Installed-Size}\n' "${pr99_sdk[@]%%=*}" > "$pr99_evidence/installed-sdk.tsv"
df -B1 "$pr99_evidence" > "$pr99_evidence/disk-after-sdk.txt"
free -b > "$pr99_evidence/memory-after-sdk.txt"
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2
[[ -x "$ROCM_PATH/llvm/bin/clang++" ]]
"$ROCM_PATH/llvm/bin/clang++" --version > "$pr99_evidence/compiler.txt"
cmake --version > "$pr99_evidence/cmake-version.txt"

# Exact existing CMake pin. Fetch source only; no model and no helper GPU program.
pr99_llama="$GITHUB_WORKSPACE/pr99-llama"
git init -q "$pr99_llama"
git -C "$pr99_llama" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$pr99_llama" fetch --depth 1 origin "$PR99_LLAMA_SHA"
git -C "$pr99_llama" checkout --detach FETCH_HEAD
[[ $(git -C "$pr99_llama" rev-parse HEAD) == "$PR99_LLAMA_SHA" ]]
printf '%s\n' "$PR99_LLAMA_SHA" > "$pr99_evidence/ggml-head.txt"
# This patch is required by this source's CMake (not a workaround or guard change).
sha256sum "$pr99_source/tools/xolotl/ggml/xolotl-ggml.patch" > "$pr99_evidence/ggml-patch-sha256.txt"
git -C "$pr99_llama" apply --check "$pr99_source/tools/xolotl/ggml/xolotl-ggml.patch"
git -C "$pr99_llama" apply "$pr99_source/tools/xolotl/ggml/xolotl-ggml.patch"
git -C "$pr99_llama" diff --binary > "$pr99_evidence/ggml-applied.patch"
pr99_build="$GITHUB_WORKSPACE/pr99-build-hip"
# STRATA_DENSE stays at its existing ON default: CMakeLists.txt:957 requires
# STRATA_ENABLE_CUDA before adding strata-dense/dense_model/dense_kernels. The
# other DENSE use at :2306 is inside the CUDA MMQ branch; no HIP target is hidden.
cmake -S "$pr99_source" -B "$pr99_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DSTRATA_GGML_DIR="$pr99_llama" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_M31=OFF | tee "$pr99_evidence/configure.log"
cp "$pr99_build/CMakeCache.txt" "$pr99_evidence/CMakeCache.txt"
cmake --build "$pr99_build" --target strata strata-q36 ds41_generate persistent_prefix_test \
  q36_persistent_payload_test m31_prefix_codec_check test_ds41_persistent_identity \
  test_ds41_persistent_wire q36_adapter_protocol_mock -j1 | tee "$pr99_evidence/build.log"
ctest --test-dir "$pr99_build" -N -R '^(persistent_prefix_test|q36_persistent_payload_test|m31_prefix_codec_check|test_ds41_persistent_identity|test_ds41_persistent_wire|test_q36_adapter_protocol)$' | tee "$pr99_evidence/host-tests-inventory.txt"
grep -q '^Total Tests: 6$' "$pr99_evidence/host-tests-inventory.txt"
ctest --test-dir "$pr99_build" --output-on-failure -R '^(persistent_prefix_test|q36_persistent_payload_test|m31_prefix_codec_check|test_ds41_persistent_identity|test_ds41_persistent_wire|test_q36_adapter_protocol)$' | tee "$pr99_evidence/host-tests.log"
for pr99_binary in strata strata-q36 ds41_generate; do
  sha256sum "$pr99_build/$pr99_binary"
  stat --printf='%n %s bytes\n' "$pr99_build/$pr99_binary"
done > "$pr99_evidence/native-sha256-size.txt"
cp "$pr99_build/compile_commands.json" "$pr99_evidence/compile_commands.json"
git -C "$pr99_source" diff --exit-code > "$pr99_evidence/feature-source-diff.txt"
printf '%s\n' 'SUPPORTED HIP TARGET COMPILE/HOST-TEST PASS; AMD GPU/model execution NOT RUN. DS41 is CPU tier on HIP; M31 HIP target unsupported by existing CMake. SYCL and Metal not tested by this job.' > "$pr99_evidence/RESULT.txt"

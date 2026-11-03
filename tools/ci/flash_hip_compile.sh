#!/usr/bin/env bash
# Compile-only, own-fork hosted runner. No AMD device, model downloads or inference.
set -euo pipefail
export LC_ALL=C
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONDONTWRITEBYTECODE=1
[[ ${GITHUB_ACTIONS:-} == true && ${GITHUB_REPOSITORY:-} == NektoDron/Xolotl && ${RUNNER_OS:-} == Linux ]]
[[ ${FLASH_SOURCE_SHA:-} == 1290e14f886c326ef8484f00004562bb2407be5b ]]
[[ ${FLASH_LLAMA_SHA:-} == 6050b467c07a95c7b87debd4ddac2f8ba8d9b807 ]]
flash_source=$(realpath "$1")
flash_evidence=$(realpath -m "$2")
flash_harness=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$flash_evidence"
exec > >(tee "$flash_evidence/qualification.log") 2>&1
flash_build="$GITHUB_WORKSPACE/flash-build-hip"
flash_finish() {
  flash_status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$flash_status" > "$flash_evidence/exit-code.txt"
  df -B1 "$flash_evidence" > "$flash_evidence/disk-final.txt"
  free -b > "$flash_evidence/memory-final.txt"
  for flash_diag in CMakeCache.txt compile_commands.json CMakeFiles/CMakeConfigureLog.yaml Testing/Temporary/LastTest.log; do
    [[ ! -f "$flash_build/$flash_diag" ]] || cp "$flash_build/$flash_diag" "$flash_evidence/$(basename "$flash_diag")"
  done
  exit "$flash_status"
}
trap flash_finish EXIT
[[ $(git -C "$flash_source" rev-parse HEAD) == "$FLASH_SOURCE_SHA" ]]
[[ -z $(git -C "$flash_source" status --porcelain --untracked-files=no) ]]
[[ ! -e /dev/kfd ]] # This job deliberately has no AMD device access.
printf '%s\n' "$FLASH_SOURCE_SHA" > "$flash_evidence/source-head.txt"
printf '%s\n' "$GITHUB_SHA" > "$flash_evidence/harness-head.txt"
printf '%s\n' 'Compile/link only. No model or device execution. AMD GPU hardware qualification NOT RUN.' > "$flash_evidence/SCOPE.txt"
uname -a > "$flash_evidence/host.txt"
df -B1 "$flash_evidence" > "$flash_evidence/disk-before.txt"
free -b > "$flash_evidence/memory-before.txt"
python3 "$flash_harness/pr99_rocm_plan.py" --out "$flash_evidence/rocm-metadata.json" --packages "$flash_evidence/sdk-packages.txt"

# Official AMD 7.0.2/noble repository, authenticated by its published signing key.
# This is an ephemeral hosted runner; no amdgpu/dkms or model packages are requested.
curl -fL --retry 3 https://repo.radeon.com/rocm/rocm.gpg.key -o "$flash_evidence/amd-repository-key.asc"
printf '%s  %s\n' '2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b' "$flash_evidence/amd-repository-key.asc" | sha256sum --check
# Primary fingerprint observed from AMD's official URL; fail closed on key rotation.
gpg --show-keys --with-colons "$flash_evidence/amd-repository-key.asc" > "$flash_evidence/key-identity.txt"
[[ $(awk -F: '$1=="fpr"{print $10;exit}' "$flash_evidence/key-identity.txt") == CA8BB4727A47B4D09B4EE8969386B48A1A693C5C ]]
gpg --dearmor --output "$flash_evidence/amd-repository-key.gpg" "$flash_evidence/amd-repository-key.asc"
sudo install -m 0644 "$flash_evidence/amd-repository-key.gpg" /etc/apt/keyrings/flash-rocm.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/flash-rocm.gpg] https://repo.radeon.com/rocm/apt/7.0.2 noble main' | sudo tee /etc/apt/sources.list.d/flash-rocm.list
sudo apt-get update | tee "$flash_evidence/apt-update.log"
mapfile -t flash_sdk < "$flash_evidence/sdk-packages.txt"
flash_host=(build-essential cmake ninja-build libnuma-dev python3 git)
sudo apt-get -s --no-install-recommends install "${flash_sdk[@]}" "${flash_host[@]}" | tee "$flash_evidence/apt-simulate.log"
sudo apt-get --print-uris --download-only --yes --no-install-recommends install "${flash_sdk[@]}" "${flash_host[@]}" | tee "$flash_evidence/apt-uri-plan.log"
python3 - "$flash_evidence" <<'PY'
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
# The local CUDA build directory was observed at 6.1 GiB. Reserve 16 GiB
# for two isolated builds plus headroom; this is deliberately not called measured HIP use.
reserve = 16 << 30
needed = archives + known + unknown + reserve
receipt = dict(free_bytes=free, root_free_bytes=root_free, workspace_free_bytes=workspace_free,
               apt_archive_bytes=archives, known_selected_installed_bytes=known,
               unknown_installed_allowance_bytes=unknown,
               build_reserve_bytes=reserve, build_reserve_basis='16GiB allowance for two isolated builds from observed 6.1GiB single local CUDA tree; HIP footprint unmeasured', planned_peak_bytes=needed, mem_available_bytes=mem['MemAvailable'], jobs=1,
               scope='Resolved apt plan including Ubuntu deps; installed-size metadata plus 4x archive allowance for missing fields. Not measured peak RAM/build size.')
(out/'resource-preflight.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert free >= needed, 'Insufficient measured disk for conservative SDK/archive plan; no SDK download/install started'
assert mem['MemAvailable'] >= 1 << 30, 'Less than 1GiB observed memory headroom; no SDK download/install started'
PY
sudo apt-get --yes --no-install-recommends -o APT::Keep-Downloaded-Packages=false -o Binary::apt-get::APT::Keep-Downloaded-Packages=false install "${flash_sdk[@]}" "${flash_host[@]}" | tee "$flash_evidence/apt-install.log"
dpkg-query -W -f='${Package}\t${Version}\t${Installed-Size}\n' "${flash_sdk[@]%%=*}" > "$flash_evidence/installed-sdk.tsv"
df -B1 "$flash_evidence" > "$flash_evidence/disk-after-sdk.txt"
free -b > "$flash_evidence/memory-after-sdk.txt"
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2
[[ -x "$ROCM_PATH/llvm/bin/clang++" ]]
"$ROCM_PATH/llvm/bin/clang++" --version > "$flash_evidence/compiler.txt"
cmake --version > "$flash_evidence/cmake-version.txt"

# Exact existing CMake pin. Fetch source only; no model and no helper GPU program.
flash_llama="$GITHUB_WORKSPACE/flash-llama"
git init -q "$flash_llama"
git -C "$flash_llama" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$flash_llama" fetch --depth 1 origin "$FLASH_LLAMA_SHA"
git -C "$flash_llama" checkout --detach FETCH_HEAD
[[ $(git -C "$flash_llama" rev-parse HEAD) == "$FLASH_LLAMA_SHA" ]]
printf '%s\n' "$FLASH_LLAMA_SHA" > "$flash_evidence/ggml-head.txt"
# This patch is required by this source's CMake (not a workaround or guard change).
sha256sum "$flash_source/tools/xolotl/ggml/xolotl-ggml.patch" > "$flash_evidence/ggml-patch-sha256.txt"
git -C "$flash_llama" apply --check "$flash_source/tools/xolotl/ggml/xolotl-ggml.patch"
git -C "$flash_llama" apply "$flash_source/tools/xolotl/ggml/xolotl-ggml.patch"
git -C "$flash_llama" diff --binary > "$flash_evidence/ggml-applied.patch"

for flash_mode in OFF ON; do
  flash_build="$GITHUB_WORKSPACE/flash-build-hip-$flash_mode"
  cmake -S "$flash_source" -B "$flash_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DSTRATA_GGML_DIR="$flash_llama" -DSTRATA_NATIVE_EXPERTS=ON \
    -DSTRATA_FLASH_GRAMMAR="$flash_mode" -DSTRATA_Q36_GRAMMAR=OFF -DSTRATA_M31=OFF \
    -DSTRATA_BUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_HIP_FLAGS=-ftemplate-backtrace-limit=1 -DCMAKE_CXX_FLAGS=-ftemplate-backtrace-limit=1 \
    | tee "$flash_evidence/configure-$flash_mode.log"
  flash_targets=(strata)
  if [[ $flash_mode == ON ]]; then flash_targets+=(flash-grammar-state-check); fi
  cmake --build "$flash_build" --target "${flash_targets[@]}" -j1 | tee "$flash_evidence/build-$flash_mode.log"
  cp "$flash_build/CMakeCache.txt" "$flash_evidence/CMakeCache-$flash_mode.txt"
  cp "$flash_build/compile_commands.json" "$flash_evidence/compile_commands-$flash_mode.json"
  ninja -C "$flash_build" -t commands strata > "$flash_evidence/commands-$flash_mode.txt"
  python3 - "$flash_evidence" "$flash_mode" <<'PYISO'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
commands=json.loads((out/f'compile_commands-{mode}.json').read_text())
rows=[r for r in commands if r['file'].endswith('/src/program/generate.cpp') and 'CMakeFiles/strata.dir/' in r['command']]
assert len(rows)==1, rows
assert ('-DSTRATA_Q36_GRAMMAR=1' in rows[0]['command']) == (mode=='ON')
link=(out/f'commands-{mode}.txt').read_text().splitlines()[-1]
assert ('libq36_llama' in link) == (mode=='ON'), link
(out/f'isolation-{mode}.txt').write_text(f'PASS: Flash grammar {mode} definition and q36_llama link isolation\n')
PYISO
  sha256sum "$flash_build/strata" > "$flash_evidence/binary-$flash_mode.sha256"
  if [[ $flash_mode == ON ]]; then
    sha256sum "$flash_build/flash-grammar-state-check" >> "$flash_evidence/binary-$flash_mode.sha256"
  fi
done
git -C "$flash_source" diff --exit-code > "$flash_evidence/feature-source-diff.txt"
printf '%s\n' 'HIP strata OFF/ON and ON checker compile/link PASS; no device/model/checker execution. Metal/SYCL not tested here.' > "$flash_evidence/RESULT.txt"

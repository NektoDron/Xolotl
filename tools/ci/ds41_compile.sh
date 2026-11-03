#!/usr/bin/env bash
# Compile-only, own-fork hosted runner. No AMD device, model downloads or inference.
set -euo pipefail
export LC_ALL=C
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONDONTWRITEBYTECODE=1
[[ ${GITHUB_ACTIONS:-} == true && ${GITHUB_REPOSITORY:-} == NektoDron/Xolotl && ${RUNNER_OS:-} == Linux ]]
[[ ${DS41_SOURCE_SHA:-} == f79c2e248c4e96b2924445df1b6f4b34cceb4eb4 ]]
[[ ${DS41_LLAMA_SHA:-} == 6050b467c07a95c7b87debd4ddac2f8ba8d9b807 ]]
ds41_source=$(realpath "$1")
ds41_evidence=$(realpath -m "$2")
ds41_harness=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ds41_evidence"
exec > >(tee "$ds41_evidence/qualification.log") 2>&1
ds41_build="$GITHUB_WORKSPACE/ds41-build-hip"
ds41_finish() {
  ds41_status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$ds41_status" > "$ds41_evidence/exit-code.txt"
  df -B1 "$ds41_evidence" > "$ds41_evidence/disk-final.txt"
  free -b > "$ds41_evidence/memory-final.txt"
  for ds41_diag in CMakeCache.txt compile_commands.json CMakeFiles/CMakeConfigureLog.yaml Testing/Temporary/LastTest.log; do
    [[ ! -f "$ds41_build/$ds41_diag" ]] || cp "$ds41_build/$ds41_diag" "$ds41_evidence/$(basename "$ds41_diag")"
  done
  exit "$ds41_status"
}
trap ds41_finish EXIT
[[ $(git -C "$ds41_source" rev-parse HEAD) == "$DS41_SOURCE_SHA" ]]
[[ -z $(git -C "$ds41_source" status --porcelain --untracked-files=no) ]]
[[ ! -e /dev/kfd ]] # This job deliberately has no AMD device access.
printf '%s\n' "$DS41_SOURCE_SHA" > "$ds41_evidence/source-head.txt"
printf '%s\n' "$GITHUB_SHA" > "$ds41_evidence/harness-head.txt"
printf '%s\n' 'Compile/link only. No model or device execution. AMD GPU hardware qualification NOT RUN.' > "$ds41_evidence/SCOPE.txt"
uname -a > "$ds41_evidence/host.txt"
df -B1 "$ds41_evidence" > "$ds41_evidence/disk-before.txt"
free -b > "$ds41_evidence/memory-before.txt"
python3 "$ds41_harness/pr99_rocm_plan.py" --out "$ds41_evidence/rocm-metadata.json" --packages "$ds41_evidence/sdk-packages.txt"

# Official AMD 7.0.2/noble repository, authenticated by its published signing key.
# This is an ephemeral hosted runner; no amdgpu/dkms or model packages are requested.
curl -fL --retry 3 https://repo.radeon.com/rocm/rocm.gpg.key -o "$ds41_evidence/amd-repository-key.asc"
printf '%s  %s\n' '2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b' "$ds41_evidence/amd-repository-key.asc" | sha256sum --check
# Primary fingerprint observed from AMD's official URL; fail closed on key rotation.
gpg --show-keys --with-colons "$ds41_evidence/amd-repository-key.asc" > "$ds41_evidence/key-identity.txt"
[[ $(awk -F: '$1=="fpr"{print $10;exit}' "$ds41_evidence/key-identity.txt") == CA8BB4727A47B4D09B4EE8969386B48A1A693C5C ]]
gpg --dearmor --output "$ds41_evidence/amd-repository-key.gpg" "$ds41_evidence/amd-repository-key.asc"
sudo install -m 0644 "$ds41_evidence/amd-repository-key.gpg" /etc/apt/keyrings/ds41-rocm.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/ds41-rocm.gpg] https://repo.radeon.com/rocm/apt/7.0.2 noble main' | sudo tee /etc/apt/sources.list.d/ds41-rocm.list
sudo apt-get update | tee "$ds41_evidence/apt-update.log"
mapfile -t ds41_sdk < "$ds41_evidence/sdk-packages.txt"
ds41_host=(build-essential cmake ninja-build libnuma-dev python3 git)
sudo apt-get -s --no-install-recommends install "${ds41_sdk[@]}" "${ds41_host[@]}" | tee "$ds41_evidence/apt-simulate.log"
sudo apt-get --print-uris --download-only --yes --no-install-recommends install "${ds41_sdk[@]}" "${ds41_host[@]}" | tee "$ds41_evidence/apt-uri-plan.log"
python3 - "$ds41_evidence" <<'PY'
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
# The local CUDA build directory was observed at 6.1 GiB. Reserve 24 GiB
# for four isolated builds plus headroom; this is deliberately not called measured HIP use.
reserve = 24 << 30
needed = archives + known + unknown + reserve
receipt = dict(free_bytes=free, root_free_bytes=root_free, workspace_free_bytes=workspace_free,
               apt_archive_bytes=archives, known_selected_installed_bytes=known,
               unknown_installed_allowance_bytes=unknown,
               build_reserve_bytes=reserve, build_reserve_basis='24GiB allowance for four isolated builds from observed 6.1GiB single local CUDA tree; HIP footprint unmeasured', planned_peak_bytes=needed, mem_available_bytes=mem['MemAvailable'], jobs=1,
               scope='Resolved apt plan including Ubuntu deps; installed-size metadata plus 4x archive allowance for missing fields. Not measured peak RAM/build size.')
(out/'resource-preflight.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert free >= needed, 'Insufficient measured disk for conservative SDK/archive plan; no SDK download/install started'
assert mem['MemAvailable'] >= 1 << 30, 'Less than 1GiB observed memory headroom; no SDK download/install started'
PY
sudo apt-get --yes --no-install-recommends -o APT::Keep-Downloaded-Packages=false -o Binary::apt-get::APT::Keep-Downloaded-Packages=false install "${ds41_sdk[@]}" "${ds41_host[@]}" | tee "$ds41_evidence/apt-install.log"
dpkg-query -W -f='${Package}\t${Version}\t${Installed-Size}\n' "${ds41_sdk[@]%%=*}" > "$ds41_evidence/installed-sdk.tsv"
df -B1 "$ds41_evidence" > "$ds41_evidence/disk-after-sdk.txt"
free -b > "$ds41_evidence/memory-after-sdk.txt"
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2
[[ -x "$ROCM_PATH/llvm/bin/clang++" ]]
"$ROCM_PATH/llvm/bin/clang++" --version > "$ds41_evidence/compiler.txt"
cmake --version > "$ds41_evidence/cmake-version.txt"

# Exact existing CMake pin. Fetch source only; no model and no helper GPU program.
ds41_llama="$GITHUB_WORKSPACE/ds41-llama"
git init -q "$ds41_llama"
git -C "$ds41_llama" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$ds41_llama" fetch --depth 1 origin "$DS41_LLAMA_SHA"
git -C "$ds41_llama" checkout --detach FETCH_HEAD
[[ $(git -C "$ds41_llama" rev-parse HEAD) == "$DS41_LLAMA_SHA" ]]
printf '%s\n' "$DS41_LLAMA_SHA" > "$ds41_evidence/ggml-head.txt"
# This patch is required by this source's CMake (not a workaround or guard change).
sha256sum "$ds41_source/tools/xolotl/ggml/xolotl-ggml.patch" > "$ds41_evidence/ggml-patch-sha256.txt"
git -C "$ds41_llama" apply --check "$ds41_source/tools/xolotl/ggml/xolotl-ggml.patch"
git -C "$ds41_llama" apply "$ds41_source/tools/xolotl/ggml/xolotl-ggml.patch"
git -C "$ds41_llama" diff --binary > "$ds41_evidence/ggml-applied.patch"

for ds41_backend in CPU HIP; do
 for ds41_mode in OFF ON; do
  ds41_tag="$ds41_backend-$ds41_mode"
  ds41_build="$GITHUB_WORKSPACE/ds41-build-$ds41_tag"
  ds41_backend_args=(-DSTRATA_ENABLE_HIP=OFF)
  if [[ $ds41_backend == HIP ]]; then
   ds41_backend_args=(-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH"
    -DCMAKE_PREFIX_PATH="$ROCM_PATH")
  fi
  cmake -S "$ds41_source" -B "$ds41_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
   "${ds41_backend_args[@]}" -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=ON \
   -DSTRATA_GGML_DIR="$ds41_llama" -DSTRATA_DS41_GRAMMAR="$ds41_mode" \
   -DSTRATA_Q36_GRAMMAR=OFF -DSTRATA_M31=OFF -DSTRATA_BUILD_TESTS=ON \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON | tee "$ds41_evidence/configure-$ds41_tag.log"
  cmake --build "$ds41_build" --target ds41_generate ds41_grammar_sampling_test -j1 | tee "$ds41_evidence/build-$ds41_tag.log"
  cp "$ds41_build/compile_commands.json" "$ds41_evidence/compile_commands-$ds41_tag.json"
  cp "$ds41_build/CMakeCache.txt" "$ds41_evidence/CMakeCache-$ds41_tag.txt"
  ninja -C "$ds41_build" -t commands ds41_generate > "$ds41_evidence/commands-$ds41_tag.txt"
  python3 - "$ds41_evidence" "$ds41_tag" "$ds41_mode" <<'PYISO'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]);tag=sys.argv[2];mode=sys.argv[3]
rows=json.loads((out/f'compile_commands-{tag}.json').read_text())
main=[r for r in rows if r['file'].endswith('/tools/v41/engine/ds41_generate.cpp')]
assert len(main)==1
cmd=main[0]['command']
assert ('-DSTRATA_Q36_GRAMMAR=1' in cmd)==(mode=='ON')
assert '-DSTRATA_ENABLE_CUDA=1' not in cmd
commands=(out/f'commands-{tag}.txt').read_text()
assert 'src/kernels/cuda/' not in commands, 'Unexpected unrelated full GPU kernels in DS CPU-tier dependencies'
assert ('libq36_llama' in commands.splitlines()[-1])==(mode=='ON')
if mode=='ON':
 reg=[r for r in rows if 'q36_vocab_ggml.dir/' in r['command']]
 assert len(reg)==2
 for r in reg:
  assert '-DGGML_USE_CPU' in r['command']
  assert '-DGGML_USE_CUDA' not in r['command'] and '-DGGML_USE_HIP' not in r['command']
(out/f'isolation-{tag}.txt').write_text('PASS: CPU DS tier, opt-in link/definition isolation and CPU-only vocab registry\n')
PYISO
  ctest --test-dir "$ds41_build" --output-on-failure -R '^ds41_grammar_sampling_test$' | tee "$ds41_evidence/sampling-$ds41_tag.log"
  sha256sum "$ds41_build/ds41_generate" > "$ds41_evidence/binary-$ds41_tag.sha256"
 done
done
git -C "$ds41_source" diff --exit-code > "$ds41_evidence/feature-source-diff.txt"
printf '%s\n' 'DS CPU and HIP-config CPU-tier OFF/ON compile/link and host sampling test PASS. No models or GPU execution. No AMD GPU tier claimed.' > "$ds41_evidence/RESULT.txt"

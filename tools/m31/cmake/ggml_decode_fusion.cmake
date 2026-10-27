# Extra opt-in runtime API. Apply after the common Xolotl patch, before ggml's
# subdirectory is configured. External vendor trees must already carry it.
function(m31_patch_decode_fusion vendor_dir may_apply)
  if(NOT STRATA_M31)
    return()
  endif()
  # The final incremental patch updates text introduced by the first API patch.
  # Its successful reverse check also proves the prerequisite API/marker context;
  # do not try to reapply an earlier patch to this newer coherent vendor state.
  get_filename_component(_m31_vendor_parent "${vendor_dir}" DIRECTORY)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${_m31_vendor_parent}"
      git apply --reverse --check "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../ggml-rms-only-fusion.patch"
      WORKING_DIRECTORY "${vendor_dir}" RESULT_VARIABLE _m31_rms_patched
      OUTPUT_QUIET ERROR_QUIET)
  if(_m31_rms_patched EQUAL 0)
    return()
  endif()
  foreach(_m31_patch_name ggml-decode-fusion.patch ggml-exact-mmv.patch ggml-rms-only-fusion.patch)
    set(_m31_fusion_patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../${_m31_patch_name}")
    get_filename_component(_m31_vendor_parent "${vendor_dir}" DIRECTORY)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${_m31_vendor_parent}"
        git apply --reverse --check "${_m31_fusion_patch}"
        WORKING_DIRECTORY "${vendor_dir}" RESULT_VARIABLE _m31_fusion_patched
        OUTPUT_QUIET ERROR_QUIET)
    if(_m31_fusion_patched EQUAL 0)
      continue()
    endif()
    if(NOT may_apply)
      message(FATAL_ERROR "STRATA_M31 requires ${_m31_fusion_patch} in external STRATA_GGML_DIR=${vendor_dir}; apply it to your own vendor tree or use the managed FetchContent checkout")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${_m31_vendor_parent}"
        git apply "${_m31_fusion_patch}"
        WORKING_DIRECTORY "${vendor_dir}" RESULT_VARIABLE _m31_fusion_apply)
    if(NOT _m31_fusion_apply EQUAL 0)
      message(FATAL_ERROR "M31 runtime API patch does not apply to ${vendor_dir}")
    endif()
    message(STATUS "M31: applied optional runtime API ${_m31_patch_name}")
  endforeach()
endfunction()

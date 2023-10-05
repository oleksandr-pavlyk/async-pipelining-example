# Pipelining with dpctl

This repo requires changes from https://github.com/IntelPython/dpctl/pull/1395

If this PR is merged, that the required changes would work with recent builds of `dpctl`.

The workflow is inspired by snippet from

https://github.com/IntelPython/numba-dpex/issues/147

### Running example

This execution is done on a laptop and executes on Iris Xe integrated GPU.

```bash
(dev_dpctl) opavlyk@opavlyk-mobl:~/repos/gh_147$ python gh_147_pipeline.py 1000000 8
timing 1000000 elements for 8 iterations
using 3.814697 MB of memory
pipeline time tot|pci|cmp|speedup: (0.9013102054595947, (host_dt=0.002901058178395033, device_dt=0.34462864800000004), (host_dt=0.002176661742851138, device_dt=1.0035322960000002))
serial   time tot|pci|cmp|speedup: (0.8824286460876465, (host_dt=0.0002622619504109025, device_dt=0.0015769520000000002), (host_dt=0.00016310496721416712, device_dt=0.87868144))
serial0   time tot|pci|cmp|speedup: (0.882073163986206, (host_dt=0.00017874408513307571, device_dt=0.880322248))
```


### Rebuilding SPIR-V file from source

Instruction assumes Linux.

Assumes that oneAPI compiler has been activated. This is usually done using
`source /opt/intel/oneapi/setvars.sh`,
or `source /opt/intel/oneapi/compiler/latest/env/vars.sh`.

```bash
export TOOLS_DIR=$(dirname $(dirname $(which icx)))/bin-llvm
export FNAME=increment_by_apery
$TOOLS_DIR/clang -cc1 -triple spir ${FNAME}.cl -finclude-default-header -flto -emit-llvm-bc -o ${FNAME}.bc
$TOOLS_DIR/llvm-spirv ${FNAME}.bc -o ${FNAME}.spv
rm ${FNAME}.bc
```

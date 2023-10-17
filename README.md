# Pipelining with dpctl

This repo requires changes from https://github.com/IntelPython/dpctl/pull/1395

If this PR is merged, that the required changes would work with recent builds of `dpctl`.

The workflow is inspired by snippet from

https://github.com/IntelPython/numba-dpex/issues/147

### Running example

This execution is done on a laptop and executes on Iris Xe integrated GPU.

```bash
(dev_dpctl) opavlyk@opavlyk-mobl:~/repos/gh_147$ python gh_147_pipeline.py 1000000 8 pipeline
timing 100000 elements for 8 iterations
using 0.381470 MB of memory
pipeline time tot|pci|cmp|speedup: (0.15003323554992676, (host_dt=0.002088017005007714, device_dt=0.0016425600000000002), (host_dt=0.09559680399979698, device_dt=0.14533952))
pipeline time tot|pci|cmp|speedup: (0.0033538341522216797, (host_dt=0.0011560160128283314, device_dt=0.00117424), (host_dt=0.0001863209981820546, device_dt=0.0009238400000000001))
pipeline time tot|pci|cmp|speedup: (0.0036134719848632812, (host_dt=0.0012183169965283014, device_dt=0.00126176), (host_dt=0.0001772840041667223, device_dt=0.0009169600000000001))
pipeline time tot|pci|cmp|speedup: (0.0033020973205566406, (host_dt=0.0012958299921592698, device_dt=0.00134496), (host_dt=0.0001742950189509429, device_dt=0.00091472))
pipeline time tot|pci|cmp|speedup: (0.003053426742553711, (host_dt=0.0010881849957513623, device_dt=0.00111776), (host_dt=0.00017004498658934608, device_dt=0.0009150400000000001))
```

Supported algorithms (last parameter) are: "pipeline", "pipeline_no_timer", "serial", "serial_no_timer", "serial0".


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

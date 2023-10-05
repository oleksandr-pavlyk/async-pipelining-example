# Rebuilding SPIR-V file from source

```bash
export TOOLS_DIR=$(dirname $(dirname $(which icx)))/bin-llvm
export FNAME=increment_by_one
$TOOLS_DIR/clang -cc1 -triple spir ${FNAME}.cl -finclude-default-header -flto -emit-llvm-bc -o ${FNAME}.bc
$TOOLS_DIR/llvm-spirv ${FNAME}.bc -o ${FNAME}.spv
rm ${FNAME}.bc
```

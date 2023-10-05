import numpy as np
import sys
import time

import dpctl
import dpctl.program
import dpctl.tensor as dpt
import ctypes

spirv_file = "./increment_by_apery.spv"
with open(spirv_file, "rb") as fin:
    spirv = fin.read()
program_cache = dict()
kernel_cache = dict()

def get_kernel(q, nm):
    global kernel_cache, program_cache
    ctx = q.sycl_context
    krn = None
    if ctx in kernel_cache:
        _c = kernel_cache[ctx]
        if nm in _c:
            krn = _c[nm]
    if krn is None:
        global spirv
        prog = dpctl.program.create_program_from_spirv(q, spirv)
        program_cache[ctx] = prog
        krn = prog.get_sycl_kernel(nm)
        kernel_cache[ctx] = {nm : krn}
    return krn
    

def increment_by_one(an_array, gws, lws):
    q = an_array.sycl_queue
    krn = get_kernel(q, "increment_by_one")

    args = [an_array.usm_data, ctypes.c_uint32(an_array.size),]
    return q.submit_async(krn, args, [gws,], [lws,])


def run_serial(host_arr, gws, lws, n_itr):
    q = dpctl.SyclQueue(property=["in_order", "enable_profiling"])

    timer_t = dpctl.SyclTimer()
    timer_c = dpctl.SyclTimer()

    # Copy host array into USM-host temporary, as copying from USM-host
    # to USM-device is much faster than from generic host allocation
    a_usm_host = dpt.asarray(host_arr, usm_type="host", sycl_queue=q)
    usm_host_data = a_usm_host.usm_data

    t0 = time.time()
    for _ in range(n_itr):
        with timer_t(q):
            _a = dpt.empty(a_usm_host.shape, usm_type="device", sycl_queue=q)
            _a_data = _a.usm_data
            e_copy = q.memcpy_async(_a.usm_data, usm_host_data, usm_host_data.nbytes)

        with timer_c(q):
            e_compute = increment_by_one(_a, gws, lws)

    q.wait()
    dt = time.time() - t0

    return dt, timer_t.dt, timer_c.dt


def run_serial0(host_arr, gws, lws, n_itr):
    q = dpctl.SyclQueue(property=["in_order", "enable_profiling"])

    timer = dpctl.SyclTimer()

    a_usm_host = dpt.asarray(host_arr, usm_type="host", sycl_queue=q)
    usm_host_data = a_usm_host.usm_data

    t0 = time.time()
    _a = dpt.empty(a_usm_host.shape, usm_type="device", sycl_queue=q)
    _a_data = _a.usm_data
    with timer(q):
        for _ in range(n_itr):
            e_copy = q.memcpy_async(_a.usm_data, usm_host_data, usm_host_data.nbytes)
            e_compute = increment_by_one(_a, gws, lws)

    q.wait()
    dt = time.time() - t0

    return dt, timer.dt



def run_pipeline(host_arr, gws, lws, n_itr):
    q_a = dpctl.SyclQueue(property=["in_order", "enable_profiling"])
    q_b = dpctl.SyclQueue(property=["in_order", "enable_profiling"])

    timer_t = dpctl.SyclTimer()
    timer_c = dpctl.SyclTimer()

    a_usm_host = dpt.asarray(host_arr, usm_type="host", sycl_queue=q_a)
    usm_host_data = a_usm_host.usm_data

    t0 = time.time()
    with timer_t(q_a):
        _a = dpt.empty(a_usm_host.shape, usm_type="device", sycl_queue=q_a)
        _a_data = _a.usm_data
        e_copy_a = q_a.memcpy_async(_a_data, usm_host_data, usm_host_data.nbytes)

    for i in range(n_itr-1):
        if i % 2 == 0:
            with timer_t(q_b):
                _b = dpt.empty(a_usm_host.shape, usm_type="device", sycl_queue=q_b)
                _b_data = _b.usm_data
                e_copy_b = q_b.memcpy_async(_b_data, usm_host_data, usm_host_data.nbytes)

            with timer_c(q_a):
                e_compute_a = increment_by_one(_a, gws, lws)

        else:
            with timer_t(q_a):
                _a = dpt.empty(a_usm_host.shape, usm_type="device", sycl_queue=q_a)
                _a_data = _a.usm_data
                e_copy_a = q_a.memcpy_async(_a_data, usm_host_data, usm_host_data.nbytes)

            with timer_c(q_b):
                e_compute_b = increment_by_one(_b, gws, lws)

    if n_itr % 2 == 0:
        with timer_c(q_b):
            e_compute_b = increment_by_one(_b, gws, lws)
    else:
        with timer_c(q_a):
            e_compute_a = increment_by_one(_a, gws, lws)        

    q_a.wait()
    q_b.wait()
    dt = time.time() - t0

    return dt, timer_t.dt, timer_c.dt


if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 2_000_000

if len(sys.argv) > 2:
    n_itr = int(sys.argv[2])
else:
    n_itr = 100



print("timing %d elements for %d iterations" % (n, n_itr), flush=True)

print("using %f MB of memory" % (n * 4 /1024/1024), flush=True)

a = np.arange(n, dtype=np.float32)
lws = 32
gws = ((a.size + (lws - 1)) // lws) * lws

# warm-up
dts = run_serial0(a, gws, lws, 1)

dtp = run_pipeline(a, gws, lws, n_itr)
print(f"pipeline time tot|pci|cmp|speedup: {dtp}", flush=True)

dts = run_serial(a, gws, lws, n_itr)
print(f"serial   time tot|pci|cmp|speedup: {dts}", flush=True)

dts0 = run_serial0(a, gws, lws, n_itr)
print(f"serial0   time tot|pci|cmp|speedup: {dts0}", flush=True)

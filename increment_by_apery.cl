/*
                      Data Parallel Control (dpctl)

   Copyright 2020-2023 Intel Corporation

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

===----------------------------------------------------------------------===

 \file
 This file implements a sample OpenCL kernel to use in this example.

===----------------------------------------------------------------------===
*/

__kernel void increment_by_apery(__global float *x, uint n) {
    uint idx = get_global_id(0);

    if (idx < n) {
        for(int i = 0; i < 50000; ++i) {
	    float den = (float)(i+1);
	    x[idx] += (float)1 / (den * den * den);
	}
    }
}

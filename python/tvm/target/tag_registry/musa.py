# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""MUSA target tags."""

from .registry import register_tag


def _register_musa_tag(name, arch, shared_mem, regs, warp_size=32):
    register_tag(
        name,
        {
            "kind": "musa",
            "keys": ["musa", "gpu"],
            "arch": arch,
            "max_shared_memory_per_block": shared_mem,
            "max_threads_per_block": 1024,
            "thread_warp_size": warp_size,
            "registers_per_block": regs,
        },
    )


_register_musa_tag("mthreads/mp31", "mp_31", 98304, 262144)

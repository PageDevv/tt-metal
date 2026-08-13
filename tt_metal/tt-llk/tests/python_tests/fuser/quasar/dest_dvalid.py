# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

from fuser.sfpu_node import SfpuNode

if TYPE_CHECKING:
    from fuser.compute_pipeline import ComputePipeline
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation

UNPACK = "UNPACK"
FPU = "FPU"
SFPU = "SFPU"
PACK = "PACK"

UNPACK_THREAD = "unpack"
MATH_THREAD = "math"
PACK_THREAD = "pack"
SFPU_THREAD = "isolate_sfpu"


CHAIN_ORDER = (UNPACK, FPU, SFPU, PACK)


def chain(pipeline: "ComputePipeline") -> List[str]:
    clients = set()

    for node in pipeline.math_nodes:
        if isinstance(node, SfpuNode):
            clients.add(SFPU)
        elif node.unpack_to_dest.value:
            clients.add(UNPACK)
        else:
            clients.add(FPU)

    for node in pipeline.pack_nodes:
        clients.add(SFPU if isinstance(node, SfpuNode) else PACK)

    return [client for client in CHAIN_ORDER if client in clients]


def _sfpu_thread(pipeline: "ComputePipeline") -> str:
    if any(isinstance(node, SfpuNode) for node in pipeline.pack_nodes):
        return PACK_THREAD
    return MATH_THREAD


def clients_of(pipeline: "ComputePipeline", thread: str) -> List[str]:
    sfpu = [SFPU] if _sfpu_thread(pipeline) == thread else []

    if thread == UNPACK_THREAD:
        return [UNPACK]
    if thread == MATH_THREAD:
        return [FPU] + sfpu
    if thread == PACK_THREAD:
        return [PACK] + sfpu
    return sfpu


def enable(config: "GlobalConfig", operation: "L1Operation", thread: str) -> str:
    if not config.quasar_use_dvalid:
        return ""

    members = chain(operation.math)
    owned = clients_of(operation.math, thread)
    enabled = [client for client in owned if client in members]
    code = ""

    for client in CHAIN_ORDER:
        if client in owned:
            if client not in members:
                code += f"_llk_dest_dvalid_disable_<dest_dvalid_client::{client}>();\n"
        elif enabled:
            call = "include" if client in members else "exclude"
            code += f"_llk_dest_dvalid_{call}_<dest_dvalid_client::{client}>();\n"

    for client in enabled:
        code += f"_llk_dest_dvalid_enable_<dest_dvalid_client::{client}>();\n"

    return code


def disable(config: "GlobalConfig", operation: "L1Operation", thread: str) -> str:
    if not config.quasar_use_dvalid or not operation.is_last_stage:
        return ""

    members = chain(operation.math)
    return "".join(
        f"_llk_dest_dvalid_disable_<dest_dvalid_client::{client}>();\n"
        for client in clients_of(operation.math, thread)
        if client in members
    )


def signal(
    config: "GlobalConfig", operation: "L1Operation", thread: str, client: str
) -> str:
    if config.skip_sync or not config.quasar_use_dvalid:
        return ""
    if client not in clients_of(operation.math, thread):
        return ""
    if client not in chain(operation.math):
        return ""

    params = f"dest_dvalid_client::{client}, {operation.dest_sync.cpp_enum_value}"
    if client == PACK:
        params += f", {config.dest_acc.cpp_enum_value}"

    return f"_llk_dest_dvalid_signal_<{params}>();\n"

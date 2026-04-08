# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PrivilegeDesk — Zero-Standing-Privilege Ops environment for OpenEnv.

A simulated enterprise IAM system where an AI agent handles:
  - Access Decision   (easy):  approve/deny a single access request
  - JIT Escalation   (medium): route through approval chains, set TTL
  - Access Review    (hard):   audit entitlements, revoke risky grants

Public API:
    PrivilegeDeskAction        — structured tool call action
    PrivilegeDeskObservation   — partial view of the IAM world
    PrivilegeDeskEnv           — WebSocket client for the environment server
"""

from .client import PrivilegeDeskEnv
from .models import PrivilegeDeskAction, PrivilegeDeskObservation

__all__ = [
    "PrivilegeDeskAction",
    "PrivilegeDeskObservation",
    "PrivilegeDeskEnv",
]

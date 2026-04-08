"""
Episode Generator — procedurally builds a complete WorldState for one episode.

Every call to generate() with a different seed produces a unique but coherent
enterprise environment: org chart, policies, entitlements, requests, and
hidden ground truth.
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .task_templates import get_task

# ── Name pools ───────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Liam", "Maya", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe",
]
LAST_NAMES = [
    "Chen", "Martinez", "Patel", "Kim", "Johnson", "Williams", "Brown",
    "Jones", "Garcia", "Miller", "Davis", "Wilson", "Taylor", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson",
]
DEPARTMENTS = ["Engineering", "Finance", "Legal", "Security", "Marketing", "Operations"]
RESOURCE_TYPES = ["database", "repository", "cloud_project", "api_gateway", "storage_bucket", "admin_console"]
RESOURCE_PREFIXES = {
    "database": ["prod-db", "staging-db", "analytics-db", "billing-db", "audit-db"],
    "repository": ["core-services", "frontend", "ml-platform", "infra-tools", "data-pipeline"],
    "cloud_project": ["prod-gcp", "staging-gcp", "ml-gcp", "security-gcp"],
    "api_gateway": ["customer-api", "internal-api", "partner-api"],
    "storage_bucket": ["raw-data", "processed-data", "backups", "logs"],
    "admin_console": ["aws-console", "gcp-console", "okta-admin", "github-admin"],
}
ROLES = ["viewer", "editor", "admin", "owner"]
ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2, "owner": 3}

TTL_OPTIONS = [1, 2, 4, 8, 24, 48, 72]   # hours
TICKET_PREFIXES = ["INC", "SEC", "OPS", "ENG"]

RISKY_REASONS = [
    "over_privileged",   # has admin/owner, only needs viewer/editor
    "expired_ttl",       # TTL passed, grant not cleaned up
    "unused_90d",        # no audit log entries in 90 days
    "orphaned_user",     # user changed team but still has access
]


class EpisodeGenerator:
    """Generates complete WorldState dicts for PrivilegeDesk episodes."""

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def generate(
        self,
        task_id: str = "access_decision",
        difficulty_level: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a complete WorldState for one episode.

        Returns:
            world_state dict with 'visible' and 'hidden_state' sections.
        """
        seed = seed if seed is not None else random.randint(0, 999_999)
        rng = random.Random(seed)

        template = get_task(task_id)
        entity_counts = template["required_entities"]

        # 1. Org chart
        num_users = rng.randint(*entity_counts["users"])
        users, org_graph = self._build_org(num_users, rng)

        # 2. Resources
        num_resources = rng.randint(*entity_counts["resources"])
        resources = self._build_resources(num_resources, rng)

        # 3. Policies
        num_policies = rng.randint(*entity_counts["policies"])
        policies = self._build_policies(resources, rng, num_policies)

        # 4. Groups (for access review task)
        groups: Dict[str, Any] = {}
        if "groups" in entity_counts:
            num_groups = rng.randint(*entity_counts["groups"])
            groups = self._build_groups(users, num_groups, rng)

        # 5. Existing entitlements
        num_entitlements = rng.randint(*entity_counts["entitlements"])
        entitlements, risky_ids = self._build_entitlements(
            users, resources, policies, num_entitlements, rng,
            add_risky=(task_id == "access_review")
        )

        # 6. Pending access requests
        num_requests = rng.randint(*entity_counts.get("pending_requests", (1, 1)))
        pending_requests, correct_decisions = self._build_requests(
            users, resources, policies, entitlements, num_requests, rng
        )

        # 7. Approval chains
        approval_chains: Dict[str, Any] = {}
        if "approval_chains" in entity_counts:
            approval_chains = self._build_approval_chains(
                pending_requests, users, org_graph, policies, rng
            )

        # 8. Workflows (for access review)
        workflows: Dict[str, Any] = {}
        if "workflows" in entity_counts:
            num_wf = rng.randint(*entity_counts["workflows"])
            workflows = self._build_workflows(users, resources, entitlements, num_wf, rng)

        # 9. Audit log
        num_audit = rng.randint(*entity_counts.get("audit_entries", (5, 10)))
        audit_log = self._build_audit_log(users, resources, entitlements, num_audit, rng)

        # 10. Hidden state (ground truth — never sent to agent)
        hidden_state = self._build_hidden_state(
            task_id=task_id,
            pending_requests=pending_requests,
            correct_decisions=correct_decisions,
            approval_chains=approval_chains,
            entitlements=entitlements,
            risky_entitlement_ids=risky_ids,
            workflows=workflows,
            policies=policies,
            rng=rng,
        )

        # 11. Target user for access review
        review_target_user_id = None
        if task_id == "access_review":
            # Pick a user who has entitlements (including risky ones)
            candidates = [uid for uid in risky_ids_by_user(risky_ids, entitlements)]
            review_target_user_id = candidates[0] if candidates else list(users.keys())[0]

        # 12. Subgoal tracking
        subgoals = [
            {"id": sg["id"], "description": sg["description"],
             "reward": sg["reward"], "status": "pending"}
            for sg in template["subgoals"]
        ]

        world_state = {
            "world_id": f"pd_{task_id}_{seed}",
            "seed": seed,
            "task_id": task_id,
            "difficulty_level": difficulty_level,
            "task_goal": template["task_goal"],
            "max_steps": template["max_steps"],
            "available_tools": template["available_tools"],
            "current_time": datetime(2024, 4, 8, 9, 0, 0).isoformat(),
            # Visible world
            "users": users,
            "org_graph": org_graph,
            "resources": resources,
            "policies": policies,
            "groups": groups,
            "entitlements": entitlements,
            "pending_requests": pending_requests,
            "approval_chains": approval_chains,
            "workflows": workflows,
            "audit_log": [] ,           # starts empty, populated by agent actions
            "audit_db": audit_log,      # pre-existing audit history (queryable via tool)
            "review_target_user_id": review_target_user_id,
            # Tracking
            "subgoals": subgoals,
            "completion_state": {
                "subgoal_status": {sg["id"]: "pending" for sg in template["subgoals"]},
                "tools_used": {},
                "approvals_routed": [],
                "entitlements_revoked": [],
                "review_submitted": False,
                "decision_submitted": False,
                "grant_activated": False,
            },
            # Ground truth (hidden from agent)
            "hidden_state": hidden_state,
            "_terminated": False,
        }

        return world_state

    # ── Builders ──────────────────────────────────────────────────────────────

    def _build_org(self, num_users: int, rng: random.Random):
        """Build users and manager hierarchy."""
        names_pool = [(f, l) for f in rng.sample(FIRST_NAMES, min(num_users, len(FIRST_NAMES)))
                      for l in [rng.choice(LAST_NAMES)]][:num_users]
        if len(names_pool) < num_users:
            names_pool = [(rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES))
                          for _ in range(num_users)]

        users = {}
        ids = [f"user_{i:03d}" for i in range(num_users)]

        # First user is always a manager
        for i, uid in enumerate(ids):
            first, last = names_pool[i]
            dept = rng.choice(DEPARTMENTS)
            is_manager = (i == 0) or (i < num_users // 3 and rng.random() < 0.4)
            manager_id = ids[0] if i > 0 else None
            users[uid] = {
                "user_id": uid,
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}@company.com",
                "department": dept,
                "job_title": "Engineering Manager" if is_manager else rng.choice(
                    ["Software Engineer", "Data Analyst", "Finance Analyst",
                     "Legal Counsel", "Security Engineer", "DevOps Engineer"]
                ),
                "is_manager": is_manager,
                "manager_id": manager_id,
                "status": "active",
            }

        # Build manager → reports mapping
        org_graph = {uid: {"reports_to": u["manager_id"], "department": u["department"]}
                     for uid, u in users.items()}

        return users, org_graph

    def _build_resources(self, num_resources: int, rng: random.Random) -> Dict[str, Any]:
        resources = {}
        resource_type_pool = rng.choices(RESOURCE_TYPES, k=num_resources)
        for i, rtype in enumerate(resource_type_pool):
            rid = f"res_{i:03d}"
            name = rng.choice(RESOURCE_PREFIXES[rtype])
            resources[rid] = {
                "resource_id": rid,
                "name": name,
                "type": rtype,
                "owner_team": rng.choice(DEPARTMENTS),
                "sensitivity": rng.choice(["low", "medium", "high", "critical"]),
                "description": f"{name} ({rtype})",
            }
        return resources

    def _build_policies(self, resources: Dict, rng: random.Random,
                        num_policies: int) -> Dict[str, Any]:
        policies = {}
        resource_ids = list(resources.keys())

        for i in range(min(num_policies, len(resource_ids))):
            pid = f"policy_{i:03d}"
            res_id = resource_ids[i % len(resource_ids)]
            res = resources[res_id]
            sensitivity = res["sensitivity"]

            # Sensitive resources have tighter policies
            if sensitivity == "critical":
                max_role = "viewer"
                max_ttl = rng.choice([1, 2, 4])
                requires_approval = ["manager", "resource_owner", "security_team"]
            elif sensitivity == "high":
                max_role = rng.choice(["viewer", "editor"])
                max_ttl = rng.choice([4, 8])
                requires_approval = ["manager", "resource_owner"]
            elif sensitivity == "medium":
                max_role = rng.choice(["editor", "admin"])
                max_ttl = rng.choice([8, 24])
                requires_approval = ["manager"]
            else:
                max_role = rng.choice(["editor", "admin", "owner"])
                max_ttl = rng.choice([24, 48, 72])
                requires_approval = []

            policies[pid] = {
                "policy_id": pid,
                "resource_id": res_id,
                "resource_type": res["type"],
                "max_role": max_role,
                "max_ttl_hours": max_ttl,
                "requires_approval_from": requires_approval,
                "allowed_departments": rng.sample(DEPARTMENTS, rng.randint(1, 4)),
                "description": (
                    f"Access to {res['name']}: max role={max_role}, "
                    f"max TTL={max_ttl}h, approvals={requires_approval}"
                ),
            }
        return policies

    def _build_entitlements(
        self, users: Dict, resources: Dict, policies: Dict,
        num_entitlements: int, rng: random.Random,
        add_risky: bool = False,
    ):
        entitlements = {}
        risky_ids = []
        user_ids = list(users.keys())
        resource_ids = list(resources.keys())
        now = datetime(2024, 4, 8, 9, 0, 0)

        for i in range(num_entitlements):
            eid = f"ent_{i:03d}"
            uid = rng.choice(user_ids)
            rid = rng.choice(resource_ids)
            role = rng.choice(ROLES)
            grant_time = now - timedelta(days=rng.randint(1, 180))
            ttl_hours = rng.choice(TTL_OPTIONS + [None])  # None = permanent
            is_temporary = ttl_hours is not None
            expiry = (grant_time + timedelta(hours=ttl_hours)) if is_temporary else None
            last_used = grant_time + timedelta(days=rng.randint(0, min(90, (now - grant_time).days)))
            days_since_use = (now - last_used).days

            is_risky = False
            risky_reason = None

            if add_risky and rng.random() < 0.35:
                risky_reason = rng.choice(RISKY_REASONS)
                is_risky = True
                risky_ids.append(eid)

                if risky_reason == "over_privileged":
                    role = rng.choice(["admin", "owner"])
                elif risky_reason == "expired_ttl":
                    expiry = now - timedelta(hours=rng.randint(1, 72))
                    is_temporary = True
                elif risky_reason == "unused_90d":
                    last_used = now - timedelta(days=rng.randint(91, 365))
                    days_since_use = (now - last_used).days
                elif risky_reason == "orphaned_user":
                    users[uid]["status"] = "inactive"

            entitlements[eid] = {
                "entitlement_id": eid,
                "user_id": uid,
                "resource_id": rid,
                "role": role,
                "is_temporary": is_temporary,
                "granted_at": grant_time.isoformat(),
                "expires_at": expiry.isoformat() if expiry else None,
                "granted_by": rng.choice(user_ids),
                "last_used": last_used.isoformat(),
                "days_since_use": days_since_use,
                "source": rng.choice(["direct", "group_inherited"]),
                # Hidden risky metadata — visible only via entitlement.inspect
                "_is_risky": is_risky,
                "_risky_reason": risky_reason,
            }

        return entitlements, risky_ids

    def _build_requests(self, users, resources, policies, entitlements,
                        num_requests, rng: random.Random):
        requests = {}
        correct_decisions = {}
        user_ids = list(users.keys())
        resource_ids = list(resources.keys())
        policy_list = list(policies.values())

        for i in range(num_requests):
            req_id = f"req_{i:03d}"
            requester_id = rng.choice(user_ids)
            resource_id = rng.choice(resource_ids)
            resource = resources[resource_id]
            requested_role = rng.choice(ROLES)

            # Find applicable policy
            applicable_policy = next(
                (p for p in policy_list if p["resource_id"] == resource_id),
                policy_list[0] if policy_list else None
            )

            ticket_id = f"{rng.choice(TICKET_PREFIXES)}-{rng.randint(1000, 9999)}"
            reason = rng.choice([
                "Incident response requires read access",
                "Production deployment needs elevated rights",
                "Security audit requires log access",
                "Migration task requires data access",
                "Client demo setup needs temporary access",
            ])

            # Compute correct decision
            if applicable_policy:
                correct_role = applicable_policy["max_role"]  # must not exceed this
                correct_ttl = applicable_policy["max_ttl_hours"]
                # Approve if requested role doesn't exceed max_role
                should_approve = (
                    ROLE_RANK.get(requested_role, 99)
                    <= ROLE_RANK.get(correct_role, 0)
                )
            else:
                correct_role = "viewer"
                correct_ttl = 4
                should_approve = False

            requests[req_id] = {
                "request_id": req_id,
                "requester_id": requester_id,
                "resource_id": resource_id,
                "resource_name": resource["name"],
                "resource_type": resource["type"],
                "requested_role": requested_role,
                "reason": reason,
                "ticket_id": ticket_id,
                "status": "pending",
                "submitted_at": datetime(2024, 4, 8, 8, 30, 0).isoformat(),
                "applicable_policy_id": applicable_policy["policy_id"] if applicable_policy else None,
            }

            correct_decisions[req_id] = {
                "should_approve": should_approve,
                "correct_role": correct_role,
                "correct_ttl_hours": correct_ttl,
                "correct_justification_category": (
                    "incident_response" if "Incident" in reason
                    else "deployment" if "deployment" in reason
                    else "audit" if "audit" in reason
                    else "operational"
                ),
            }

        return requests, correct_decisions

    def _build_approval_chains(self, requests, users, org_graph, policies, rng):
        chains = {}
        user_ids = list(users.keys())

        for req_id, req in requests.items():
            policy_id = req.get("applicable_policy_id")
            policy = policies.get(policy_id, {})
            approvers_needed = policy.get("requires_approval_from", ["manager"])

            chain = []
            requester_id = req["requester_id"]

            for role_needed in approvers_needed:
                if role_needed == "manager":
                    mgr = org_graph.get(requester_id, {}).get("reports_to")
                    chain.append({
                        "approver_role": "manager",
                        "approver_id": mgr or rng.choice(user_ids),
                        "status": "pending",
                    })
                elif role_needed == "resource_owner":
                    chain.append({
                        "approver_role": "resource_owner",
                        "approver_id": rng.choice(user_ids),
                        "status": "pending",
                    })
                elif role_needed == "security_team":
                    chain.append({
                        "approver_role": "security_team",
                        "approver_id": rng.choice([u for u in user_ids
                                                    if users[u]["department"] == "Security"]
                                                   or user_ids),
                        "status": "pending",
                    })

            chains[req_id] = {
                "request_id": req_id,
                "approver_chain": chain,
                "current_step": 0,
            }

        return chains

    def _build_groups(self, users, num_groups, rng):
        groups = {}
        user_ids = list(users.keys())

        for i in range(num_groups):
            gid = f"group_{i:03d}"
            dept = rng.choice(DEPARTMENTS)
            members = rng.sample(user_ids, min(rng.randint(2, 5), len(user_ids)))
            groups[gid] = {
                "group_id": gid,
                "name": f"{dept} Team",
                "department": dept,
                "members": members,
            }

        return groups

    def _build_workflows(self, users, resources, entitlements, num_wf, rng):
        workflows = {}
        user_ids = list(users.keys())
        ent_ids = list(entitlements.keys())

        for i in range(num_wf):
            wid = f"wf_{i:03d}"
            user_id = rng.choice(user_ids)
            # Each workflow depends on some entitlements
            deps = rng.sample(ent_ids, min(rng.randint(1, 3), len(ent_ids)))
            # Filter to entitlements owned by this user
            user_ents = [e for e in deps
                         if entitlements[e]["user_id"] == user_id]
            deps = user_ents or deps[:1]

            workflows[wid] = {
                "workflow_id": wid,
                "name": rng.choice([
                    "Daily data pipeline", "Incident response runbook",
                    "Monthly billing reconciliation", "Production deployment",
                    "Security audit export",
                ]),
                "user_id": user_id,
                "depends_on_entitlements": deps,
                "is_active": rng.random() > 0.3,
                "last_run": (datetime(2024, 4, 8, 0, 0) - timedelta(hours=rng.randint(1, 48))).isoformat(),
            }

        return workflows

    def _build_audit_log(self, users, resources, entitlements, num_entries, rng):
        entries = []
        user_ids = list(users.keys())
        resource_ids = list(resources.keys())
        actions = ["read", "write", "list", "delete", "admin_action"]

        now = datetime(2024, 4, 8, 9, 0, 0)
        for i in range(num_entries):
            uid = rng.choice(user_ids)
            rid = rng.choice(resource_ids)
            action = rng.choice(actions)
            ts = now - timedelta(days=rng.randint(0, 120), hours=rng.randint(0, 23))
            entries.append({
                "entry_id": f"audit_{i:04d}",
                "user_id": uid,
                "resource_id": rid,
                "action": action,
                "timestamp": ts.isoformat(),
                "outcome": rng.choice(["success", "success", "success", "denied"]),
                "source_ip": f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            })

        return sorted(entries, key=lambda e: e["timestamp"], reverse=True)

    def _build_hidden_state(self, task_id, pending_requests, correct_decisions,
                            approval_chains, entitlements, risky_entitlement_ids,
                            workflows, policies, rng):
        hidden = {
            "correct_decisions": correct_decisions,
            "correct_approval_chains": {
                req_id: [step["approver_id"] for step in chain["approver_chain"]]
                for req_id, chain in approval_chains.items()
            },
            "risky_entitlement_ids": risky_entitlement_ids,
            "minimum_revocation_set": risky_entitlement_ids[:],
            "workflow_critical_entitlements": [
                eid
                for wf in workflows.values()
                if wf.get("is_active")
                for eid in wf.get("depends_on_entitlements", [])
            ],
        }
        return hidden


def generate_episode(task_id="access_decision", difficulty_level=1, seed=42):
    gen = EpisodeGenerator(seed=seed)
    return gen.generate(task_id=task_id, difficulty_level=difficulty_level, seed=seed)


def risky_ids_by_user(risky_ids, entitlements):
    """Return user IDs who have at least one risky entitlement."""
    seen = set()
    for eid in risky_ids:
        uid = entitlements.get(eid, {}).get("user_id")
        if uid and uid not in seen:
            seen.add(uid)
            yield uid

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

from .task_templates import get_task, TASK_GOAL_VARIANTS
from .toxic_graph import IdentityGraph

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

SUB_AGENT_NAMES = [
    "reporting-bot", "analytics-agent", "etl-runner", "audit-crawler",
    "sync-daemon", "ml-pipeline", "data-exporter", "compliance-scanner",
]
SUB_AGENT_PURPOSES = [
    "data aggregation", "compliance scanning", "report generation",
    "log processing", "ETL sync", "analytics refresh", "security audit",
]

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

        Args:
            difficulty_level: 1 (easy) → uses lower bound of entity ranges.
                              2 (medium) → uses midpoint.
                              3 (hard) → uses upper bound.

        Returns:
            world_state dict with 'visible' and 'hidden_state' sections.
        """
        seed = seed if seed is not None else random.randint(0, 999_999)

        # Two independent RNGs:
        #   content_rng — seeded only by `seed`, drives all world content (names,
        #                 resources, policies, …). Changing difficulty_level does NOT
        #                 affect this stream, so the same seed always produces the
        #                 same world identity regardless of difficulty.
        #   count_rng   — seeded by seed XOR difficulty_level, used only for the
        #                 ±1 jitter on entity counts. Deterministic for a given
        #                 (seed, difficulty_level) pair but independent of content_rng.
        content_rng = random.Random(seed)
        count_rng   = random.Random(seed ^ (difficulty_level * 0x9e3779b9))

        # Keep `rng` as an alias for content_rng so the rest of the file is unchanged
        rng = content_rng

        template = get_task(task_id)
        entity_counts = template["required_entities"]

        # Scale entity counts based on difficulty level (1=min, 3=max)
        def scaled_count(lo: int, hi: int) -> int:
            """Linearly interpolate between lo and hi based on difficulty (1–3).

            Uses count_rng for jitter so the content_rng stream is never
            consumed during the sizing phase — keeping world content stable
            across difficulty levels for the same seed.
            """
            level = max(1, min(3, difficulty_level))
            t = (level - 1) / 2.0  # 0.0 at level 1, 0.5 at level 2, 1.0 at level 3
            base = int(round(lo + t * (hi - lo)))
            # ±1 jitter via count_rng — does not touch content_rng
            return max(lo, min(hi, base + count_rng.randint(-1, 1)))

        # Also scale max_steps: level 1 → 60% of template max, level 3 → 100%
        template_max_steps = template["max_steps"]
        step_scale = 0.6 + 0.2 * (max(1, min(3, difficulty_level)) - 1)  # 0.6, 0.8, 1.0
        scaled_max_steps = max(3, int(template_max_steps * step_scale))

        # 1. Org chart
        num_users = scaled_count(*entity_counts["users"])
        users, org_graph = self._build_org(num_users, rng)

        # 2. Resources
        num_resources = scaled_count(*entity_counts["resources"])
        resources = self._build_resources(num_resources, rng)

        # 3. Policies
        num_policies = scaled_count(*entity_counts["policies"])
        policies = self._build_policies(resources, rng, num_policies)

        # 4. Groups (for access review task)
        groups: Dict[str, Any] = {}
        if "groups" in entity_counts:
            num_groups = scaled_count(*entity_counts["groups"])
            groups = self._build_groups(users, num_groups, rng)

        # 5. Existing entitlements
        num_entitlements = scaled_count(*entity_counts["entitlements"])
        entitlements, risky_ids = self._build_entitlements(
            users, resources, policies, num_entitlements, rng,
            add_risky=(task_id == "access_review")
        )

        # 6. Pending access requests
        num_requests = scaled_count(*entity_counts.get("pending_requests", (1, 1)))
        pending_requests, correct_decisions = self._build_requests(
            users, resources, policies, entitlements, num_requests, rng
        )

        # 6b. SoD data (separation_of_duties_audit only)
        conflict_matrix: Dict[str, Any] = {}
        compensating_controls: Dict[str, Any] = {}
        sod_true_violations: list = []
        sod_all_violations: list = []
        if task_id == "separation_of_duties_audit":
            num_conflicts = scaled_count(*entity_counts.get("sod_conflicts", (3, 6)))
            num_sod_viol = scaled_count(*entity_counts.get("sod_violations", (2, 6)))
            num_controls = scaled_count(*entity_counts.get("compensating_controls", (1, 3)))
            conflict_matrix = self._build_conflict_matrix(resources, num_conflicts, rng)
            compensating_controls, sod_true_violations, sod_all_violations = (
                self._build_sod_data(
                    users, resources, entitlements, conflict_matrix,
                    num_sod_viol, num_controls, rng, difficulty_level,
                )
            )

        # 6d. Incidents (emergency_breakglass only)
        incidents: Dict[str, Any] = {}
        if task_id == "emergency_breakglass":
            num_incidents = scaled_count(*entity_counts.get("incidents", (1, 2)))
            incidents = self._build_incidents(
                users, resources, num_incidents, rng, difficulty_level
            )
            # Override pending_requests with a single breakglass request
            pending_requests, correct_decisions = self._build_breakglass_request(
                incidents, users, resources, policies, rng
            )

        # 6e. Multi-agent oversight — sub-agents + identity graph + rogue request
        sub_agents: Dict[str, Any] = {}
        identity_graph: Optional[IdentityGraph] = None
        oversight_hidden: Dict[str, Any] = {}
        if task_id == "multi_agent_oversight":
            num_agents = scaled_count(*entity_counts.get("sub_agents", (2, 4)))
            sub_agents = self._build_sub_agents(
                num_agents, users, resources, entitlements, rng
            )
            # Build partial world snapshot needed by graph (resources + entitlements)
            world_partial = {
                "resources": resources,
                "entitlements": entitlements,
                "pending_requests": pending_requests,
                "current_time": datetime(2024, 4, 8, 9, 0, 0).isoformat(),
                "policies": policies,
            }
            identity_graph = self._build_identity_graph(
                users, sub_agents, resources, entitlements, rng, difficulty_level
            )
            rogue_requests, oversight_hidden = self._build_rogue_requests(
                identity_graph, world_partial, rng
            )
            # Merge rogue request into pending_requests
            pending_requests.update(rogue_requests)
            # Update legitimate_request_ids in oversight_hidden now that we know all req IDs
            oversight_hidden["legitimate_request_ids"] = [
                rid for rid in pending_requests
                if rid not in oversight_hidden.get("rogue_request_ids", [])
            ]

        # 7. Approval chains
        approval_chains: Dict[str, Any] = {}
        if "approval_chains" in entity_counts:
            approval_chains = self._build_approval_chains(
                pending_requests, users, org_graph, policies, rng
            )

        # 8. Workflows (for access review)
        workflows: Dict[str, Any] = {}
        if "workflows" in entity_counts:
            num_wf = scaled_count(*entity_counts["workflows"])
            workflows = self._build_workflows(users, resources, entitlements, num_wf, rng)

        # 9. Audit log
        num_audit = scaled_count(*entity_counts.get("audit_entries", (5, 10)))
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
            incidents=incidents,
            sod_true_violations=sod_true_violations,
            sod_all_violations=sod_all_violations,
            oversight_hidden=oversight_hidden,
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
            "task_goal": content_rng.choice(TASK_GOAL_VARIANTS[task_id]).format(
                review_target_user_id=review_target_user_id
            ) if task_id == "access_review" else content_rng.choice(TASK_GOAL_VARIANTS[task_id]),
            "max_steps": scaled_max_steps,
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
            "conflict_matrix": conflict_matrix,
            "compensating_controls": compensating_controls,
            "audit_log": [],            # starts empty, populated by agent actions
            "audit_db": audit_log,      # pre-existing audit history (queryable via tool)
            "review_target_user_id": review_target_user_id,
            # Visible incidents (full details hidden until incident.verify is called)
            "incidents": {
                inc_id: {k: v for k, v in inc.items() if not k.startswith("_")}
                for inc_id, inc in incidents.items()
            },
            # Multi-agent oversight fields (empty for other tasks)
            "sub_agents": sub_agents,
            "identity_graph": identity_graph.get_sanitized_json() if identity_graph else {},
            "rogue_agent_requests": {
                req_id: req
                for req_id, req in pending_requests.items()
                if req.get("_is_rogue", False)
            },
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
                # break-glass specific
                "incident_verified": False,
                "security_flagged": False,
                # SoD audit specific
                "sod_violations_identified": [],
                "sod_controls_checked": [],
                "sod_report_submitted": False,
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

            # Break-glass fields: emergency TTL cap and whether BG is permitted
            breakglass_max_ttl = max(1, min(4, max_ttl // 2))
            breakglass_allowed = rng.random() < 0.80  # 80% of policies allow break-glass

            policies[pid] = {
                "policy_id": pid,
                "resource_id": res_id,
                "resource_type": res["type"],
                "max_role": max_role,
                "max_ttl_hours": max_ttl,
                "requires_approval_from": requires_approval,
                "allowed_departments": rng.sample(DEPARTMENTS, rng.randint(1, 4)),
                "breakglass_max_ttl_hours": breakglass_max_ttl,
                "breakglass_allowed": breakglass_allowed,
                "description": (
                    f"Access to {res['name']}: max role={max_role}, "
                    f"max TTL={max_ttl}h, approvals={requires_approval}, "
                    f"breakglass_allowed={breakglass_allowed}"
                ),
            }
        return policies

    def _build_incidents(
        self, users: Dict, resources: Dict, num_incidents: int,
        rng: random.Random, difficulty_level: int = 1,
    ) -> Dict[str, Any]:
        """Build active/resolved incidents. Hidden fields revealed via incident.verify."""
        incidents = {}
        user_ids = list(users.keys())
        resource_ids = list(resources.keys())

        for i in range(num_incidents):
            inc_id = f"inc_{i:03d}"
            resource_id = rng.choice(resource_ids)
            reporter_id = rng.choice(user_ids)
            others = [u for u in user_ids if u != reporter_id] or user_ids
            on_call_id = rng.choice(others)
            resource = resources.get(resource_id, {})

            if i == 0:
                # Main incident: valid at level 1/2, possibly invalid at level 3
                if difficulty_level < 3:
                    severity = rng.choice(["P1", "P2"])
                    status = "active"
                else:
                    severity = rng.choice(["P1", "P2", "P2", "P3"])
                    status = rng.choice(["active", "active", "resolved"])
            else:
                # Decoy incidents for higher difficulty
                severity = rng.choice(["P3", "P3", "P2"])
                status = rng.choice(["resolved", "resolved", "active"])

            created_at = (
                datetime(2024, 4, 8, 8, 0, 0) - timedelta(hours=rng.randint(0, 3))
            ).isoformat()

            incidents[inc_id] = {
                "incident_id": inc_id,
                "severity": severity,
                "status": status,
                "created_at": created_at,
                "description": f"Production incident on {resource.get('name', resource_id)}",
                # Hidden until agent calls incident.verify
                "_on_call_engineer_id": on_call_id,
                "_affected_resource_id": resource_id,
                "_reporter_id": reporter_id,
            }

        return incidents

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

    def _build_breakglass_request(
        self, incidents: Dict, users: Dict, resources: Dict,
        policies: Dict, rng: random.Random,
    ):
        """Build a single breakglass pending request tied to the main incident."""
        # Main incident is always inc_000
        main_inc = incidents.get("inc_000", next(iter(incidents.values())))
        resource_id = main_inc["_affected_resource_id"]
        on_call_id = main_inc["_on_call_engineer_id"]

        policy = next(
            (p for p in policies.values() if p["resource_id"] == resource_id),
            next(iter(policies.values()), None),
        )
        correct_role = policy["max_role"] if policy else "editor"
        correct_ttl = policy.get("breakglass_max_ttl_hours", 2) if policy else 2

        req_id = "req_000"
        resource = resources.get(resource_id, {})
        requests = {
            req_id: {
                "request_id": req_id,
                "requester_id": on_call_id,
                "resource_id": resource_id,
                "resource_name": resource.get("name", resource_id),
                "resource_type": resource.get("type", ""),
                "requested_role": correct_role,
                "reason": "Emergency break-glass access for production incident",
                "ticket_id": main_inc["incident_id"],
                "status": "pending",
                "submitted_at": datetime(2024, 4, 8, 8, 55, 0).isoformat(),
                "applicable_policy_id": policy["policy_id"] if policy else None,
                "_breakglass": True,
                "_incident_id": main_inc["incident_id"],
            }
        }
        correct_decisions = {
            req_id: {
                "should_approve": (
                    main_inc["status"] == "active"
                    and main_inc["severity"] in ("P1", "P2")
                    and (policy.get("breakglass_allowed", True) if policy else True)
                ),
                "correct_role": correct_role,
                "correct_ttl_hours": correct_ttl,
            }
        }
        return requests, correct_decisions

    def _build_conflict_matrix(self, resources: Dict, num_conflicts: int,
                               rng: random.Random) -> Dict[str, Any]:
        """Build SoD conflict matrix — pairs of (resource_type, min_role) that must not co-exist."""
        resource_types = list(set(r["type"] for r in resources.values()))
        if len(resource_types) < 2:
            resource_types = resource_types + resource_types  # allow same-type conflicts

        conflicts = {}
        for i in range(num_conflicts):
            cid = f"conflict_{i:03d}"
            rtype_a = rng.choice(resource_types)
            rtype_b = rng.choice([t for t in resource_types if t != rtype_a] or resource_types)
            role_a = rng.choice(["editor", "admin", "owner"])
            role_b = rng.choice(["editor", "admin", "owner"])
            conflicts[cid] = {
                "conflict_id": cid,
                "name": f"SoD: {rtype_a}/{role_a}+ with {rtype_b}/{role_b}+",
                "resource_type_a": rtype_a,
                "min_role_a": role_a,
                "resource_type_b": rtype_b,
                "min_role_b": role_b,
                "severity": rng.choice(["high", "critical"]),
                "description": (
                    f"Users must not hold {role_a}+ on {rtype_a} "
                    f"AND {role_b}+ on {rtype_b} simultaneously"
                ),
            }
        return conflicts

    def _build_sod_data(
        self, users: Dict, resources: Dict, entitlements: Dict,
        conflict_matrix: Dict, num_violations: int, num_controls: int,
        rng: random.Random, difficulty_level: int = 1,
    ):
        """Inject SoD-violation entitlements and create compensating controls.

        Modifies `entitlements` in-place by appending violation-causing entries.
        Returns (compensating_controls, true_violations, all_violations).
        """
        now = datetime(2024, 4, 8, 9, 0, 0)
        user_ids = list(users.keys())
        conflicts = list(conflict_matrix.values())
        compensating_controls: Dict[str, Any] = {}
        all_violations = []

        if not conflicts:
            return compensating_controls, [], []

        for i in range(num_violations):
            conflict = conflicts[i % len(conflicts)]
            cid = conflict["conflict_id"]

            # Find resources of each required type
            res_a = next((r for r in resources.values()
                          if r["type"] == conflict["resource_type_a"]), None)
            res_b = next((r for r in resources.values()
                          if r["type"] == conflict["resource_type_b"]), None)
            if not res_a or not res_b:
                continue

            user_id = user_ids[i % len(user_ids)]
            grant_time = now - timedelta(days=rng.randint(10, 60))
            eid_a = f"ent_sod_{i:03d}a"
            eid_b = f"ent_sod_{i:03d}b"

            for eid, res, min_role in [(eid_a, res_a, conflict["min_role_a"]),
                                        (eid_b, res_b, conflict["min_role_b"])]:
                entitlements[eid] = {
                    "entitlement_id": eid,
                    "user_id": user_id,
                    "resource_id": res["resource_id"],
                    "role": min_role,
                    "is_temporary": False,
                    "granted_at": grant_time.isoformat(),
                    "expires_at": None,
                    "granted_by": rng.choice(user_ids),
                    "last_used": (grant_time + timedelta(days=rng.randint(1, 10))).isoformat(),
                    "days_since_use": rng.randint(5, 30),
                    "source": "direct",
                    "_is_risky": False,
                    "_risky_reason": None,
                    "_sod_violation": True,
                    "_conflict_id": cid,
                }

            all_violations.append({
                "user_id": user_id,
                "conflict_id": cid,
                "entitlement_id_a": eid_a,
                "entitlement_id_b": eid_b,
            })

        # Create compensating controls for the first num_controls violations
        for j in range(min(num_controls, len(all_violations))):
            v = all_violations[j]
            ctrl_id = f"ctrl_{j:03d}"
            # At difficulty 3, ~50% of controls are expired
            if difficulty_level == 3 and rng.random() < 0.5:
                expires_at = (now - timedelta(days=rng.randint(1, 30))).isoformat()
                is_active = False
            else:
                expires_at = (now + timedelta(days=rng.randint(30, 180))).isoformat()
                is_active = True
            compensating_controls[ctrl_id] = {
                "control_id": ctrl_id,
                "user_id": v["user_id"],
                "conflict_id": v["conflict_id"],
                "description": rng.choice([
                    "Enhanced monitoring and alerting in place",
                    "Quarterly security review by the access governance team",
                    "Manager approval required for all privileged actions",
                    "Read-only audit logging enforced with SIEM integration",
                ]),
                "expires_at": expires_at,
                "is_active": is_active,
            }

        # True violations = violations without any active compensating control
        mitigated = {
            (c["user_id"], c["conflict_id"])
            for c in compensating_controls.values()
            if c["is_active"]
        }
        true_violations = [
            v for v in all_violations
            if (v["user_id"], v["conflict_id"]) not in mitigated
        ]

        return compensating_controls, true_violations, all_violations

    # ── Multi-Agent Oversight builders ────────────────────────────────────────

    def _build_sub_agents(
        self,
        n_agents: int,
        users: Dict[str, Any],
        resources: Dict[str, Any],
        entitlements: Dict[str, Any],
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Create autonomous sub-agent identities that mirror the user schema."""
        sub_agents: Dict[str, Any] = {}
        user_ids = list(users.keys())
        ent_ids = list(entitlements.keys())
        names = rng.sample(SUB_AGENT_NAMES, min(n_agents, len(SUB_AGENT_NAMES)))

        for i in range(n_agents):
            sid = f"agent_{i:03d}"
            creator = rng.choice(user_ids) if user_ids else None
            purpose = rng.choice(SUB_AGENT_PURPOSES)
            name = names[i] if i < len(names) else f"agent-{i}"

            # Assign 1-2 existing entitlements to each sub-agent (for graph edges)
            agent_ents = rng.sample(ent_ids, min(rng.randint(1, 2), len(ent_ids)))

            sub_agents[sid] = {
                "sub_agent_id": sid,
                "name": name,
                "purpose": purpose,
                "created_by": creator,
                "status": "active",
                "entitlement_ids": agent_ents,
                # Mirrors user schema so existing tools (entitlement.list, sod.check_user) work
                "user_id": sid,
                "department": "Automation",
                "job_title": "Autonomous Agent",
                "is_manager": False,
                "manager_id": creator,
                "email": f"{name}@agents.internal",
            }

        return sub_agents

    def _build_identity_graph(
        self,
        users: Dict[str, Any],
        sub_agents: Dict[str, Any],
        resources: Dict[str, Any],
        entitlements: Dict[str, Any],
        rng: random.Random,
        difficulty: int,
    ) -> IdentityGraph:
        """Build the identity DAG and inject a toxic combination."""
        graph = IdentityGraph()
        world_snapshot = {
            "users": users,
            "resources": resources,
            "entitlements": entitlements,
        }
        graph.build_from_world(world_snapshot, sub_agents, rng)
        graph.inject_toxic_combination(difficulty, rng)
        return graph

    def _build_rogue_requests(
        self,
        identity_graph: IdentityGraph,
        world_state_partial: Dict[str, Any],
        rng: random.Random,
    ) -> tuple:
        """Generate the rogue agent's access request and its hidden metadata."""
        return identity_graph.generate_rogue_request(world_state_partial, rng)

    def _build_hidden_state(self, task_id, pending_requests, correct_decisions,
                            approval_chains, entitlements, risky_entitlement_ids,
                            workflows, policies, incidents=None,
                            sod_true_violations=None, sod_all_violations=None,
                            oversight_hidden=None, rng=None):
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

        # Break-glass ground truth
        if task_id == "emergency_breakglass" and incidents:
            main_inc = incidents.get("inc_000", next(iter(incidents.values())))
            req = next(iter(pending_requests.values()), {})
            policy_id = req.get("applicable_policy_id")
            policy = policies.get(policy_id, {}) if policy_id else {}
            correct_d = correct_decisions.get(req.get("request_id", "req_000"), {})
            hidden["correct_breakglass"] = {
                "incident_id": main_inc["incident_id"],
                "on_call_user_id": main_inc["_on_call_engineer_id"],
                "resource_id": main_inc["_affected_resource_id"],
                "correct_role": correct_d.get("correct_role", policy.get("max_role", "editor")),
                "correct_ttl_hours": correct_d.get(
                    "correct_ttl_hours", policy.get("breakglass_max_ttl_hours", 2)
                ),
                "incident_is_valid": (
                    main_inc["status"] == "active"
                    and main_inc["severity"] in ("P1", "P2")
                ),
                "breakglass_allowed": policy.get("breakglass_allowed", True),
            }

        # SoD ground truth
        if task_id == "separation_of_duties_audit" and sod_true_violations is not None:
            hidden["sod_true_violations"] = sod_true_violations
            hidden["sod_all_violations"] = sod_all_violations or []
            # Minimum revocations: entitlement_id_a from each true violation
            hidden["sod_minimum_revocations"] = [
                v["entitlement_id_a"] for v in sod_true_violations
            ]

        # Multi-agent oversight ground truth
        if task_id == "multi_agent_oversight" and oversight_hidden:
            hidden.update(oversight_hidden)

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

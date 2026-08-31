from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from vision_bot.controllers.combat import CombatController
from vision_bot.controllers.loot import LootController
from vision_bot.controllers.mining_approach import MiningApproachController
from vision_bot.controllers.mounting import MountingController
from vision_bot.controllers.recovery import RecoveryController
from vision_bot.controllers.travel import TravelController
from vision_bot.core.commands import CommandGroup
from vision_bot.core.geometry import PhysicalRoute
from vision_bot.core.hazards import PhysicalHazardGuard
from vision_bot.engine.kernel import DeterministicKernel, DomainController
from vision_bot.engine.route_compiler import RouteCompileReport, compile_configured_route
from vision_bot.engine.supervisor import Supervisor
from vision_bot.v09_config import V09Config


@dataclass(frozen=True)
class V09KernelBundle:
    config: V09Config
    route: PhysicalRoute
    kernel: DeterministicKernel
    controllers: Mapping[CommandGroup, DomainController]
    route_report: RouteCompileReport
    hazards: PhysicalHazardGuard


def build_v09_kernel(project_config: dict) -> V09KernelBundle:
    config = V09Config.from_mapping(project_config)
    compiled = compile_configured_route(project_config, config.zone)
    route = compiled.route
    mining_nodes = compiled.nodes
    controllers = {
        CommandGroup.RECOVERY: RecoveryController(
            config.recovery,
            config.zone,
            config.navigation,
        ),
        CommandGroup.COMBAT: CombatController(config.combat, config.navigation),
        CommandGroup.LOOT: LootController(project_config),
        CommandGroup.MINING: MiningApproachController(
            config.zone,
            config.mining,
            config.navigation,
            mining_nodes,
        ),
        CommandGroup.MOUNT: MountingController(config.mounting),
        CommandGroup.TRAVEL: TravelController(route, config.navigation),
    }
    return V09KernelBundle(
        config=config,
        route=route,
        kernel=DeterministicKernel(
            Supervisor(
                critical_max_age_seconds=config.freshness.critical_seconds,
                operational_max_age_seconds=config.freshness.operational_seconds,
                mounted_escape_seconds=config.navigation.mounted_escape_seconds,
                mounted_escape_attacker_count=(
                    config.navigation.mounted_escape_attacker_count
                ),
                emergency_hp_fraction=config.combat.emergency_hp_fraction,
                route=route,
                corridor_radius_yards=config.navigation.corridor_radius_yards,
            ),
            controllers,
        ),
        controllers=MappingProxyType(controllers),
        route_report=compiled.report,
        hazards=compiled.hazards,
    )

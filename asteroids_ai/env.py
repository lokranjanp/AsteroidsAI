from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # Gymnasium is installed by the project package; the fallback keeps core tests lightweight.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only in minimal local environments.
    class _Env:
        metadata: Dict[str, Any] = {}

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
            del seed, options

    class _Box:
        def __init__(self, low: Any, high: Any, shape: Tuple[int, ...], dtype: Any) -> None:
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

    class _DictSpace(dict):
        def __init__(self, mapping: Dict[str, Any]) -> None:
            super().__init__(mapping)
            self.spaces = mapping

    class _Gym:
        Env = _Env

    class _Spaces:
        Box, Discrete, Dict = _Box, _Discrete, _DictSpace

    gym, spaces = _Gym(), _Spaces()


class Action(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    FIRE = 3
    LEFT_FIRE = 4
    RIGHT_FIRE = 5


@dataclass(frozen=True)
class RewardConfig:
    asteroid_destroyed: float = 1.0
    collision: float = -1.0
    death: float = -2.0
    shot: float = -0.02
    pickup: float = 0.25
    survival: float = 0.001


@dataclass(frozen=True)
class EnvConfig:
    width: int = 600
    height: int = 600
    fps: int = 90
    max_steps: int = 5_000
    max_asteroids_observed: int = 10
    max_bullets_observed: int = 8
    max_pickups_observed: int = 4
    base_asteroid_speed: float = 2.0
    asteroid_speed_step: float = 0.25
    base_spawn_interval: int = 25
    min_spawn_interval: int = 12
    difficulty_interval: int = 1_000
    bullet_speed: float = 5.5
    pickup_speed: float = 2.5
    ship_acceleration: float = 0.4
    ship_drag: float = 0.95
    max_ship_speed: float = 6.0
    fire_cooldown_steps: int = 50
    passive_fuel_cost: float = 0.002
    movement_fuel_cost: float = 0.01
    shot_fuel_cost: float = 0.02
    pickup_drop_probability: float = 0.10
    health_pickup_amount: float = 20.0
    fuel_pickup_amount: float = 10.0

    @property
    def max_entities(self) -> int:
        return self.max_asteroids_observed + self.max_bullets_observed + self.max_pickups_observed


@dataclass
class _Ship:
    x: float
    y: float
    vx: float = 0.0
    health: float = 100.0
    fuel: float = 100.0
    radius: float = 20.0


@dataclass
class _Entity:
    x: float
    y: float
    vx: float
    vy: float
    radius: float


@dataclass
class _Pickup(_Entity):
    kind: str = "fuel"


Observation = Dict[str, np.ndarray]


class AsteroidsEnv(gym.Env):
    """Deterministic vertical dodger-shooter environment.

    Physics are frame based, rendering is optional, and all mutable game state belongs
    to the environment instance so separate runs cannot contaminate each other.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}
    GLOBAL_FEATURES = 7
    ENTITY_FEATURES = 9

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(-1.0, 1.0, (self.GLOBAL_FEATURES,), np.float32),
                "entities": spaces.Box(
                    -1.0,
                    1.0,
                    (self.config.max_entities, self.ENTITY_FEATURES),
                    np.float32,
                ),
                "entity_mask": spaces.Box(0.0, 1.0, (self.config.max_entities,), np.float32),
            }
        )
        self.np_random = np.random.default_rng()
        self._seed: Optional[int] = None
        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self._assets: Dict[str, Any] = {}
        self._closed = False
        self._initialize_state()

    def _initialize_state(self) -> None:
        cfg = self.config
        self.ship = _Ship(cfg.width / 2.0, cfg.height - 100.0)
        self.asteroids: List[_Entity] = []
        self.bullets: List[_Entity] = []
        self.pickups: List[_Pickup] = []
        self.step_count = 0
        self.fire_cooldown = 0
        self.hits = 0
        self.shots = 0
        self.collisions = 0
        self.pickups_collected = 0
        self.terminated = False
        self.truncated = False
        self.death_reason = ""
        self.episode_return = 0.0

    @property
    def difficulty_level(self) -> int:
        return self.step_count // self.config.difficulty_interval

    @property
    def asteroid_speed(self) -> float:
        return self.config.base_asteroid_speed + self.difficulty_level * self.config.asteroid_speed_step

    @property
    def spawn_interval(self) -> int:
        return max(
            self.config.min_spawn_interval,
            self.config.base_spawn_interval - 2 * self.difficulty_level,
        )

    @property
    def score(self) -> int:
        return self.hits * 100 - self.shots * 2

    @property
    def accuracy(self) -> float:
        return self.hits / self.shots if self.shots else 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, Dict[str, Any]]:
        del options
        try:
            super().reset(seed=seed)
        except (AttributeError, TypeError):
            pass
        if seed is not None:
            self._seed = int(seed)
            self.np_random = np.random.default_rng(self._seed)
        self._initialize_state()
        observation = self._observation()
        info = self._info({})
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode end; call reset() first")
        try:
            selected = Action(int(action))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid action {action!r}") from exc

        components = {
            "asteroid_destroyed": 0.0,
            "collision": 0.0,
            "death": 0.0,
            "shot": 0.0,
            "pickup": 0.0,
            "survival": 0.0,
        }
        self.step_count += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        moving_left = selected in (Action.LEFT, Action.LEFT_FIRE)
        moving_right = selected in (Action.RIGHT, Action.RIGHT_FIRE)
        firing = selected in (Action.FIRE, Action.LEFT_FIRE, Action.RIGHT_FIRE)

        if moving_left:
            self.ship.vx -= self.config.ship_acceleration
        elif moving_right:
            self.ship.vx += self.config.ship_acceleration
        self.ship.vx = float(np.clip(self.ship.vx, -self.config.max_ship_speed, self.config.max_ship_speed))
        self.ship.x = float(np.clip(self.ship.x + self.ship.vx, self.ship.radius, self.config.width - self.ship.radius))
        self.ship.vx *= self.config.ship_drag

        fuel_cost = self.config.passive_fuel_cost
        if moving_left or moving_right:
            fuel_cost += self.config.movement_fuel_cost
        if firing and self.fire_cooldown == 0 and self.ship.fuel > fuel_cost + self.config.shot_fuel_cost:
            self.bullets.append(_Entity(self.ship.x, self.ship.y - self.ship.radius, 0.0, -self.config.bullet_speed, 3.0))
            self.fire_cooldown = self.config.fire_cooldown_steps
            self.shots += 1
            fuel_cost += self.config.shot_fuel_cost
            components["shot"] += self.reward_config.shot
        self.ship.fuel = max(0.0, self.ship.fuel - fuel_cost)

        if self.step_count % self.spawn_interval == 0:
            self._spawn_asteroid()
        self._update_entities()
        self._resolve_bullet_hits(components)
        self._resolve_ship_collisions(components)
        self._resolve_pickups(components)

        if self.ship.health <= 0.0 or self.ship.fuel <= 0.0:
            self.terminated = True
            self.death_reason = "health" if self.ship.health <= 0.0 else "fuel"
            components["death"] += self.reward_config.death
        elif self.step_count >= self.config.max_steps:
            self.truncated = True
            self.death_reason = "time_limit"
        else:
            components["survival"] += self.reward_config.survival

        reward = float(sum(components.values()))
        self.episode_return += reward
        observation = self._observation()
        info = self._info(components)
        if self.render_mode == "human":
            self.render()
        return observation, reward, self.terminated, self.truncated, info

    def _spawn_asteroid(self) -> None:
        speed = self.asteroid_speed
        vx = float(self.np_random.uniform(-0.5, 0.5) * speed)
        vy = float(np.sqrt(max(speed * speed - vx * vx, 0.0)))
        radius = float(self.np_random.choice((17.5, 20.0)))
        x = float(self.np_random.uniform(radius, self.config.width - radius))
        self.asteroids.append(_Entity(x, -radius, vx, vy, radius))

    def _update_entities(self) -> None:
        for entity in self.asteroids:
            entity.x += entity.vx
            entity.y += entity.vy
        for entity in self.bullets:
            entity.x += entity.vx
            entity.y += entity.vy
        for entity in self.pickups:
            entity.x += entity.vx
            entity.y += entity.vy
        self.asteroids = [
            item
            for item in self.asteroids
            if -60.0 <= item.x <= self.config.width + 60.0 and item.y <= self.config.height + 60.0
        ]
        self.bullets = [item for item in self.bullets if item.y >= -20.0]
        self.pickups = [item for item in self.pickups if item.y <= self.config.height + 30.0]

    @staticmethod
    def _collides(first: _Entity, second: _Entity) -> bool:
        return (first.x - second.x) ** 2 + (first.y - second.y) ** 2 <= (first.radius + second.radius) ** 2

    def _resolve_bullet_hits(self, components: Dict[str, float]) -> None:
        consumed_bullets: set[int] = set()
        consumed_asteroids: set[int] = set()
        for bullet_index, bullet in enumerate(self.bullets):
            for asteroid_index, asteroid in enumerate(self.asteroids):
                if asteroid_index in consumed_asteroids or not self._collides(bullet, asteroid):
                    continue
                consumed_bullets.add(bullet_index)
                consumed_asteroids.add(asteroid_index)
                self.hits += 1
                components["asteroid_destroyed"] += self.reward_config.asteroid_destroyed
                if self.np_random.random() < self.config.pickup_drop_probability:
                    kind = "fuel" if self.np_random.random() < 0.5 else "health"
                    self.pickups.append(
                        _Pickup(asteroid.x, asteroid.y, 0.0, self.config.pickup_speed, 10.0, kind)
                    )
                break
        self.bullets = [item for index, item in enumerate(self.bullets) if index not in consumed_bullets]
        self.asteroids = [item for index, item in enumerate(self.asteroids) if index not in consumed_asteroids]

    def _resolve_ship_collisions(self, components: Dict[str, float]) -> None:
        ship_entity = _Entity(self.ship.x, self.ship.y, self.ship.vx, 0.0, self.ship.radius)
        survivors: List[_Entity] = []
        for asteroid in self.asteroids:
            if self._collides(ship_entity, asteroid):
                self.collisions += 1
                self.ship.health = max(0.0, self.ship.health - max(5.0, 1.5 * self.asteroid_speed))
                components["collision"] += self.reward_config.collision
            else:
                survivors.append(asteroid)
        self.asteroids = survivors

    def _resolve_pickups(self, components: Dict[str, float]) -> None:
        ship_entity = _Entity(self.ship.x, self.ship.y, self.ship.vx, 0.0, self.ship.radius)
        survivors: List[_Pickup] = []
        for pickup in self.pickups:
            if not self._collides(ship_entity, pickup):
                survivors.append(pickup)
                continue
            if pickup.kind == "fuel":
                missing = 100.0 - self.ship.fuel
                restored = min(missing, self.config.fuel_pickup_amount)
                self.ship.fuel += restored
                fraction = restored / self.config.fuel_pickup_amount
            else:
                missing = 100.0 - self.ship.health
                restored = min(missing, self.config.health_pickup_amount)
                self.ship.health += restored
                fraction = restored / self.config.health_pickup_amount
            if restored > 0.0:
                self.pickups_collected += 1
                components["pickup"] += self.reward_config.pickup * fraction
        self.pickups = survivors

    def _observation(self) -> Observation:
        cfg = self.config
        global_features = np.asarray(
            [
                self.ship.x / cfg.width * 2.0 - 1.0,
                self.ship.vx / cfg.max_ship_speed,
                self.ship.health / 100.0,
                self.ship.fuel / 100.0,
                self.fire_cooldown / max(cfg.fire_cooldown_steps, 1),
                min(self.difficulty_level / 5.0, 1.0),
                min(self.step_count / cfg.max_steps, 1.0),
            ],
            dtype=np.float32,
        )

        asteroids = sorted(self.asteroids, key=self._threat_key)[: cfg.max_asteroids_observed]
        bullets = sorted(self.bullets, key=self._distance_key)[: cfg.max_bullets_observed]
        pickups = sorted(self.pickups, key=self._distance_key)[: cfg.max_pickups_observed]
        rows: List[np.ndarray] = []
        rows.extend(self._entity_row(item, 0) for item in asteroids)
        rows.extend(self._entity_row(item, 1) for item in bullets)
        rows.extend(self._entity_row(item, 2 if item.kind == "fuel" else 3) for item in pickups)

        entities = np.zeros((cfg.max_entities, self.ENTITY_FEATURES), dtype=np.float32)
        mask = np.zeros((cfg.max_entities,), dtype=np.float32)
        if rows:
            count = min(len(rows), cfg.max_entities)
            entities[:count] = np.asarray(rows[:count], dtype=np.float32)
            mask[:count] = 1.0
        return {"global": global_features, "entities": entities, "entity_mask": mask}

    def _entity_row(self, entity: _Entity, type_index: int) -> np.ndarray:
        cfg = self.config
        row = np.zeros((self.ENTITY_FEATURES,), dtype=np.float32)
        row[0] = np.clip((entity.x - self.ship.x) / cfg.width, -1.0, 1.0)
        row[1] = np.clip((entity.y - self.ship.y) / cfg.height, -1.0, 1.0)
        velocity_scale = max(cfg.bullet_speed, cfg.base_asteroid_speed + 5 * cfg.asteroid_speed_step)
        row[2] = np.clip(entity.vx / velocity_scale, -1.0, 1.0)
        row[3] = np.clip(entity.vy / velocity_scale, -1.0, 1.0)
        row[4] = np.clip(entity.radius / 20.0, 0.0, 1.0)
        row[5 + type_index] = 1.0
        return row

    def _distance_key(self, entity: _Entity) -> Tuple[float, float, float]:
        return (entity.x - self.ship.x) ** 2 + (entity.y - self.ship.y) ** 2, entity.y, entity.x

    def _threat_key(self, entity: _Entity) -> Tuple[float, float, float]:
        horizontal = abs(entity.x - self.ship.x)
        vertical = abs(entity.y - self.ship.y)
        return horizontal + 0.35 * vertical, vertical, entity.x

    def _info(self, reward_components: Dict[str, float]) -> Dict[str, Any]:
        return {
            "score": self.score,
            "hits": self.hits,
            "shots": self.shots,
            "accuracy": self.accuracy,
            "collisions": self.collisions,
            "pickups": self.pickups_collected,
            "survival_steps": self.step_count,
            "difficulty_level": self.difficulty_level,
            "death_reason": self.death_reason,
            "episode_return": self.episode_return,
            "reward_components": dict(reward_components),
        }

    def render(self) -> Optional[np.ndarray]:
        self._ensure_renderer()
        pygame = self._pygame
        surface = self._screen
        surface.fill((4, 7, 18))
        surface.blit(self._assets["ship"], self._assets["ship"].get_rect(center=(self.ship.x, self.ship.y)))
        for asteroid in self.asteroids:
            image = self._assets["asteroid"]
            surface.blit(image, image.get_rect(center=(asteroid.x, asteroid.y)))
        for bullet in self.bullets:
            pygame.draw.circle(surface, (255, 214, 64), (int(bullet.x), int(bullet.y)), 3)
        for pickup in self.pickups:
            color = (255, 210, 40) if pickup.kind == "fuel" else (60, 220, 90)
            pygame.draw.circle(surface, color, (int(pickup.x), int(pickup.y)), 10)
        pygame.draw.rect(surface, (100, 20, 20), (20, 15, 100, 12))
        pygame.draw.rect(surface, (30, 190, 70), (20, 15, int(self.ship.health), 12))
        pygame.draw.rect(surface, (35, 35, 35), (self.config.width - 120, 15, 100, 12))
        pygame.draw.rect(surface, (235, 190, 30), (self.config.width - 120, 15, int(self.ship.fuel), 12))
        label = self._font.render(
            f"Score {self.score}  Hits {self.hits}  Level {self.difficulty_level + 1}",
            True,
            (240, 240, 240),
        )
        surface.blit(label, (20, self.config.height - 35))
        if self.render_mode == "human":
            pygame.display.flip()
            pygame.event.pump()
            self._clock.tick(self.config.fps)
            return None
        pixels = pygame.surfarray.array3d(surface)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def _ensure_renderer(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        pygame.init()
        pygame.font.init()
        self._pygame = pygame
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("AsteroidsAI v2")
        else:
            self._screen = pygame.Surface((self.config.width, self.config.height))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 22)
        resource_dir = Path(__file__).resolve().parent.parent / "resources"
        try:
            ship = pygame.image.load(str(resource_dir / "ship.png"))
            ship = pygame.transform.rotate(pygame.transform.scale(ship, (40, 40)), -90)
            asteroid = pygame.image.load(str(resource_dir / "ast1.png"))
            asteroid = pygame.transform.scale(asteroid, (40, 40))
        except (FileNotFoundError, pygame.error):
            ship = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.polygon(ship, (100, 180, 255), ((20, 0), (3, 38), (37, 38)))
            asteroid = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(asteroid, (140, 140, 140), (20, 20), 18)
        self._assets = {"ship": ship, "asteroid": asteroid}

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._closed = True
        self._pygame = self._screen = self._clock = self._font = None
        self._assets = {}

    def config_dict(self) -> Dict[str, Any]:
        return {"environment": asdict(self.config), "reward": asdict(self.reward_config)}

    def rng_state(self) -> Dict[str, Any]:
        return dict(self.np_random.bit_generator.state)

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        self.np_random.bit_generator.state = state

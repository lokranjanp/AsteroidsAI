from __future__ import annotations

from .env import Action, AsteroidsEnv


def main() -> None:
    import pygame

    env = AsteroidsEnv(render_mode="human")
    env.reset()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()
            if not running:
                break
            keys = pygame.key.get_pressed()
            left = bool(keys[pygame.K_a] or keys[pygame.K_LEFT])
            right = bool(keys[pygame.K_d] or keys[pygame.K_RIGHT])
            fire = bool(keys[pygame.K_SPACE])
            if left and fire:
                action = Action.LEFT_FIRE
            elif right and fire:
                action = Action.RIGHT_FIRE
            elif left:
                action = Action.LEFT
            elif right:
                action = Action.RIGHT
            elif fire:
                action = Action.FIRE
            else:
                action = Action.IDLE
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()


import math
import pygame
import random
import time
from constants import *
import constants

all_sprites = pygame.sprite.Group()
bullets = pygame.sprite.Group()
pills = pygame.sprite.Group()


def getaction(act):
    if act == [1, 0, 0, 0]:
        return "move_left"
    if act == [0, 1, 0, 0]:
        return "move_right"
    if act == [0, 0, 1, 0]:
        return "fire"
    if act == [0, 0, 0, 1]:
        return "idle"


class Game_Score:
    def __init__(self):
        self.asteroids_hit = 0
        self.bullets_used = 0
        self.score = 0
        self.accuracy = 0

    def asteroid_hit(self):
        self.asteroids_hit += 1
        self.update_score()
        self.update_accuracy()

    def bullet_fired(self):
        self.bullets_used += 1
        self.update_score()
        self.update_accuracy()

    def update_score(self):
        self.score = (self.asteroids_hit * 100) - (self.bullets_used * 2)

    def update_accuracy(self):
        if self.bullets_used > 0:
            self.accuracy = self.asteroids_hit / self.bullets_used
        else:
            self.accuracy = 0

    def get_accuracy(self):
        self.update_accuracy()
        return round(self.accuracy, 2)

    def get_score(self):
        self.update_score()
        return int(self.score)


class Fuel_Pill(pygame.sprite.Sprite):
    def __init__(self, position, direction):
        super().__init__()
        self.image = pygame.image.load("resources/bolt_gold.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (20, 30))
        self.rect = self.image.get_rect(center=position)
        self.position = pygame.Vector2(position)
        self.direction = direction
        self.distance = 0
        self.type = "Fuel Pill"

    def update(self):
        self.position += self.direction * PILL_SPEED
        self.distance += PILL_SPEED
        self.rect.center = self.position
        if self.distance > PILL_RANGE:
            self.kill()


class Health_Pill(pygame.sprite.Sprite):
    def __init__(self, position, direction):
        super().__init__()
        self.image = pygame.image.load("resources/pill_green.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (20, 20))
        self.rect = self.image.get_rect(center=position)
        self.position = pygame.Vector2(position)
        self.direction = direction
        self.distance = 0
        self.type = "Health Pill"

    def update(self):
        self.position += self.direction * PILL_SPEED
        self.distance += PILL_SPEED
        self.rect.center = self.position
        if self.distance > PILL_RANGE:
            self.kill()


class Healthbar(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h, maxh, over, below):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.hp = maxh
        self.max = maxh
        self.over = over
        self.below = below

    def draw(self, screen):
        font_score = pygame.font.SysFont('Bauhaus 93', 30)
        pygame.draw.rect(screen, self.below, (self.x, self.y, self.w, self.h))
        pygame.draw.rect(screen, self.over, (self.x, self.y, self.hp, self.h))
        health_text = font_score.render('H', True, (255, 255, 255))
        screen.blit(health_text, (5, 10))

        fuel_text = font_score.render('F', True, (255, 255, 255))
        screen.blit(fuel_text, (screen.get_width() - 118, 10))


class Rocket(pygame.sprite.Sprite):
    def __init__(self, rocket_img, screen):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = rocket_img
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(screen.get_width() // 2, screen.get_height() - 100))
        self.angle = 90
        self.dx = 0.1
        self.position = pygame.Vector2(self.rect.center)
        self.x = pygame.Vector2(0, 2)
        self.deceleration = 0.95

    def update(self):
        global action
        if action == "move_left":
            print("moved left")
            self.x.x -= self.dx

        if action == "move_right":
            print("moved right")
            self.x.x += self.dx

        self.position += self.x
        self.x *= self.deceleration
        if self.position.x < 20:
            self.position.x = 20

        if self.position.x > constants.SCREEN_WIDTH - 20:
            self.position.x = constants.SCREEN_WIDTH - 20

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)

    def shoot(self):
        bullet_dir = pygame.Vector2(math.cos(math.radians(self.angle)), -math.sin(math.radians(self.angle)))
        bullet = Bullet(self.position, bullet_dir)
        all_sprites.add(bullet)
        bullets.add(bullet)


class Bullet(pygame.sprite.Sprite):
    def __init__(self, position, direction):
        super().__init__()
        self.image = pygame.image.load("resources/bullet.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (6, 6))
        self.rect = self.image.get_rect(center=position)
        self.position = pygame.Vector2(position)
        self.direction = direction
        self.distance = 0

    def update(self):
        self.position += self.direction * BULLET_SPEED
        self.distance += BULLET_SPEED
        self.rect.center = self.position
        if self.distance > BULLET_RANGE:
            self.kill()


class Asteroid(pygame.sprite.Sprite):
    def __init__(self, speed, screen, ast_imgs):
        super().__init__()
        self.image = random.choice(ast_imgs)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen.get_width())
        self.rect.y = -50
        self.direction = pygame.Vector2(random.uniform(-0.5, 0.5), 1).normalize()
        self.position = pygame.Vector2(self.rect.topleft)
        self.speed = pygame.Vector2(speed)

    def update(self):
        self.position.x += self.direction.x * self.speed.x
        self.position.y += self.direction.y * self.speed.y
        self.rect.center = self.position
        if self.rect.top > constants.SCREEN_HEIGHT + 20 or self.rect.left < -20 or self.rect.right > constants.SCREEN_WIDTH + 20:
            self.kill()

    def shoot(self):
        pill_dir = pygame.Vector2(math.cos(math.radians(90)), math.sin(math.radians(90)))
        pill_type = random.randint(1, 100) % 2
        if pill_type:
            pill = Fuel_Pill(self.position, pill_dir)
        else:
            pill = Health_Pill(self.position, pill_dir)
        all_sprites.add(pill)
        pills.add(pill)


class GameAI:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.start = 0
        self.end = 0
        self.death_reason = ""
        self.current_score = 0
        self.reward = 0
        self.end = 0
        self.player_accuracy = 0
        self.asteroids_hit = 0
        self.font_score = pygame.font.SysFont('Bauhaus 93', 30)
        self.screen = pygame.display.set_mode((600, 600))
        self.screen.fill(BLACK)
        self.clock = pygame.time.Clock()
        self.run = True
        self.actions_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.rocket_img = pygame.image.load("resources/ship.png").convert()
        self.rocket_img = pygame.transform.scale(self.rocket_img, (40, 40))
        self.rocket_img.set_colorkey(BLACK)
        self.rocket_img = pygame.transform.rotate(self.rocket_img, -90)

        self.ast_img1 = pygame.image.load("resources/ast1.png")
        self.ast_img1 = pygame.transform.scale(self.ast_img1, (40, 40))
        self.ast_img1.set_colorkey(BLACK)

        self.ast_img2 = pygame.image.load("resources/ast2.png")
        self.ast_img2 = pygame.transform.scale(self.ast_img2, (35, 35))
        self.ast_img2.set_colorkey(BLACK)

        self.ast_img3 = pygame.image.load("resources/ast3.png")
        self.ast_img3 = pygame.transform.scale(self.ast_img3, (40, 40))
        self.ast_img3.set_colorkey(BLACK)

        self.ast_img4 = pygame.image.load("resources/ast4.png")
        self.ast_img4 = pygame.transform.scale(self.ast_img4, (35, 35))
        self.ast_img4.set_colorkey(BLACK)

        self.pill_green = pygame.image.load("resources/bolt_gold.png")
        self.pill_green = pygame.transform.scale(self.pill_green, (20, 20))
        self.pill_green.set_colorkey(BLACK)
        self.ast_imgs = [self.ast_img1, self.ast_img2, self.ast_img3, self.ast_img4]

        self.reset_game()

    def reset_game(self):
        all_sprites.empty()
        self.ship = Rocket(self.rocket_img, self.screen)
        self.asteroids = pygame.sprite.Group()
        self.pills = pygame.sprite.Group()
        self.health = Healthbar(20, 10, 100, 15, 100, "green", "red")
        self.fuel = Healthbar(self.screen.get_width() - 105, 10, 100, 15, 100, "yellow", "black")
        self.game_score = Game_Score()
        all_sprites.add(self.ship)
        all_sprites.add(self.asteroids)
        all_sprites.add(self.pills)
        self.current_score = 0
        self.reward = 0
        self.start = time.time()

    def get_states(self):
        states_to_return = [self.ship.x.x, self.health.h, self.fuel.hp, self.game_score.accuracy, self.reward]
        asteroids_info = []
        for asteroid in self.asteroids:
            asteroids_info.extend([
                asteroid.rect.centerx,
                asteroid.rect.centery,
                asteroid.speed.x,
                asteroid.speed.y
            ])

        while len(asteroids_info) < 40:
            asteroids_info.extend([0, 0, 0, 0])

        states_to_return.extend(asteroids_info[:40])
        return states_to_return

    def spawn_asteroids(self):
        asteroid = Asteroid(constants.ASTEROID_SPEED, self.screen, self.ast_imgs)
        all_sprites.add(asteroid)
        self.asteroids.add(asteroid)
        all_sprites.update()

    def play_action(self, act):
        self.done = False
        self.current_score = self.game_score.get_score()
        self.player_accuracy = self.game_score.get_accuracy()
        self.asteroids_hit = self.game_score.asteroids_hit

        if self.health.hp <= 0:
            self.death_reason = "Spacecraft Health 0"
            self.ship.kill()
            self.end = time.time()
            self.reset_game()
            self.done = True

        elif self.fuel.hp <= 0:
            self.death_reason = "Spacecraft Fuel 0"
            self.ship.kill()
            self.end = time.time()
            self.reset_game()
            self.done = True

        self.clock.tick(constants.FPS)
        global action
        action = getaction(act)

        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()

        if action == 'fire':
            self.ship.shoot()
            self.game_score.bullet_fired()

        if action == 'move_left' or action == 'move_right':
            self.fuel.hp -= 0.1 * (constants.ASTEROID_SPEED/15)

        self.screen.fill(BLACK)
        all_sprites.update()
        all_sprites.draw(self.screen)
        self.fuel.draw(self.screen)
        self.health.draw(self.screen)

        for asteroid in self.asteroids:
            if self.ship.rect.colliderect(asteroid.rect):
                asteroid.kill()
                self.reward -= 15
                self.health.hp -= 1.5 * constants.ASTEROID_SPEED

        for bullet in bullets:
            for asteroid in self.asteroids:
                if asteroid.rect.colliderect(bullet.rect):
                    self.game_score.asteroid_hit()
                    bullet.kill()
                    if self.game_score.asteroids_hit % (random.randint(1, 100)) == 0:
                        asteroid.shoot()
                    asteroid.kill()
                    self.spawn_asteroids()
                    self.reward += 10
                    constants.ASTEROID_SPEED += 0.15
                    print(f"Accuracy : {self.game_score.accuracy}")

        if self.game_score.accuracy < 0.1:
            self.reward -= 100

        for pill in pills:
            if self.ship.rect.colliderect(pill.rect):
                self.reward += 10
                if pill.type == "Fuel Pill":
                    pill.kill()
                    self.fuel.hp += 10
                    if self.fuel.hp > self.fuel.max:
                        self.fuel.hp = self.fuel.max

                if pill.type == "Health Pill":
                    pill.kill()
                    self.health.hp += 20
                    if self.health.hp > self.health.max:
                        self.health.hp = self.health.max

        pygame.display.flip()

        return self.reward, self.done, self.current_score
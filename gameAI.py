import math
import pygame
import random
import time
import csv
from datetime import date, datetime
import os

from constants import *
import constants


global all_sprites
global bullets
global pills
global asteroids

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

    def bullet_fired(self):
        self.bullets_used += 1
        self.update_score()

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
        pygame.draw.rect(screen, self.below, (self.x, self.y, self.w, self.h))
        pygame.draw.rect(screen, self.over, (self.x, self.y, self.hp, self.h))


class Rocket(pygame.sprite.Sprite):
    def __init__(self, rocket_img, screen):
        super().__init__()
        self.original_image = rocket_img
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(screen.get_width() // 2, screen.get_height() - 100))
        self.angle = 90
        self.rotation_speed = 5
        self.dx = 0.1
        self.position = pygame.Vector2(self.rect.center)
        self.x = pygame.Vector2(0, 2)
        self.deceleration = 0.95

    def update(self, fuel, screen):
        global action
        if action == "move_left":
            self.x.x -= self.dx
            fuel.hp -= 0.1 * (ASTEROID_SPEED / 15)

        if action == "move_right":
            self.x.x += self.dx
            fuel.hp -= 0.1 * (ASTEROID_SPEED / 15)

        self.position += self.x
        self.x *= self.deceleration
        if self.position.x < 20:
            self.position.x = 20

        if self.position.x > screen.get_width() - 20:
            self.position.x = screen.get_width() - 20

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
    def __init__(self, speed, ast_imgs, screen):
        super().__init__()
        self.image = random.choice(ast_imgs)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen.get_width())
        self.rect.y = -50
        self.direction = pygame.Vector2(random.uniform(-0.5, 0.5), 1).normalize()
        self.position = pygame.Vector2(self.rect.topleft)
        self.speed = pygame.Vector2(speed)

    def update(self, screen):
        self.position.x += self.direction.x * (self.speed.x)
        self.position.y += self.direction.y * (self.speed.y)
        self.rect.center = self.position
        if self.rect.top > screen.get_height() + 20 or self.rect.left < -20 or self.rect.right > screen.get_width() + 20:
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


def spawn_asteroid():


class GameAI:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font_score = pygame.font.SysFont('Bauhaus 93', 30)
        self.screen = pygame.display.set_mode((600, 600))
        self.screen.fill(BLACK)
        self.clock = pygame.time.Clock()
        self.run = True
        self.actions_list = [[1,0,0,0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
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
        self.ship = Rocket(self.rocket_img, self.screen)

    def reset_game(self):
        self.firstgame = False
        self.asteroids = pygame.sprite.Group()
        self.pills = pygame.sprite.Group()
        self.health = Healthbar(20, 10, 100, 15, 100, "green", "red")
        self.fuel = Healthbar(self.screen.get_width() - 105, 10, 100, 15, 100, "yellow", "black")
        self.game_score = Game_Score()

        all_sprites.add(self.ship)
        all_sprites.add(self.asteroids)
        all_sprites.add(self.pills)

    def get_states(self, asteroids):
        states_to_return = []
        states_to_return.append(self.ship.x)
        states_to_return.append(self.health.hp)
        states_to_return.append(self.fuel.hp)

        if len(asteroids) >= 10:
            for i in range(0, 10):
                states_to_return.append(asteroids[i].position.x)
                states_to_return.append(asteroids[i].position.y)
                states_to_return.append(asteroids[i].speed.x)
                states_to_return.append(asteroids[i].speed.y)
        else :
            init_len = len(states_to_return)
            for asteroid in asteroids:
                states_to_return.append(asteroid.position.x)
                states_to_return.append(asteroid.position.y)
                states_to_return.append(asteroid.speed.x)
                states_to_return.append(asteroid.speed.y)

            pres_len = len(states_to_return)
            filler = 4 * (pres_len - init_len)

            for i in range(0, filler):
                states_to_return.append(0)

    def play_action(self, act):
        self.clock.tick(constants.FPS)
        global action
        action = getaction(act)

        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()

        if action == 'fire':
            self.ship.shoot()

        all_sprites.update()
        asteroid = Asteroid(constants.ASTEROID_SPEED)
        all_sprites.add(asteroid)
        asteroids.add(asteroid)

        for bullet in bullets:
            for asteroid in asteroids:
                if asteroid.rect.collidepoint(bullet.position.x, bullet.position.y):
                    self.game_score.asteroid_hit()
                    bullet.kill()
                    if self.game_score.asteroids_hit % (random.randint(1, 10)) == 0:
                        asteroid.shoot()
                    asteroid.kill()
                    constants.ASTEROID_SPEED += 0.15


# Create player

current_score = 0
end = 0
time_elap = 0
player_accuracy = 0
asteroids_hit = 0
death_reason = ""

frame_count = 0
start = time.time()
while run:
    speed_modif = ASTEROID_SPEED
    game_score.display_score(screen)
    if health.hp <= 0:
        asteroids_hit = game_score.asteroids_hit
        death_reason = "Spacecraft Health 0"
        player.kill()
        run = False
        end = time.time()
    elif fuel.hp <= 0:
        death_reason = "Spacecraft Fuel 0"
        player.kill()
        run = False
        end = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            asteroids_hit = game_score.asteroids_hit
            death_reason = "Player Quit"
            run = False
            end = time.time()

    for asteroid in asteroids:
        if player.rect.collidepoint(asteroid.position.x, asteroid.position.y):
            asteroid.kill()
            health.hp -= 1.5 * ASTEROID_SPEED

    for pill in pills:
        if player.rect.collidepoint(pill.position.x, pill.position.y):
            if pill.type == "Fuel Pill":
                pill.kill()
                fuel.hp += 10
                if fuel.hp > fuel.max:
                    fuel.hp = fuel.max

            if pill.type == "Health Pill":
                pill.kill()
                health.hp += 20
                if health.hp > health.max:
                    health.hp = health.max

    frame_count += 1
    if frame_count % SPAWN_RATE == 0:
        spawn_asteroid()

    current_score = game_score.get_score()
    player_accuracy = game_score.get_accuracy()
    asteroids_hit = game_score.asteroids_hit
    game_score.display_score(screen)
    all_sprites.draw(screen)
    health.draw(screen)
    pygame.display.flip()
    clock.tick(80)

end = time.time()
pygame.quit()
time_elap = (end - start)
with open(DATA_FILE, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        [game_date, game_time, round(time_elap, 2), death_reason, current_score, player_accuracy, asteroids_hit])

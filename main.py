import math
import pygame
import random
import time
import csv
from datetime import date, datetime
import os

pygame.init()
pygame.mixer.init()
pygame.font.init()
font_score = pygame.font.SysFont('Bauhaus 93', 30)
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
run = True
DATA_FILE = "game_data.csv"
game_date = date.today()
game_time = datetime.now().time()
game_time = game_time.strftime("%H:%M:%S")

file_exists = os.path.exists(DATA_FILE) == 1
if file_exists:
    file_empty = os.path.getsize(DATA_FILE) == 0
else :
    file_empty = True


if not file_exists:
    with open(DATA_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        if file_empty:
            writer.writerow(['Game Date', 'Game Time', 'Elapsed Time', 'Reason', 'Score', 'Accuracy', 'Asteroids Hit'])

with open(DATA_FILE, 'a', newline='') as file:
    writer = csv.writer(file)
    if file_empty:
        writer.writerow(['Game Date', 'Game Time', 'Elapsed Time', 'Reason', 'Score', 'Accuracy', 'Asteroids Hit'])

# Constants
BLACK = (0, 0, 0)
BULLET_SPEED = 5.5
BULLET_RANGE = 600
PILL_SPEED = 2.5
PILL_RANGE = 700
ASTEROID_SPEED = 2
SPAWN_RATE = 25

# Load images
rocket_img = pygame.image.load("resources/ship.png").convert()
rocket_img = pygame.transform.scale(rocket_img, (40, 40))
rocket_img.set_colorkey(BLACK)
rocket_img = pygame.transform.rotate(rocket_img, -90)
shoot_sound = pygame.mixer.Sound("resources/bf.wav")
explo_sound = pygame.mixer.Sound("resources/explosion.wav")

ast_img1 = pygame.image.load("resources/ast1.png")
ast_img1 = pygame.transform.scale(ast_img1, (40, 40))
ast_img1.set_colorkey(BLACK)

ast_img2 = pygame.image.load("resources/ast2.png")
ast_img2 = pygame.transform.scale(ast_img2, (35, 35))
ast_img2.set_colorkey(BLACK)

ast_img3 = pygame.image.load("resources/ast3.png")
ast_img3 = pygame.transform.scale(ast_img3, (40, 40))
ast_img3.set_colorkey(BLACK)

ast_img4 = pygame.image.load("resources/ast4.png")
ast_img4 = pygame.transform.scale(ast_img4, (35, 35))
ast_img4.set_colorkey(BLACK)

pill_green=pygame.image.load("resources/bolt_gold.png")
pill_green=pygame.transform.scale(pill_green,(20,20))
pill_green.set_colorkey(BLACK)

ast_imgs = [ast_img1, ast_img2, ast_img3, ast_img4]

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

    def display_score(self, screen):
        self.update_accuracy()
        score_text = font_score.render(f'Score: {int(self.score)} Accuracy: {self.accuracy:.2f}', True, (255, 255, 255))
        text_rect = score_text.get_rect()
        screen.blit(score_text, (screen.get_width() - text_rect.width - 10, screen.get_height() - text_rect.height - 10))

        health_text = font_score.render('H', True, (255, 255, 255))
        health_text_rect = health_text.get_rect()
        screen.blit(health_text, (5, 10))

        fuel_text = font_score.render('F',True,(255,255,255))
        fuel_text_rect = fuel_text.get_rect()
        screen.blit(fuel_text,(screen.get_width()-118,10))


class Fuel_Pill(pygame.sprite.Sprite):
    def __init__(self, position, direction):
        super().__init__()
        self.image = pygame.image.load("resources/bolt_gold.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (15, 25))
        self.rect = self.image.get_rect(center=position)
        self.position = pygame.Vector2(position)
        self.direction = direction
        self.distance = 0

    def update(self):
        self.position += self.direction * PILL_SPEED
        self.distance += PILL_SPEED
        self.rect.center = self.position
        if self.distance > PILL_RANGE:
            self.kill()


class Healthbar():
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
    def __init__(self):
        super().__init__()
        self.original_image = rocket_img
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(screen.get_width()//2, screen.get_height()-100))
        self.angle = 90
        self.rotation_speed = 5
        self.dx = 0.1
        self.position = pygame.Vector2(self.rect.center)
        self.x = pygame.Vector2(0, 2)
        self.deceleration = 0.95

    def update(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.x.x -= self.dx
            fuel.hp -= 0.1 * (ASTEROID_SPEED/10)

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.x.x += self.dx
            fuel.hp -= 0.1 * (ASTEROID_SPEED/10)

        self.position += self.x
        self.x *= self.deceleration
        if self.position.x < 20:
            self.position.x = 20

        if self.position.x > screen.get_width()-20 :
            self.position.x = screen.get_width()-20

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
    def __init__(self, speed):
        super().__init__()
        self.image = random.choice(ast_imgs)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen.get_width())
        self.rect.y = -50
        self.direction = pygame.Vector2(random.uniform(-0.5, 0.5), 1).normalize()
        self.position = pygame.Vector2(self.rect.topleft)
        self.speed = speed

    def update(self):
        self.position += self.direction * (self.speed)
        self.rect.center = self.position
        if self.rect.top > screen.get_height() + 20 or self.rect.left < -20 or self.rect.right > screen.get_width() + 20:
            self.kill()

    def shoot(self):
        pill_dir = pygame.Vector2(math.cos(math.radians(90)),math.sin(math.radians(90)))
        pill = Fuel_Pill(self.position, pill_dir)
        all_sprites.add(pill)
        pills.add(pill)

def spawn_asteroid():
    asteroid = Asteroid(ASTEROID_SPEED)
    all_sprites.add(asteroid)
    asteroids.add(asteroid)

# Sprite groups
all_sprites = pygame.sprite.Group()
bullets = pygame.sprite.Group()
asteroids = pygame.sprite.Group()
pills = pygame.sprite.Group()

# Create player
player = Rocket()
all_sprites.add(player)
health = Healthbar(20, 10, 100, 15, 100, "green","red")
fuel = Healthbar(screen.get_width()-105, 10, 100, 15, 100, "yellow", "black")
game_score = Game_Score()

current_score = 0
end = 0
time_elap = 0
player_accuracy = 0
asteroids_hit = 0
death_reason = ""

frame_count = 0
start = time.time()
while run:
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

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_score.bullet_fired()
                player.shoot()
                shoot_sound.play()

    screen.fill('black')
    all_sprites.update()
    fuel.draw(screen)

    for bullet in bullets:
        for asteroid in asteroids:
            if asteroid.rect.collidepoint(bullet.position.x, bullet.position.y):
                game_score.asteroid_hit()
                explo_sound.play()
                bullet.kill()
                mid = time.time()
                if game_score.asteroids_hit%5 == 0 :
                    asteroid.shoot()
                asteroid.kill()
                ASTEROID_SPEED += 0.15

    for asteroid in asteroids :
        if player.rect.collidepoint(asteroid.position.x,asteroid.position.y):
            explo_sound.play()
            asteroid.kill()
            health.hp -= 15

    for pill in pills :
        if player.rect.collidepoint(pill.position.x,pill.position.y):
            pill.kill()
            fuel.hp += 10
            if fuel.hp > fuel.max :
                fuel.hp = fuel.max

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
    writer.writerow([game_date, game_time, round(time_elap, 2), death_reason, current_score, player_accuracy, asteroids_hit])

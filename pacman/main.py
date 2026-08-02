from game import Game
import pygame

if __name__ == '__main__':
    pygame.init()
    
    game = Game(render=True)
    running = True
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:    game.step(0)
        if keys[pygame.K_DOWN]:  game.step(1)
        if keys[pygame.K_LEFT]:  game.step(2)
        if keys[pygame.K_RIGHT]: game.step(3)

    # Display score at the end
    font = pygame.font.Font(None, 36)
    text = font.render(f"Final Score: {game.player.score}", True, (255, 255, 255))
    text_rect = text.get_rect(center=(game.screen.get_width() // 2, game.screen.get_height() // 2))
    game.screen.blit(text, text_rect)
    pygame.display.flip()

    pygame.time.delay(3000)  # Keep the score displayed for 3 seconds
    pygame.quit()
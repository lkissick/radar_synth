import pygame
import math
import sys
import numpy as np

# ─────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()

WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Radar Sequencer")
clock = pygame.time.Clock()

SAMPLE_RATE = 44100

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
BG_COLOR        = (8, 10, 14)
CIRCLE_COLOR    = (30, 60, 40)
SWEEP_COLOR     = (0, 255, 80)
GRID_COLOR      = (20, 45, 28)
FONT_COLOR      = (100, 180, 120)
UI_DIM          = (40, 80, 50)
UI_ACTIVE       = (0, 255, 80)
UI_INACTIVE     = (25, 55, 35)

CENTER          = (WIDTH // 2, HEIGHT // 2 + 30)
RADIUS          = 250

TRAIL_DEGREES   = 60
TRAIL_STEPS     = 60

MIN_BPM         = 10
MAX_BPM         = 200

SLIDER_LEFT     = 180
SLIDER_RIGHT    = 620
SLIDER_Y        = HEIGHT - 80
SLIDER_W        = SLIDER_RIGHT - SLIDER_LEFT
KNOB_R          = 10

TOKEN_RADIUS    = 10

# Grid subdivisions available
GRID_OPTIONS    = [2, 4, 8, 16, 32]   # beats per loop (subdivisions of the circle)

# ─────────────────────────────────────────
#  MUSIC THEORY
# ─────────────────────────────────────────
SCALE_INTERVALS = {
    "Major":      [0, 2, 4, 5, 7, 9, 11],
    "Minor":      [0, 2, 3, 5, 7, 8, 10],
    "Dorian":     [0, 2, 3, 5, 7, 9, 10],
    "Pentatonic": [0, 2, 4, 7, 9],
    "Blues":      [0, 3, 5, 6, 7, 10],
}

ROOT_NOTES     = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
ROOT_SEMITONES = {"A": 0, "A#": 1, "B": 2, "C": 3, "C#": 4, "D": 5,
                  "D#": 6, "E": 7, "F": 8, "F#": 9, "G": 10, "G#": 11}
A2_FREQ        = 110.0

def build_scale_freqs(root, scale_name, octaves=4):
    root_offset = ROOT_SEMITONES[root]
    intervals   = SCALE_INTERVALS[scale_name]
    freqs = []
    for octave in range(octaves):
        for interval in intervals:
            semitones = root_offset + interval + octave * 12
            freqs.append(A2_FREQ * (2 ** (semitones / 12)))
    return sorted(freqs)

def radius_to_snapped_freq(r, scale_freqs):
    t       = r / RADIUS
    min_f   = scale_freqs[0]
    max_f   = scale_freqs[-1]
    log_min = math.log(min_f)
    log_max = math.log(max_f)
    target_f = math.exp(log_min + t * (log_max - log_min))
    return min(scale_freqs, key=lambda f: abs(f - target_f))

def freq_to_note_name(freq):
    semitones  = 12 * math.log2(freq / A2_FREQ)
    note_index = round(semitones) % 12
    octave     = 2 + round(semitones) // 12
    return f"{ROOT_NOTES[note_index]}{octave}"

# ─────────────────────────────────────────
#  QUANTIZATION
# ─────────────────────────────────────────
def quantize_angle(a, subdivisions):
    """Snap angle to the nearest grid line."""
    step = (2 * math.pi) / subdivisions
    return round(a / step) * step

# ─────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────
def make_sine_sound(freq, duration=0.25, volume=0.4):
    n_samples = int(SAMPLE_RATE * duration)
    t         = np.linspace(0, duration, n_samples, endpoint=False)
    wave      = np.sin(2 * np.pi * freq * t)
    wave     += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    wave     /= 1.3

    attack_samples = int(SAMPLE_RATE * 0.008)
    envelope = np.ones(n_samples)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:] = np.exp(-5 * np.linspace(0, 1, n_samples - attack_samples))

    wave   = (wave * envelope * volume * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
angle          = -math.pi / 2
bpm            = 60.0
knob_x         = SLIDER_LEFT + (bpm - MIN_BPM) / (MAX_BPM - MIN_BPM) * SLIDER_W
dragging_knob  = False

selected_root  = "C"
selected_scale = "Major"
scale_freqs    = build_scale_freqs(selected_root, selected_scale)

snap_enabled   = True
grid_subdiv    = 16        # default: 16th-note grid

tokens         = []
dragging_token = None
token_history  = []
last_click_time = 0
last_click_pos  = None
DOUBLE_CLICK_MS = 400   # max ms between clicks to count as double-click

# ─────────────────────────────────────────
#  TOKEN HELPERS
# ─────────────────────────────────────────
def angle_to_point(a, r=RADIUS, cx=CENTER[0], cy=CENTER[1]):
    return (cx + r * math.cos(a), cy + r * math.sin(a))

def point_to_polar(px, py):
    dx = px - CENTER[0]
    dy = py - CENTER[1]
    return math.atan2(dy, dx), math.hypot(dx, dy)

def make_token(px, py):
    a, r = point_to_polar(px, py)
    r    = min(r, RADIUS)
    if snap_enabled:
        a = quantize_angle(a, grid_subdiv)
        # recompute screen position from snapped angle
        px = int(CENTER[0] + r * math.cos(a))
        py = int(CENTER[1] + r * math.sin(a))
    freq  = radius_to_snapped_freq(r, scale_freqs)
    sound = make_sine_sound(freq)
    return {"px": px, "py": py, "token_angle": a, "r": r, "freq": freq, "sound": sound, "muted": False, "last_triggered": 0}

def rebuild_all_tokens():
    for i, tok in enumerate(tokens):
        freq  = radius_to_snapped_freq(tok["r"], scale_freqs)
        sound = make_sine_sound(freq)
        tokens[i]["freq"]  = freq
        tokens[i]["sound"] = sound

def inside_circle(px, py):
    return math.hypot(px - CENTER[0], py - CENTER[1]) <= RADIUS

def token_at(px, py):
    for i, tok in enumerate(tokens):
        if math.hypot(px - tok["px"], py - tok["py"]) <= TOKEN_RADIUS + 4:
            return i
    return None

def crossed(prev, curr, target):
    diff   = (curr   - prev)   % (2 * math.pi)
    offset = (target - prev)   % (2 * math.pi)
    return 0 < offset <= diff

# ─────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────
PILL_H     = 22
PILL_PAD_X = 8
PILL_GAP   = 5
SELECTOR_Y = 90

def get_font():
    return pygame.font.SysFont("monospace", 12)

def draw_pill_row(surface, items, selected, y, label):
    font       = get_font()
    label_surf = font.render(label, True, (50, 100, 65))
    surface.blit(label_surf, (20, y + 4))
    x     = 100
    rects = []
    for item in items:
        text = str(item)
        w    = font.size(text)[0] + PILL_PAD_X * 2
        rect = pygame.Rect(x, y, w, PILL_H)
        color = UI_ACTIVE if item == selected else UI_INACTIVE
        pygame.draw.rect(surface, color, rect, border_radius=4)
        if item == selected:
            pygame.draw.rect(surface, (0, 180, 60), rect, 1, border_radius=4)
        text_color = BG_COLOR if item == selected else (60, 120, 75)
        surf = font.render(text, True, text_color)
        surface.blit(surf, (x + PILL_PAD_X, y + (PILL_H - surf.get_height()) // 2))
        rects.append((rect, item))
        x += w + PILL_GAP
    return rects

def draw_toggle(surface, label, value, x, y):
    """Draw a small ON/OFF toggle pill. Returns the rect for hit testing."""
    font  = get_font()
    lsurf = font.render(label, True, (50, 100, 65))
    surface.blit(lsurf, (x, y + 4))
    tx    = x + lsurf.get_width() + 8
    text  = "ON" if value else "OFF"
    w     = font.size(text)[0] + PILL_PAD_X * 2
    rect  = pygame.Rect(tx, y, w, PILL_H)
    color = UI_ACTIVE if value else (60, 30, 30)
    pygame.draw.rect(surface, color, rect, border_radius=4)
    pygame.draw.rect(surface, (0, 180, 60) if value else (120, 40, 40), rect, 1, border_radius=4)
    tsurf = font.render(text, True, BG_COLOR if value else (200, 80, 80))
    surface.blit(tsurf, (tx + PILL_PAD_X, y + (PILL_H - tsurf.get_height()) // 2))
    return rect

# ─────────────────────────────────────────
#  DRAW
# ─────────────────────────────────────────
def draw_grid(surface, subdivisions):
    """Draw angular grid lines around the circle."""
    step = (2 * math.pi) / subdivisions
    for i in range(subdivisions):
        a         = i * step
        is_beat   = (i % (subdivisions // min(4, subdivisions))) == 0
        color     = (35, 75, 45) if is_beat else (20, 42, 27)
        thickness = 1
        inner_r   = RADIUS - (18 if is_beat else 10)
        outer_r   = RADIUS + (8 if is_beat else 4)
        # tick on rim
        inner = angle_to_point(a, inner_r)
        outer = angle_to_point(a, outer_r)
        pygame.draw.line(surface, color,
                         (int(inner[0]), int(inner[1])), (int(outer[0]), int(outer[1])), thickness)
        # faint spoke inside circle
        spoke_end = angle_to_point(a, RADIUS - 2)
        pygame.draw.line(surface, (15, 35, 20), CENTER, (int(spoke_end[0]), int(spoke_end[1])), 1)

def draw_radar_trail(surface, current_angle):
    trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for i in range(TRAIL_STEPS):
        t           = i / TRAIL_STEPS
        trail_angle = current_angle - math.radians(TRAIL_DEGREES * (1 - t))
        alpha       = int(180 * t * t)
        green       = int(220 * t)
        end         = angle_to_point(trail_angle)
        pygame.draw.line(trail_surf, (0, green, 60, alpha),
                         CENTER, (int(end[0]), int(end[1])), 2)
    surface.blit(trail_surf, (0, 0))

def draw_sweep_line(surface, current_angle):
    end = angle_to_point(current_angle)
    pygame.draw.line(surface, (0, 120, 50),    CENTER, (int(end[0]), int(end[1])), 4)
    pygame.draw.line(surface, (180, 255, 180), CENTER, (int(end[0]), int(end[1])), 1)

def draw_circle_base(surface):
    for r_frac in [0.33, 0.66, 1.0]:
        pygame.draw.circle(surface, GRID_COLOR, CENTER, int(RADIUS * r_frac), 1)
    pygame.draw.circle(surface, SWEEP_COLOR, CENTER, RADIUS, 1)
    pygame.draw.circle(surface, SWEEP_COLOR, CENTER, 4)

def draw_tokens(surface, tokens):
    font      = get_font()
    now       = pygame.time.get_ticks()
    GLOW_MS   = 600   # how long the glow fades over in milliseconds

    for tok in tokens:
        px, py  = int(tok["px"]), int(tok["py"])
        muted   = tok.get("muted", False)
        elapsed = now - tok.get("last_triggered", 0)
        t_glow  = max(0.0, 1.0 - elapsed / GLOW_MS)   # 1.0 = just triggered, 0.0 = faded

        if muted:
            rim_col = (40, 40, 40)
            dot_col = (60, 60, 60)
            txt_col = (80, 80, 80)
            pygame.draw.circle(surface, rim_col, (px, py), TOKEN_RADIUS + 3, 1)
            pygame.draw.circle(surface, dot_col, (px, py), TOKEN_RADIUS)
            pygame.draw.circle(surface, BG_COLOR, (px, py), TOKEN_RADIUS - 4)
        else:
            # Outer glow ring — expands and fades
            if t_glow > 0:
                glow_r   = int(TOKEN_RADIUS + 3 + t_glow * 10)
                glow_a   = int(t_glow * 180)
                glow_col = (0, int(180 + t_glow * 75), 60, glow_a)
                glow_surf = pygame.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, glow_col,
                                   (glow_r + 2, glow_r + 2), glow_r, 2)
                surface.blit(glow_surf, (px - glow_r - 2, py - glow_r - 2))

            # Rim — bright green flash fading to dim
            rim_g   = int(60 + t_glow * 120)
            rim_col = (0, rim_g, 40)
            pygame.draw.circle(surface, rim_col, (px, py), TOKEN_RADIUS + 3, 1)

            # Fill — full green flash fading to dark
            fill_g   = int(40 + t_glow * 215)
            fill_col = (0, fill_g, int(30 + t_glow * 70))
            pygame.draw.circle(surface, fill_col, (px, py), TOKEN_RADIUS)
            pygame.draw.circle(surface, BG_COLOR,  (px, py), TOKEN_RADIUS - 4)

        note_name = freq_to_note_name(tok["freq"])
        txt_col   = (80, 80, 80) if muted else (0, int(100 + t_glow * 100), 60)
        label     = font.render(note_name, True, txt_col)
        surface.blit(label, (px - label.get_width() // 2, py - TOKEN_RADIUS - 14))

def draw_slider(surface, knob_x, bpm):
    font = get_font()
    pygame.draw.line(surface, UI_DIM,      (SLIDER_LEFT, SLIDER_Y), (SLIDER_RIGHT, SLIDER_Y), 2)
    pygame.draw.line(surface, SWEEP_COLOR, (SLIDER_LEFT, SLIDER_Y), (int(knob_x), SLIDER_Y), 2)
    pygame.draw.circle(surface, SWEEP_COLOR, (int(knob_x), SLIDER_Y), KNOB_R)
    pygame.draw.circle(surface, BG_COLOR,    (int(knob_x), SLIDER_Y), KNOB_R - 3)
    label = font.render(f"TEMPO  {int(bpm)} BPM", True, FONT_COLOR)
    surface.blit(label, (CENTER[0] - label.get_width() // 2, SLIDER_Y + 18))
    surface.blit(font.render(f"{MIN_BPM}", True, (50, 90, 60)), (SLIDER_LEFT - 20, SLIDER_Y - 8))
    surface.blit(font.render(f"{MAX_BPM}", True, (50, 90, 60)), (SLIDER_RIGHT + 6,  SLIDER_Y - 8))

def draw_title(surface):
    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)
    title = font_big.render("RADAR SEQUENCER", True, SWEEP_COLOR)
    sub   = font_small.render(
        "left-click: place token   right-click: remove   drag: move", True, (50, 100, 65))
    surface.blit(title, (CENTER[0] - title.get_width() // 2, 18))
    surface.blit(sub,   (CENTER[0] - sub.get_width()   // 2, 42))

# ─────────────────────────────────────────
#  GAME LOOP
# ─────────────────────────────────────────
prev_angle   = angle
root_rects   = []
scale_rects  = []
grid_rects   = []
snap_rect    = None

while True:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL):
                if token_history:
                    tokens[:] = token_history.pop()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            # Snap toggle
            if snap_rect and snap_rect.collidepoint(mx, my):
                snap_enabled = not snap_enabled

            # Grid subdivision pills
            for rect, item in grid_rects:
                if rect.collidepoint(mx, my):
                    grid_subdiv = item

            # Root note pills
            for rect, item in root_rects:
                if rect.collidepoint(mx, my):
                    selected_root = item
                    scale_freqs   = build_scale_freqs(selected_root, selected_scale)
                    rebuild_all_tokens()

            # Scale pills
            for rect, item in scale_rects:
                if rect.collidepoint(mx, my):
                    selected_scale = item
                    scale_freqs    = build_scale_freqs(selected_root, selected_scale)
                    rebuild_all_tokens()

            # Slider knob
            if math.hypot(mx - knob_x, my - SLIDER_Y) <= KNOB_R + 4:
                dragging_knob = True

            elif event.button == 1:
                hit = token_at(mx, my)
                now = pygame.time.get_ticks()
                is_double = (
                    hit is not None and
                    last_click_pos == hit and
                    now - last_click_time < DOUBLE_CLICK_MS
                )
                if is_double:
                    tokens[hit]["muted"] = not tokens[hit]["muted"]
                    last_click_time = 0   # reset so triple-click doesn't re-toggle
                elif hit is not None:
                    dragging_token  = hit
                    last_click_time = now
                    last_click_pos  = hit
                elif inside_circle(mx, my):
                    token_history.append([t.copy() for t in tokens])
                    tokens.append(make_token(mx, my))
                    last_click_time = 0

            elif event.button == 3:
                hit = token_at(mx, my)
                if hit is not None:
                    token_history.append([t.copy() for t in tokens])
                    tokens.pop(hit)

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_knob  = False
            dragging_token = None

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if dragging_knob:
                knob_x = max(SLIDER_LEFT, min(SLIDER_RIGHT, mx))
                bpm    = MIN_BPM + (knob_x - SLIDER_LEFT) / SLIDER_W * (MAX_BPM - MIN_BPM)
            elif dragging_token is not None:
                dx, dy = mx - CENTER[0], my - CENTER[1]
                dist   = math.hypot(dx, dy)
                if dist > RADIUS:
                    mx = int(CENTER[0] + dx * RADIUS / dist)
                    my = int(CENTER[1] + dy * RADIUS / dist)
                tokens[dragging_token] = make_token(mx, my)

    # ── Update ───────────────────────────
    prev_angle = angle
    angle     += 2 * math.pi * (bpm / 60.0 / 4.0) * dt
    if angle > math.pi:
        angle -= 2 * math.pi

    # ── Trigger ──────────────────────────
    for tok in tokens:
        if crossed(prev_angle, angle, tok["token_angle"]):
            tok["last_triggered"] = pygame.time.get_ticks()
            if not tok.get("muted"):
                tok["sound"].play()

    # ── Draw ─────────────────────────────
    screen.fill(BG_COLOR)
    draw_title(screen)

    # Pill rows
    root_rects  = draw_pill_row(screen, ROOT_NOTES,
                                selected_root,  SELECTOR_Y,      "KEY  ")
    scale_rects = draw_pill_row(screen, list(SCALE_INTERVALS.keys()),
                                selected_scale, SELECTOR_Y + 30, "SCALE")
    grid_rects  = draw_pill_row(screen, GRID_OPTIONS,
                                grid_subdiv,    SELECTOR_Y + 60, "GRID ")
    snap_rect   = draw_toggle(screen, "SNAP ", snap_enabled,
                              20, SELECTOR_Y + 90)

    draw_grid(screen, grid_subdiv)
    draw_circle_base(screen)
    draw_tokens(screen, tokens)
    draw_radar_trail(screen, angle)
    draw_sweep_line(screen, angle)
    draw_slider(screen, knob_x, bpm)

    pygame.display.flip()
"""
Pygame Battery Trading Visualizer — Gymnasium-style render for SAC evaluation.

I display the battery trading game in real-time:
- Battery gauge (SoC bar that fills/empties)
- Scrolling price chart with buy/sell zones
- Agent action indicator (BUY/SELL/IDLE)
- Running P&L counter per market

Controls:
  SPACE  = pause/resume
  LEFT   = step back (when paused)
  RIGHT  = step forward (when paused)
  +/-    = speed up/slow down
  R      = restart
  Q/ESC  = quit
"""

import pygame
import numpy as np
from typing import Dict, List, Optional


# I define colors
BLACK = (15, 15, 25)
WHITE = (240, 240, 240)
GRAY = (80, 80, 100)
DARK_GRAY = (40, 40, 55)
LIGHT_GRAY = (120, 120, 140)

GREEN = (0, 200, 100)
GREEN_DARK = (0, 120, 60)
RED = (220, 60, 60)
RED_DARK = (140, 30, 30)
BLUE = (60, 140, 255)
YELLOW = (255, 200, 40)
ORANGE = (255, 140, 40)
CYAN = (40, 220, 220)
PURPLE = (160, 80, 240)

BG_PANEL = (25, 25, 40)
BG_CHART = (20, 20, 35)


class BatteryRenderer:
    """I render the battery trading environment as a pygame game display."""

    WIDTH = 1200
    HEIGHT = 750

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("EntsoDRL Battery Trading")
        self.clock = pygame.time.Clock()

        # I load fonts
        self.font_large = pygame.font.SysFont('Consolas', 28, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 18)
        self.font_small = pygame.font.SysFont('Consolas', 14)
        self.font_tiny = pygame.font.SysFont('Consolas', 11)

        # I keep history for charts
        self.price_history: List[float] = []
        self.forecast_history: List[float] = []  # I store DAM forecast prices
        self.soc_history: List[float] = []
        self.profit_history: List[float] = []
        self.action_history: List[float] = []  # net_mw
        self.max_history = 192  # 2 days of 15-min data

        # I track gate closure events for notifications
        self._gate_events: List[Dict] = []  # {text, color, ttl}
        self._prev_ida1_locked = False
        self._prev_ida2_locked = False

        # I keep a history of info dicts for step-back
        self._info_history: List[Dict] = []
        self._max_info_history = 672  # full episode
        self._replay_idx: Optional[int] = None  # None = live, int = replaying

        # I track state
        self.paused = False
        self.speed = 4  # steps per second
        self.step_count = 0
        self.running = True
        self.step_forward = False
        self.step_backward = False
        self._current_info: Optional[Dict] = None  # I cache last info for re-drawing

    def handle_events(self) -> bool:
        """I handle keyboard input. Returns False if user wants to quit."""
        self.step_forward = False
        self.step_backward = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    if not self.paused:
                        self._replay_idx = None  # I exit replay mode on resume
                elif event.key == pygame.K_RIGHT:
                    if self.paused:
                        self.step_forward = True
                elif event.key == pygame.K_LEFT:
                    if self.paused:
                        self.step_backward = True
                elif event.key == pygame.K_r:
                    self.price_history.clear()
                    self.forecast_history.clear()
                    self.soc_history.clear()
                    self.profit_history.clear()
                    self.action_history.clear()
                    self._info_history.clear()
                    self._replay_idx = None
                    self.step_count = 0

                # I handle speed with unicode to work on all keyboard layouts
                ch = event.unicode
                if ch == '+' or ch == '=':
                    self.speed = min(96, self.speed * 2)
                elif ch == '-' or ch == '_':
                    self.speed = max(1, self.speed // 2)

        return True

    def should_step(self) -> bool:
        """I return True if the main loop should call env.step().

        When paused:
          - LEFT arrow replays backward through history (no env.step needed)
          - RIGHT arrow advances one step (env.step needed)
          - Otherwise I just re-draw the current frame
        """
        if self.paused:
            if self.step_backward and self._info_history:
                # I go back in history
                if self._replay_idx is None:
                    self._replay_idx = len(self._info_history) - 2
                else:
                    self._replay_idx = max(0, self._replay_idx - 1)
                self._draw_frame(self._info_history[self._replay_idx])
            elif self.step_forward:
                # I advance — if replaying, move forward in history first
                if self._replay_idx is not None:
                    if self._replay_idx < len(self._info_history) - 1:
                        self._replay_idx += 1
                        self._draw_frame(self._info_history[self._replay_idx])
                        return False  # I drew from history, no env.step needed
                    else:
                        self._replay_idx = None  # I caught up, need new step
                return True  # I need env.step for fresh data
            else:
                # I re-draw current frame while paused
                # If I am in replay mode, I use the replay position, not live
                if self._replay_idx is not None and self._info_history:
                    self._draw_frame(self._info_history[self._replay_idx])
                elif self._current_info is not None:
                    self._draw_frame(self._current_info)
            self.clock.tick(30)
            return False
        else:
            self.clock.tick(self.speed)
            return True

    def render(self, info: Dict):
        """I update state with new step data and draw the frame."""
        self.step_count += 1
        self._current_info = info

        # I store in history for replay
        self._info_history.append(info)
        if len(self._info_history) > self._max_info_history:
            self._info_history.pop(0)

        # I update chart histories
        self.price_history.append(info.get('dam_price', 100))
        self.forecast_history.append(info.get('predicted_price', info.get('dam_price', 100)))
        self.soc_history.append(info.get('soc', 0.5))
        self.profit_history.append(info.get('total_profit', 0))
        self.action_history.append(info.get('net_mw', 0))

        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        if len(self.forecast_history) > self.max_history:
            self.forecast_history.pop(0)
            self.soc_history.pop(0)
            self.profit_history.pop(0)
            self.action_history.pop(0)

        # I detect IDA gate closure events
        ida1_locked = info.get('ida1_locked', False)
        ida2_locked = info.get('ida2_locked', False)
        if ida1_locked and not self._prev_ida1_locked:
            self._gate_events.append({'text': 'IDA1 GATE CLOSED (15:00)', 'color': ORANGE, 'ttl': 30})
        if ida2_locked and not self._prev_ida2_locked:
            self._gate_events.append({'text': 'IDA2 GATE CLOSED (22:00)', 'color': ORANGE, 'ttl': 30})
        self._prev_ida1_locked = ida1_locked
        self._prev_ida2_locked = ida2_locked

        self._draw_frame(info)

    def _draw_frame(self, info: Dict):
        """I draw a single frame from an info dict (used for both live and replay)."""
        soc = info.get('soc', 0.5)
        net_mw = info.get('net_mw', 0)
        timestamp = info.get('timestamp', '')

        # I compute the display step number (replay-aware)
        if self._replay_idx is not None:
            display_step = self._replay_idx + 1
        else:
            display_step = self.step_count

        self.screen.fill(BLACK)
        self._draw_header(info, timestamp, display_step)
        self._draw_battery(soc, net_mw)
        self._draw_price_chart(display_step)
        self._draw_action_panel(info)
        self._draw_pnl_panel(info, display_step)
        self._draw_gate_events()
        self._draw_controls()
        pygame.display.flip()

    def _draw_header(self, info: Dict, timestamp: str, display_step: int = 0):
        """I draw the top header bar."""
        pygame.draw.rect(self.screen, BG_PANEL, (0, 0, self.WIDTH, 45))
        pygame.draw.line(self.screen, GRAY, (0, 45), (self.WIDTH, 45))

        # I format timestamp
        ts_text = str(timestamp)[:16] if timestamp else f"Step {display_step}"
        day = (display_step - 1) // 96 + 1
        step_in_day = (display_step - 1) % 96

        left = self.font_medium.render(f"  {ts_text}", True, WHITE)
        self.screen.blit(left, (10, 12))

        center = self.font_medium.render(
            f"Step {display_step}/672   Day {day}/7   Slot {step_in_day}/96",
            True, LIGHT_GRAY)
        self.screen.blit(center, (400, 12))

        if self.paused:
            if self._replay_idx is not None:
                speed_text = f"REPLAY {self._replay_idx+1}/{len(self._info_history)}"
                speed_color = ORANGE
            else:
                speed_text = "PAUSED"
                speed_color = YELLOW
        else:
            speed_text = f"x{self.speed}"
            speed_color = GREEN
        right = self.font_medium.render(speed_text, True, speed_color)
        self.screen.blit(right, (self.WIDTH - 250, 12))

    def _draw_battery(self, soc: float, net_mw: float):
        """I draw the battery gauge on the left side."""
        # I draw battery panel background
        panel_x, panel_y = 15, 55
        panel_w, panel_h = 180, 340
        pygame.draw.rect(self.screen, BG_PANEL,
                         (panel_x, panel_y, panel_w, panel_h), border_radius=8)

        # I draw title
        title = self.font_medium.render("BATTERY", True, WHITE)
        self.screen.blit(title, (panel_x + 45, panel_y + 10))

        # I draw battery outline
        batt_x = panel_x + 45
        batt_y = panel_y + 45
        batt_w = 90
        batt_h = 200

        # I draw battery terminal (top nub)
        pygame.draw.rect(self.screen, GRAY,
                         (batt_x + 30, batt_y - 8, 30, 10), border_radius=3)

        # I draw battery body outline
        pygame.draw.rect(self.screen, LIGHT_GRAY,
                         (batt_x, batt_y, batt_w, batt_h), 2, border_radius=6)

        # I fill battery level with gradient color
        fill_h = int(soc * (batt_h - 6))
        if soc > 0.6:
            fill_color = GREEN
        elif soc > 0.3:
            fill_color = YELLOW
        elif soc > 0.15:
            fill_color = ORANGE
        else:
            fill_color = RED

        if fill_h > 2:
            pygame.draw.rect(self.screen, fill_color,
                             (batt_x + 3, batt_y + batt_h - 3 - fill_h,
                              batt_w - 6, fill_h),
                             border_radius=4)

        # I draw SoC percentage text centered on battery
        soc_text = self.font_large.render(f"{soc*100:.0f}%", True, WHITE)
        text_rect = soc_text.get_rect(center=(batt_x + batt_w//2,
                                              batt_y + batt_h//2))
        self.screen.blit(soc_text, text_rect)

        # I draw charge/discharge indicator
        y_ind = batt_y + batt_h + 15
        if net_mw > 0.5:
            # Discharging (selling)
            indicator = self.font_medium.render(f"SELL {net_mw:.1f} MW", True, RED)
            arrow = self.font_large.render("▼", True, RED)
        elif net_mw < -0.5:
            # Charging (buying)
            indicator = self.font_medium.render(f"BUY {abs(net_mw):.1f} MW", True, GREEN)
            arrow = self.font_large.render("▲", True, GREEN)
        else:
            indicator = self.font_medium.render("IDLE", True, GRAY)
            arrow = self.font_large.render("─", True, GRAY)

        self.screen.blit(arrow, (batt_x + 30, y_ind))
        self.screen.blit(indicator, (panel_x + 15, y_ind + 35))

        # I draw capacity info
        mwh = soc * 146
        cap_text = self.font_small.render(f"{mwh:.0f} / 146 MWh", True, LIGHT_GRAY)
        self.screen.blit(cap_text, (panel_x + 30, y_ind + 60))

    def _draw_price_chart(self, display_step: int = 0):
        """I draw the scrolling price chart with replay cursor."""
        chart_x, chart_y = 210, 55
        chart_w, chart_h = 640, 220

        # I draw chart background
        pygame.draw.rect(self.screen, BG_CHART,
                         (chart_x, chart_y, chart_w, chart_h), border_radius=6)

        title = self.font_medium.render("DAM PRICE (EUR/MWh)", True, WHITE)
        self.screen.blit(title, (chart_x + 10, chart_y + 5))

        # I draw legend: actual (cyan) vs forecast (orange)
        legend_x = chart_x + chart_w - 200
        pygame.draw.line(self.screen, CYAN, (legend_x, chart_y + 12), (legend_x + 20, chart_y + 12), 2)
        self.screen.blit(self.font_tiny.render("Actual", True, CYAN), (legend_x + 25, chart_y + 6))
        pygame.draw.line(self.screen, ORANGE, (legend_x + 80, chart_y + 12), (legend_x + 100, chart_y + 12), 1)
        self.screen.blit(self.font_tiny.render("Forecast", True, ORANGE), (legend_x + 105, chart_y + 6))

        if len(self.price_history) < 2:
            return

        # I compute chart area
        cx = chart_x + 50
        cy = chart_y + 30
        cw = chart_w - 65
        ch = chart_h - 50

        prices = self.price_history
        p_min = max(0, min(prices) - 10)
        p_max = max(prices) + 10
        p_range = max(p_max - p_min, 1)

        # I draw grid lines and price axis
        for i in range(5):
            price_val = p_min + (p_range * i / 4)
            y = cy + ch - int((price_val - p_min) / p_range * ch)
            pygame.draw.line(self.screen, DARK_GRAY, (cx, y), (cx + cw, y))
            label = self.font_tiny.render(f"{price_val:.0f}", True, LIGHT_GRAY)
            self.screen.blit(label, (cx - 35, y - 6))

        # I draw mean price line
        mean_p = np.mean(prices)
        mean_y = cy + ch - int((mean_p - p_min) / p_range * ch)
        pygame.draw.line(self.screen, PURPLE, (cx, mean_y), (cx + cw, mean_y), 1)

        # I draw price line
        n = len(prices)
        points = []
        for i, p in enumerate(prices):
            x = cx + int(i / max(n - 1, 1) * cw)
            y = cy + ch - int((p - p_min) / p_range * ch)
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(self.screen, CYAN, False, points, 2)

        # I draw forecast line (dashed orange) if available
        if len(self.forecast_history) >= 2:
            forecast_points = []
            for i, fp in enumerate(self.forecast_history):
                if i < n:
                    x = cx + int(i / max(n - 1, 1) * cw)
                    y = cy + ch - int((fp - p_min) / p_range * ch)
                    y = max(cy, min(cy + ch, y))
                    forecast_points.append((x, y))
            # I draw dashed line by drawing segments with gaps
            if len(forecast_points) >= 2:
                for i in range(0, len(forecast_points) - 1, 2):
                    end = min(i + 1, len(forecast_points) - 1)
                    pygame.draw.line(self.screen, ORANGE,
                                     forecast_points[i], forecast_points[end], 1)

        # I draw current position dot (replay-aware)
        # During replay, I highlight the replay position; otherwise the last point
        if self._replay_idx is not None and display_step > 0:
            # I compute which index in price_history corresponds to display_step
            # price_history stores the last max_history items, so:
            history_offset = display_step - 1 - max(0, self.step_count - len(self.price_history))
            cursor_idx = max(0, min(len(points) - 1, history_offset))
            cursor_pt = points[cursor_idx]
            # I draw vertical cursor line
            pygame.draw.line(self.screen, YELLOW,
                             (cursor_pt[0], cy), (cursor_pt[0], cy + ch), 1)
            pygame.draw.circle(self.screen, WHITE, cursor_pt, 6)
            price_label = self.font_small.render(
                f"{prices[cursor_idx]:.1f}", True, YELLOW)
            self.screen.blit(price_label, (cursor_pt[0] + 8, cursor_pt[1] - 8))
        elif points:
            last = points[-1]
            pygame.draw.circle(self.screen, WHITE, last, 5)
            price_label = self.font_small.render(
                f"{prices[-1]:.1f}", True, YELLOW)
            self.screen.blit(price_label, (last[0] + 8, last[1] - 8))

        # I shade action areas (buy = green bottom, sell = red top)
        for i, (mw, p) in enumerate(zip(self.action_history, prices)):
            x = cx + int(i / max(n - 1, 1) * cw)
            y = cy + ch - int((p - p_min) / p_range * ch)
            if mw > 1.0:  # selling
                col = (*RED_DARK, 60)
                s = pygame.Surface((3, y - cy), pygame.SRCALPHA)
                s.fill(col)
                self.screen.blit(s, (x - 1, cy))
            elif mw < -1.0:  # buying
                col = (*GREEN_DARK, 60)
                s = pygame.Surface((3, cy + ch - y), pygame.SRCALPHA)
                s.fill(col)
                self.screen.blit(s, (x - 1, y))

    def _draw_action_panel(self, info: Dict):
        """I draw the action detail panel (bottom-left)."""
        panel_x, panel_y = 15, 405
        panel_w, panel_h = 180, 285
        pygame.draw.rect(self.screen, BG_PANEL,
                         (panel_x, panel_y, panel_w, panel_h), border_radius=8)

        title = self.font_medium.render("DECISIONS", True, WHITE)
        self.screen.blit(title, (panel_x + 35, panel_y + 8))

        y = panel_y + 35
        line_h = 22

        # I show market breakdown (all markets except IDA3)
        markets = [
            ("DAM", info.get('dam_commitment_mw', 0), info.get('step_dam_revenue', 0)),
            ("IDA1", info.get('ida1_mw', 0), info.get('step_ida1_revenue', 0)),
            ("IDA2", info.get('ida2_mw', 0), info.get('step_ida2_revenue', 0)),
            ("aFRR", info.get('afrr_commitment_mw', 0), info.get('step_afrr_cap_revenue', 0)),
            ("XBID", info.get('xbid_mw', 0), info.get('step_xbid_revenue', 0)),
            ("mFRR", info.get('mfrr_auto_mw', 0), info.get('step_mfrr_revenue', 0)),
        ]

        for name, mw, rev in markets:
            color = GREEN if rev > 0.5 else (RED if rev < -0.5 else GRAY)
            mw_text = f"{mw:+.1f}" if abs(mw) > 0.1 else "  --"
            line = self.font_small.render(f"{name:<5}{mw_text:>6}MW", True, color)
            self.screen.blit(line, (panel_x + 10, y))

            rev_text = f"{rev:+.0f}" if abs(rev) > 0.5 else ""
            rev_surf = self.font_small.render(rev_text, True, color)
            self.screen.blit(rev_surf, (panel_x + 120, y))
            y += line_h

        y += 10
        pygame.draw.line(self.screen, GRAY, (panel_x + 10, y), (panel_x + panel_w - 10, y))
        y += 8

        # I show step profit
        step_profit = info.get('step_profit', 0)
        profit_color = GREEN if step_profit > 0 else RED
        profit_text = self.font_medium.render(
            f"Step: {step_profit:+.0f} EUR", True, profit_color)
        self.screen.blit(profit_text, (panel_x + 10, y))
        y += 28

        # I show delivery ratio
        dr = info.get('delivery_ratio', 1.0)
        dr_color = GREEN if dr > 0.99 else (YELLOW if dr > 0.5 else RED)
        dr_text = self.font_small.render(f"Delivery: {dr*100:.0f}%", True, dr_color)
        self.screen.blit(dr_text, (panel_x + 10, y))
        y += line_h

        # I show imbalance
        imb = info.get('step_imbalance_cost', 0)
        if imb > 0.5:
            imb_text = self.font_small.render(f"Imbalance: -{imb:.0f}", True, RED)
            self.screen.blit(imb_text, (panel_x + 10, y))
        y += line_h

        # I show cycles
        cycles = info.get('daily_cycles', 0)
        cyc_text = self.font_small.render(f"Cycles: {cycles:.2f}/day", True, LIGHT_GRAY)
        self.screen.blit(cyc_text, (panel_x + 10, y))

    def _draw_pnl_panel(self, info: Dict, display_step: int = 0):
        """I draw the P&L summary panel (bottom-right)."""
        panel_x, panel_y = 210, 285
        panel_w, panel_h = 640, 405
        pygame.draw.rect(self.screen, BG_PANEL,
                         (panel_x, panel_y, panel_w, panel_h), border_radius=8)

        title = self.font_medium.render("P&L TRACKER", True, WHITE)
        self.screen.blit(title, (panel_x + 10, panel_y + 8))

        # I draw cumulative P&L bars
        y = panel_y + 38
        bar_w = 250

        markets = [
            ("DAM", info.get('dam_profit', 0), BLUE),
            ("IDA1+IDA2", info.get('ida_profit', 0), (100, 180, 255)),
            ("aFRR capacity", info.get('afrr_capacity_profit', 0), PURPLE),
            ("aFRR energy", info.get('afrr_energy_profit', 0), CYAN),
            ("XBID", info.get('intraday_profit', 0), YELLOW),
            ("mFRR", info.get('mfrr_profit', 0), ORANGE),
        ]

        max_val = max(abs(m[1]) for m in markets) if markets else 1
        max_val = max(max_val, 100)

        for name, value, color in markets:
            # I draw label
            label = self.font_small.render(f"{name:<14}", True, LIGHT_GRAY)
            self.screen.blit(label, (panel_x + 15, y + 2))

            # I draw bar
            bar_x = panel_x + 140
            bar_center = bar_x + bar_w // 2

            # I draw center line (zero)
            pygame.draw.line(self.screen, GRAY,
                             (bar_center, y), (bar_center, y + 18))

            if abs(value) > 0.5:
                bar_len = int(value / max_val * (bar_w // 2))
                bar_len = max(-bar_w // 2, min(bar_w // 2, bar_len))

                if bar_len > 0:
                    pygame.draw.rect(self.screen, color,
                                     (bar_center, y + 2, bar_len, 14),
                                     border_radius=3)
                elif bar_len < 0:
                    pygame.draw.rect(self.screen, color,
                                     (bar_center + bar_len, y + 2, -bar_len, 14),
                                     border_radius=3)

            # I draw value
            val_text = self.font_small.render(f"{value:+,.0f}", True, WHITE)
            self.screen.blit(val_text, (bar_x + bar_w + 10, y + 2))

            y += 24

        # I draw total
        y += 5
        pygame.draw.line(self.screen, WHITE,
                         (panel_x + 15, y), (panel_x + panel_w - 15, y))
        y += 8

        total = info.get('total_profit', 0)
        total_color = GREEN if total > 0 else RED
        total_text = self.font_large.render(f"TOTAL: {total:+,.0f} EUR", True, total_color)
        self.screen.blit(total_text, (panel_x + 15, y))
        y += 35

        # I compute EUR/day using display_step (replay-aware)
        effective_steps = display_step if display_step > 0 else self.step_count
        if effective_steps > 0:
            eur_day = total / (effective_steps / 96)
            eur_year = eur_day * 365
            daily_text = self.font_medium.render(
                f"{eur_day:+,.0f} EUR/day  =  {eur_year/1e6:+,.2f}M EUR/year",
                True, total_color)
            self.screen.blit(daily_text, (panel_x + 15, y))

        # I draw mini SoC chart (bottom of panel)
        if len(self.soc_history) > 2:
            chart_y = panel_y + panel_h - 130
            chart_h = 110
            chart_x = panel_x + 15
            chart_w = panel_w - 30

            label = self.font_small.render("SoC Trajectory", True, LIGHT_GRAY)
            self.screen.blit(label, (chart_x, chart_y - 15))

            # I draw SoC chart background
            pygame.draw.rect(self.screen, BG_CHART,
                             (chart_x, chart_y, chart_w, chart_h), border_radius=4)

            # I draw 5%/95% limit lines
            for limit, label_txt in [(0.05, "5%"), (0.95, "95%")]:
                ly = chart_y + chart_h - int(limit * chart_h)
                pygame.draw.line(self.screen, RED_DARK,
                                 (chart_x, ly), (chart_x + chart_w, ly), 1)

            # I draw 50% line
            mid_y = chart_y + chart_h // 2
            pygame.draw.line(self.screen, DARK_GRAY,
                             (chart_x, mid_y), (chart_x + chart_w, mid_y), 1)

            # I draw SoC line
            n = len(self.soc_history)
            points = []
            for i, s in enumerate(self.soc_history):
                x = chart_x + int(i / max(n - 1, 1) * chart_w)
                y_pos = chart_y + chart_h - int(s * chart_h)
                points.append((x, y_pos))

            if len(points) >= 2:
                pygame.draw.lines(self.screen, GREEN, False, points, 2)

            # I draw replay cursor on SoC chart too
            if self._replay_idx is not None and display_step > 0:
                history_offset = display_step - 1 - max(0, self.step_count - len(self.soc_history))
                cursor_idx = max(0, min(len(points) - 1, history_offset))
                cp = points[cursor_idx]
                pygame.draw.line(self.screen, YELLOW,
                                 (cp[0], chart_y), (cp[0], chart_y + chart_h), 1)
                pygame.draw.circle(self.screen, WHITE, cp, 4)

    def _draw_gate_events(self):
        """I draw IDA gate closure notifications (fade out over time)."""
        y = 55
        remaining = []
        for evt in self._gate_events:
            alpha = min(255, evt['ttl'] * 8)
            text = self.font_medium.render(f">> {evt['text']} <<", True, evt['color'])
            # I center the notification on the price chart area
            x = 210 + 320 - text.get_width() // 2
            self.screen.blit(text, (x, y))
            y += 25
            evt['ttl'] -= 1
            if evt['ttl'] > 0:
                remaining.append(evt)
        self._gate_events = remaining

    def _draw_controls(self):
        """I draw control hints at the bottom."""
        y = self.HEIGHT - 18
        controls = "SPACE=Pause  +/-=Speed  R=Reset  Q=Quit"
        if self.paused:
            controls += "  LEFT/RIGHT=Step"
        text = self.font_tiny.render(controls, True, GRAY)
        self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, y))

    def close(self):
        """I close the pygame window."""
        pygame.quit()

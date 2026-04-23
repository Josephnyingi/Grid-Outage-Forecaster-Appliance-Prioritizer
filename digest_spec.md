# digest_spec.md — Product & Business Adaptation Artifact
## T2.3 Grid Outage Forecaster + Appliance Prioritizer

---

## 1. Morning SMS Digest — Salon Owner (Feature Phone)

The salon owner receives up to **3 SMS messages at 06:30** each morning, before opening at 07:00.
Each SMS is ≤ 160 characters (standard SMS limit, works on any feature phone, no data required).

### Format Specification

```
SMS 1/3 — Overview
KEZA SALON {DD Mon}: Grid risk {LEVEL} today. Peak risk {HH}h ({P}%).
Expected rev: {REV} RWF.
```
> Max 160 chars. Sent always. Tells owner the day's overall risk level and revenue expectation.

```
SMS 2/3 — Action (sent only if risk > LOW in any hour)
{HH}h ACTION: Turn OFF {LUXURY_LIST}. Keep {CRITICAL_LIST} ON.
Risk clears by {CLEAR_HH}h.
```
> Max 160 chars. Sent only when a load-shed action is needed. Names what to switch off and when it's safe again.

```
SMS 3/3 — Contingency (sent only if any hour p_outage ≥ 0.15)
If power cuts: est. {DUR}min outage. Priority: {TOP_CRITICAL} 1st when
restored. Reply HELP for support.
```
> Max 160 chars. Sent only on elevated-risk days. Gives contingency guidance.

### Example — Today (2024-06-29, Salon)

| SMS | Content | Chars |
|-----|---------|-------|
| 1/3 | `KEZA SALON 29 Jun: Grid risk LOW today. Peak risk 08h (11%). Expected rev: 195,522 RWF.` | 88 |
| 2/3 | `08h ACTION: Turn OFF TV+Sound+Decor light. Keep Dryer+Fan+LED+POS ON. Risk clears by 09h.` | 91 |
| 3/3 | `If power cuts: est. 76min outage. Priority: Hair dryer 1st when restored. Reply HELP for support.` | 98 |

All 3 SMS are under 160 characters and use no internet on the recipient's phone.

---

## 2. Mid-Day Internet Drop Scenario

**Scenario:** Internet connection drops at 13:00. The forecast cannot refresh.

### What the device shows

1. **Stale banner**: A persistent yellow banner appears:
   ```
   ⚠ PLAN LAST UPDATED 06:30 — 6h30m AGO. MAY BE STALE.
   ```
2. **Staleness budget**: The system trusts the cached plan for up to **4 hours** after the last successful refresh.
   - Rationale: The DGP parameters (load patterns, rain) change slowly; a 4h-old plan has ~85% forecast validity.
   - After 4 hours, the model degrades because rain and load features drift.

3. **After 4h (10:30)**: The device switches to **SAFE MODE**:
   - All non-critical appliances shown as OFF in the UI.
   - The LED relay board turns AMBER (medium risk).
   - An SMS is sent to the owner: `FORECAST STALE 4h+. Switching to SAFE MODE: keep critical only. Check connection.`

4. **Risk budget**: Staleness budget of 4h was chosen because:
   - Outage correlation decays with ~4h half-life in the synthetic DGP.
   - Beyond 4h, a naïve "always caution" policy beats the stale forecast in expected Brier contribution.

---

## 3. Accessibility Adaptation — Non-Reading Users

**Design choice: Coloured LEDs on a relay board** (justified below).

### Option selected: LED relay board with 3 colours

The lite_ui.html is designed for smartphones, but many SME owners use feature phones or
are not literate. We add a hardware relay board (Raspberry Pi Zero or Arduino Nano, ~$8)
connected to 3 LEDs and relay switches:

| LED Colour | Risk Level | Appliances |
|-----------|------------|-----------|
| 🟢 GREEN | Low (< 10%) | All ON |
| 🟡 AMBER | Medium (10–25%) | Critical + Comfort ON; Luxury OFF |
| 🔴 RED | High (≥ 25%) | Critical only |

The board receives a JSON payload via local Bluetooth or USB from the owner's phone
(no internet required after morning sync). The relay physically cuts power to luxury-tier
circuits when AMBER or RED.

**Why LEDs over voice or icons:**
- Voice requires audio hardware and literacy in the language spoken.
- Icons on lite_ui require a smartphone screen.
- LEDs + relay board work without a screen, without reading, without internet after morning sync.
  The owner sees green/amber/red at a glance — universal and illiteracy-proof.
  The relay board also acts as an automatic switch, removing the need for the owner to act.

---

## 4. Revenue Calculation — Salon vs. Naïve Full-On Operation

### Scenario: Typical outage week (4% base rate → ~6.7 outage-hours/week)

| Metric | Naïve (all on) | Our Plan | Saving |
|--------|---------------|---------|--------|
| Outage hours/week | ~7 | ~7 | — |
| Revenue lost to outages | ~7 × 7,600 = 53,200 RWF | ~7 × 7,600 × 0.9 = 47,880 RWF | 5,320 RWF |
| Revenue from shedding (avoids wasted startup spikes) | 0 | +2,100 RWF | +2,100 RWF |
| **Net weekly benefit** | — | — | **~7,420 RWF/week** |

> Salon total revenue rate = 7,600 RWF/h (Hair Dryer 6,000 + POS 1,000 + Lighting 800 + Fan 500 + Luxury 300).
> During outage windows, our plan sheds luxury (600 RWF/h) but avoids startup spikes for non-critical appliances,
> netting approximately 7,420 RWF/week saved (~30,000 RWF/month).

---

## 5. Neighbour Signal (Stretch Goal)

When 2 or more businesses within a 500m radius report actual outages via SMS in the last 30 minutes,
the system re-ranks the current hour to **HIGH RISK** regardless of the model's P(outage).
This crowd signal captures fault propagation that the load/weather features cannot see.

Implementation: a simple counter on the SMS gateway; no additional ML required.

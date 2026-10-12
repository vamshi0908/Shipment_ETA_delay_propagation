# Example per-shipment explanations (held-out shipments)

Delayed = final delay > 12 h. 'Main drivers' = additive Ridge contributions relative to the average shipment, grouped into human-readable causes. 'Root cause' = structural propagation split (sums exactly to the actual final delay).

## 1. Same shipment, three checkpoints (SH0233) - the explanation sharpens as data arrives

**SH0233** | Shanghai->Los Angeles | checkpoint `T0_departure`
- Scheduled delivery: 2025-08-02 17:00 | **Predicted ETA: 2025-08-04 04:20 (+35.3 h)** | P(delay > 12 h) = 0.99 | actual delay: +51.3 h
- Average shipment delay (model baseline): +14.2 h. Main drivers of this prediction:
  - Origin departure delay: +21.9 h
  - Route baseline (historical): +4.4 h
  - Sea weather / transit: -2.1 h
  - Weekend clearance risk: -2.0 h
  - Peak season (Q4): -0.8 h

**SH0233** | Shanghai->Los Angeles | checkpoint `T1_port_arrival`
- Scheduled delivery: 2025-08-02 17:00 | **Predicted ETA: 2025-08-04 09:29 (+40.5 h)** | P(delay > 12 h) = 1.00 | actual delay: +51.3 h
- Average shipment delay (model baseline): +14.2 h. Main drivers of this prediction:
  - Congestion x late arrival: +9.6 h
  - Late port arrival: +9.5 h
  - Port congestion: +4.6 h
  - Missed customs slot (>12h late): +2.9 h
  - Origin departure delay: +1.8 h

**SH0233** | Shanghai->Los Angeles | checkpoint `T2_customs_clear`
- Scheduled delivery: 2025-08-02 17:00 | **Predicted ETA: 2025-08-04 19:45 (+50.8 h)** | P(delay > 12 h) = 1.00 | actual delay: +51.3 h
- Average shipment delay (model baseline): +14.2 h. Main drivers of this prediction:
  - Customs stage delay: +10.3 h
  - Late port arrival: +8.5 h
  - Customs slower than schedule: +7.6 h
  - Weekend clearance risk: +7.2 h
  - Missed customs slot (>12h late): +3.4 h

## 2. Other held-out shipments at the customs checkpoint

**SH0014** | Mumbai->Singapore | checkpoint `T2_customs_clear`
- Scheduled delivery: 2025-08-03 03:00 | **Predicted ETA: 2025-08-03 19:37 (+16.6 h)** | P(delay > 12 h) = 0.88 | actual delay: +12.5 h
- Average shipment delay (model baseline): +14.2 h. Main drivers of this prediction:
  - Weekend clearance risk: +7.2 h
  - Customs slower than schedule: -3.0 h
  - Customs stage delay: -1.5 h
  - Port congestion: +1.4 h
  - Congestion x processing time: -1.3 h

**SH0037** | Shanghai->Rotterdam | checkpoint `T2_customs_clear`
- Scheduled delivery: 2026-01-15 21:00 | **Predicted ETA: 2026-01-15 20:46 (-0.2 h)** | P(delay > 12 h) = 0.00 | actual delay: -5.7 h
- Average shipment delay (model baseline): +14.2 h. Main drivers of this prediction:
  - Customs stage delay: -5.2 h
  - Customs slower than schedule: -4.0 h
  - Late port arrival: -3.7 h
  - Weekend clearance risk: -2.9 h
  - Port congestion: +1.9 h

## 3. Root-cause propagation breakdown (actual delays, exact)

- **SH0233** (Shanghai->Los Angeles) final delay +51.3 h = baseline +13.9 h; Origin departure: +25.6 h, Sea transit / port arrival: +0.4 h, Customs & port processing: +4.9 h, Last mile: +6.6 h
- **SH0014** (Mumbai->Singapore) final delay +12.5 h = baseline +13.9 h; Origin departure: -4.1 h, Sea transit / port arrival: +6.7 h, Customs & port processing: -6.9 h, Last mile: +2.9 h
- **SH0037** (Shanghai->Rotterdam) final delay -5.7 h = baseline +13.9 h; Origin departure: -2.9 h, Sea transit / port arrival: -8.7 h, Customs & port processing: -3.7 h, Last mile: -4.2 h
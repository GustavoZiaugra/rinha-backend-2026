import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '5s', target: 5 },    // warmup
    { duration: '10s', target: 10 },  // ramp up
    { duration: '10s', target: 20 },  // peak
    { duration: '5s', target: 0 },    // cooldown
  ],
  thresholds: {
    http_req_duration: ['p(99) < 2000'], // must be under 2s
  },
  noConnectionReuse: false,
};

const payload = JSON.stringify({
  id: "tx-bench-1",
  transaction: { amount: 41.12, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
  customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ["MERC-003", "MERC-016"] },
  merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
  terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
  last_transaction: null
});

// A few different payloads for variety
const payloads = [
  payload,
  JSON.stringify({
    id: "tx-bench-2",
    transaction: { amount: 9505.97, installments: 10, requested_at: "2026-03-14T05:15:12Z" },
    customer: { avg_amount: 81.28, tx_count_24h: 20, known_merchants: ["MERC-008", "MERC-007", "MERC-005"] },
    merchant: { id: "MERC-068", mcc: "7802", avg_amount: 54.86 },
    terminal: { is_online: false, card_present: true, km_from_home: 952.27 },
    last_transaction: null
  }),
  JSON.stringify({
    id: "tx-bench-3",
    transaction: { amount: 150.00, installments: 1, requested_at: "2026-03-12T10:30:00Z" },
    customer: { avg_amount: 200.0, tx_count_24h: 5, known_merchants: ["MERC-001", "MERC-002"] },
    merchant: { id: "MERC-001", mcc: "5411", avg_amount: 180.0 },
    terminal: { is_online: true, card_present: false, km_from_home: 5.0 },
    last_transaction: { timestamp: "2026-03-12T09:00:00Z", km_from_current: 2.0 }
  }),
];

const headers = { 'Content-Type': 'application/json' };

export default function () {
  const p = payloads[Math.floor(Math.random() * payloads.length)];
  const res = http.post('http://localhost:8080/fraud-score', p, { headers });
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has approved field': (r) => r.body.includes('approved'),
  });
  errorRate.add(res.status !== 200);
}

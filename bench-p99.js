import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const myTrend = new Trend('req_duration');

export const options = {
  stages: [
    { duration: '5s', target: 3 },
    { duration: '5s', target: 5 },
    { duration: '5s', target: 10 },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(99) < 2000'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
};

const payload = JSON.stringify({
  id: "tx-bench-1",
  transaction: { amount: 41.12, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
  customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ["MERC-003", "MERC-016"] },
  merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
  terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
  last_transaction: null
});

const headers = { 'Content-Type': 'application/json' };

export default function () {
  const res = http.post('http://localhost:8080/fraud-score', payload, { headers });
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
  myTrend.add(res.timings.duration);
}
